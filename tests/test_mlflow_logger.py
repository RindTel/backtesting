"""``mlflow_logger.log_backtest_run``: everything required is logged and can be read back.

Each test uses its own throwaway MLflow store under ``tmp_path``.
"""

import math

import polars as pl
import pytest
from mlflow.tracking import MlflowClient
from polars.testing import assert_frame_equal

from metrics import compute_metrics
from mlflow_logger import BROKEN_RUN_NOTE, log_backtest_run
from runner import TRADE_LOG_SCHEMA, run_backtest
from strategies import BuyAndHold, MovingAverageCrossover
from tests.helpers import CLOSES, COSTS, make_bars

BACKTEST_PARAMS = {
    "symbol": "TEST",
    "source": "binance",
    "start": "2024-01-01",
    "end": "2024-01-21",
    "initial_capital": 10_000,
    "fee_rate": COSTS.fee_rate,
    "slippage_rate": COSTS.slippage_rate,
}


def _backtest(strategy):
    result = run_backtest(make_bars(CLOSES), strategy, COSTS, 10_000)
    return result, compute_metrics(result.equity_curve, result.trades, periods_per_year=252)


def _log(tmp_path, strategy, runner="correct", **overrides) -> tuple[str, MlflowClient]:
    result, metrics = _backtest(strategy)
    uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    kwargs = (
        dict(
            strategy_name=strategy.name,
            strategy_params=strategy.params,
            dataset_id="abcdef0123456789",
            metrics=metrics,
            equity_curve=result.equity_curve,
            trades=result.trades,
            runner=runner,
            backtest_params=BACKTEST_PARAMS,
            tracking_uri=uri,
            artifact_root=(tmp_path / "mlruns").as_uri(),
        )
        | overrides
    )
    return log_backtest_run(**kwargs), MlflowClient(tracking_uri=uri)


def test_logs_params_metrics_tags_and_artifacts(tmp_path) -> None:
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    result, metrics = _backtest(strategy)
    run_id, client = _log(tmp_path, strategy)
    run = client.get_run(run_id)

    assert run.info.status == "FINISHED"
    assert run.info.run_name == "moving_average_crossover"

    # Params: MLflow stores them as strings.
    assert run.data.params == {
        "short_window": "2",
        "long_window": "4",
        **{k: str(v) for k, v in BACKTEST_PARAMS.items()},
        "strategy": "moving_average_crossover",
        "dataset_snapshot_id": "abcdef0123456789",
    }

    # Metrics: every Metrics field, with the drawdown logged as its depth.
    assert run.data.metrics == {
        "total_return": metrics.total_return,
        "cagr": metrics.cagr,
        "sharpe_ratio": metrics.sharpe_ratio,
        "max_drawdown": metrics.max_drawdown.depth,
        "win_rate": metrics.win_rate,
        "num_round_trips": metrics.num_round_trips,
        "num_trades": metrics.num_trades,
    }

    tags = run.data.tags
    assert tags["runner"] == "correct"
    assert tags["dataset_snapshot_id"] == "abcdef0123456789"
    assert tags["max_drawdown_peak"] == metrics.max_drawdown.peak.date().isoformat()
    assert tags["max_drawdown_trough"] == metrics.max_drawdown.trough.date().isoformat()
    assert "mlflow.note.content" not in tags

    # Artifacts: a real PNG, and CSVs that round-trip exactly.
    # Artifacts land under the configured root, one subdirectory per experiment.
    assert run.info.artifact_uri.startswith((tmp_path / "mlruns" / "backtests").as_uri())
    assert sorted(a.path for a in client.list_artifacts(run_id)) == [
        "equity_curve.csv",
        "equity_curve.png",
        "trades.csv",
    ]
    local = tmp_path / "download"
    local.mkdir()
    png = client.download_artifacts(run_id, "equity_curve.png", str(local))
    with open(png, "rb") as f:
        assert f.read(8) == b"\x89PNG\r\n\x1a\n"

    trades_csv = client.download_artifacts(run_id, "trades.csv", str(local))
    assert result.trades.height > 0
    assert_frame_equal(pl.read_csv(trades_csv, schema=TRADE_LOG_SCHEMA), result.trades)

    equity_csv = client.download_artifacts(run_id, "equity_curve.csv", str(local))
    assert_frame_equal(pl.read_csv(equity_csv, schema=result.equity_curve.schema), result.equity_curve)


def test_broken_runs_are_tagged_named_and_filterable(tmp_path) -> None:
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    correct_id, client = _log(tmp_path, strategy, runner="correct")
    broken_id, _ = _log(tmp_path, strategy, runner="broken")

    broken = client.get_run(broken_id)
    assert broken.data.tags["runner"] == "broken"
    assert broken.data.tags["mlflow.note.content"] == BROKEN_RUN_NOTE
    assert broken.info.run_name == "BROKEN-moving_average_crossover"

    # Both runs share one experiment and are separated by the tag.
    exp_id = broken.info.experiment_id
    assert client.get_run(correct_id).info.experiment_id == exp_id
    only_correct = client.search_runs([exp_id], filter_string="tags.runner = 'correct'")
    only_broken = client.search_runs([exp_id], filter_string="tags.runner = 'broken'")
    assert [r.info.run_id for r in only_correct] == [correct_id]
    assert [r.info.run_id for r in only_broken] == [broken_id]


def test_undefined_metrics_are_logged_as_nan(tmp_path) -> None:
    """Buy-and-hold never closes a trade, so win rate is NaN. It must still log."""
    run_id, client = _log(tmp_path, BuyAndHold())
    run = client.get_run(run_id)
    assert math.isnan(run.data.metrics["win_rate"])
    assert run.data.metrics["num_round_trips"] == 0


def test_param_name_collision_is_rejected(tmp_path) -> None:
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    with pytest.raises(ValueError, match="duplicate"):
        _log(tmp_path, strategy, backtest_params={**BACKTEST_PARAMS, "short_window": 5})


def test_unknown_runner_is_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="runner"):
        _log(tmp_path, BuyAndHold(), runner="leaky")
