"""The CLI scripts, end to end, against a throwaway data store and MLflow store."""

import argparse
from functools import partial

import pytest
from mlflow.tracking import MlflowClient

from data_loader import write_partitioned
from mlflow_logger import log_backtest_run
from scripts import run_backtest, sweep
from tests.helpers import CLOSES, make_bars


@pytest.fixture
def stores(tmp_path, monkeypatch):
    data_root = tmp_path / "processed"
    write_partitioned(make_bars(CLOSES, symbol="TEST"), "binance", root=data_root)
    tracking = f"sqlite:///{tmp_path / 'mlflow.db'}"
    # Point the CLI's MLflow logging at a throwaway store.
    monkeypatch.setattr(
        run_backtest,
        "log_backtest_run",
        partial(log_backtest_run, tracking_uri=tracking, artifact_root=(tmp_path / "mlruns").as_uri()),
    )
    common = [
        "--source", "binance", "--symbol", "TEST", "--start", "2024-01-01", "--end", "2024-01-31",
        "--strategy", "moving_average_crossover", "--param", "short_window=2", "--param", "long_window=4",
        "--data-root", str(data_root),
    ]  # fmt: skip
    return common, MlflowClient(tracking_uri=tracking)


def _only_run(client: MlflowClient, runner: str):
    exp = client.get_experiment_by_name("backtests")
    runs = client.search_runs([exp.experiment_id], filter_string=f"tags.runner = '{runner}'")
    assert len(runs) == 1
    return runs[0]


def test_correct_run_logs_to_mlflow(stores, capsys) -> None:
    args, client = stores
    assert run_backtest.main(args) == 0

    run = _only_run(client, "correct")
    assert run.data.params["short_window"] == "2"
    assert run.data.params["symbol"] == "TEST"
    assert run.data.params["start"] == "2024-01-01"
    assert run.info.run_id in capsys.readouterr().out


def test_broken_runner_flag_is_tagged(stores, capsys) -> None:
    args, client = stores
    assert run_backtest.main([*args, "--runner", "broken"]) == 0

    run = _only_run(client, "broken")
    assert run.info.run_name == "BROKEN-moving_average_crossover"
    assert "BROKEN RUNNER" in capsys.readouterr().out


def test_same_data_gives_same_dataset_id_for_both_runners(stores) -> None:
    args, client = stores
    run_backtest.main(args)
    run_backtest.main([*args, "--runner", "broken"])
    ids = {_only_run(client, r).data.params["dataset_snapshot_id"] for r in ("correct", "broken")}
    assert len(ids) == 1


def test_missing_data_exits_2_with_fetch_hint(stores, capsys) -> None:
    args, _ = stores
    args = [a if a != "TEST" else "NOPE" for a in args]
    assert run_backtest.main(args) == 2
    assert "python -m scripts.fetch_data" in capsys.readouterr().err


def test_invalid_strategy_params_exit_2(stores, capsys) -> None:
    args, _ = stores
    assert run_backtest.main([*args, "--param", "short_window=9"]) == 2  # 9 >= long_window 4
    assert "short_window" in capsys.readouterr().err


def test_zero_fee_is_rejected(stores, capsys) -> None:
    args, _ = stores
    assert run_backtest.main([*args, "--fee-rate", "0"]) == 2
    assert "frictionless" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "expected"),
    [("n=20", ("n", 20)), ("x=0.5", ("x", 0.5)), ("mode=fast", ("mode", "fast")), ("e=", ("e", ""))],
)
def test_parse_param_types(text, expected) -> None:
    assert run_backtest.parse_param(text) == expected
    assert type(run_backtest.parse_param(text)[1]) is type(expected[1])


def test_parse_param_rejects_missing_equals() -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        run_backtest.parse_param("short_window")


def test_split_date_logs_in_and_out_of_sample_metrics(stores, capsys) -> None:
    args, client = stores
    assert run_backtest.main([*args, "--split-date", "2024-01-11"]) == 0

    run = _only_run(client, "correct")
    assert run.data.params["split_date"] == "2024-01-11"
    assert {"in_sample_total_return", "out_of_sample_total_return", "out_of_sample_sharpe_ratio"} <= set(
        run.data.metrics
    )
    assert "out-of-sample" in capsys.readouterr().out


def test_split_date_outside_data_exits_2(stores, capsys) -> None:
    args, _ = stores
    assert run_backtest.main([*args, "--split-date", "2030-01-01"]) == 2
    assert "at least 2 bars" in capsys.readouterr().err


def test_every_run_logs_buy_and_hold_benchmark(stores) -> None:
    args, client = stores
    assert run_backtest.main([*args, "--runner", "broken"]) == 0

    m = _only_run(client, "broken").data.metrics
    assert {"benchmark_total_return", "benchmark_sharpe_ratio"} <= set(m)
    assert m["excess_return"] == pytest.approx(m["total_return"] - m["benchmark_total_return"])


def test_sweep_logs_one_run_per_valid_pair(stores, capsys) -> None:
    args, client = stores
    data_args = args[: args.index("--strategy")] + args[args.index("--data-root") :]
    rc = sweep.main([*data_args, "--split-date", "2024-01-11", "--short", "2,3,4", "--long", "4,5"])
    assert rc == 0

    exp = client.get_experiment_by_name("backtests")
    runs = client.search_runs([exp.experiment_id])
    # (4, 4) is skipped: short must be < long.
    pairs = sorted((r.data.params["short_window"], r.data.params["long_window"]) for r in runs)
    assert pairs == [("2", "4"), ("2", "5"), ("3", "4"), ("3", "5"), ("4", "5")]
    assert len({r.data.params["sweep_id"] for r in runs}) == 1
    assert all("out_of_sample_sharpe_ratio" in r.data.metrics for r in runs)
    assert "Picked by in-sample Sharpe" in capsys.readouterr().out


def test_sweep_requires_split_date(stores) -> None:
    args, _ = stores
    with pytest.raises(SystemExit):
        sweep.main([*args, "--short", "2", "--long", "4"])
