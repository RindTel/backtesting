"""All interaction with MLflow happens here, and only here.

``log_backtest_run`` records one backtest as one MLflow run:
    * params: every strategy param, every backtest param (symbol, dates,
      capital, fee, slippage), the strategy name, and ``dataset_snapshot_id``
    * metrics: every field of ``metrics.Metrics`` (drawdown depth as ``max_drawdown``),
      plus any ``extra_metrics`` (e.g. ``in_sample_*``, ``benchmark_*``, ``excess_return``)
    * tags: ``runner`` (``correct`` | ``broken``), ``dataset_snapshot_id``, and
      the max drawdown's peak/trough/recovery dates
    * artifacts: ``equity_curve.png`` (chart), ``trades.csv`` (full trade log),
      ``equity_curve.csv`` (the raw curve, so every metric can be recomputed)

Tracking is local only: run metadata in a SQLite file (``./mlflow.db``) and
artifacts under ``./mlruns``. There is no server and no account. View it with:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns

Correct and broken runs share one experiment so they can be compared side by
side. Filter them in the UI with ``tags.runner = "correct"`` or ``tags.runner = "broken"``.
Broken runs are also named ``BROKEN-<strategy>``, carry a warning in the run
description, and their chart title says so.

Errors are never swallowed. If anything fails, MLflow marks the run FAILED
and the exception propagates.
"""

import dataclasses
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import mlflow
import polars as pl
from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

from config import MLFLOW_ARTIFACT_ROOT, MLFLOW_EXPERIMENT_NAME, MLFLOW_TRACKING_URI
from metrics import Drawdown, Metrics

Runner = Literal["correct", "broken"]

BROKEN_RUN_NOTE = (
    "DELIBERATELY BROKEN RUNNER: results are inflated by look-ahead bias. Demo only. Never treat these numbers as real."
)


def log_backtest_run(
    *,
    strategy_name: str,
    strategy_params: Mapping[str, Any],
    dataset_id: str,
    metrics: Metrics,
    equity_curve: pl.DataFrame,
    trades: pl.DataFrame,
    runner: Runner = "correct",
    extra_metrics: Mapping[str, float] | None = None,
    backtest_params: Mapping[str, Any] | None = None,
    tracking_uri: str = MLFLOW_TRACKING_URI,
    artifact_root: str = MLFLOW_ARTIFACT_ROOT,
) -> str:
    """Log one complete backtest as a single MLflow run.

    Args:
        strategy_name: e.g. ``"moving_average_crossover"``.
        strategy_params: The strategy's tunable params (``BaseStrategy.params``).
        dataset_id: ``LoadedDataset.dataset_id``, logged as ``dataset_snapshot_id``.
        metrics: The result of ``metrics.compute_metrics``.
        equity_curve: The runner's equity curve.
        trades: The runner's trade log.
        runner: ``"correct"`` for ``runner.py``, ``"broken"`` for ``broken_runner.py``.
        extra_metrics: More numbers to log as-is, e.g. ``flatten_metrics(m, "benchmark_")``.
        backtest_params: Run settings: symbol, source, start, end,
            initial_capital, fee_rate, slippage_rate.
        tracking_uri: Where run metadata lives. Defaults to ``./mlflow.db``.
        artifact_root: Where artifacts of newly created experiments live.
            Defaults to ``./mlruns``, with one subdirectory per experiment.

    Returns:
        The MLflow run id.

    Raises:
        ValueError: on an unknown ``runner``, or if a backtest param name
            collides with a strategy param (MLflow params can't be overwritten).
    """
    if runner not in ("correct", "broken"):
        raise ValueError(f"runner must be 'correct' or 'broken', got {runner!r}")
    backtest_params = backtest_params or {}
    clash = strategy_params.keys() & backtest_params.keys()
    if clash:
        raise ValueError(f"duplicate MLflow param names: {sorted(clash)}")
    params = {**strategy_params, **backtest_params, "strategy": strategy_name, "dataset_snapshot_id": dataset_id}
    drawdown = metrics.max_drawdown
    broken = runner == "broken"

    mlflow.set_tracking_uri(tracking_uri)
    if mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME) is None:
        mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME, artifact_location=f"{artifact_root}/{MLFLOW_EXPERIMENT_NAME}")
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)

    tags = {
        "runner": runner,
        "dataset_snapshot_id": dataset_id,
        "max_drawdown_peak": _iso(drawdown.peak),
        "max_drawdown_trough": _iso(drawdown.trough),
        "max_drawdown_recovery": _iso(drawdown.recovery),
    }
    if broken:
        tags["mlflow.note.content"] = BROKEN_RUN_NOTE  # shown as the run description in the UI

    run_name = f"BROKEN-{strategy_name}" if broken else strategy_name
    with mlflow.start_run(run_name=run_name, tags=tags) as run:
        mlflow.log_params(params)
        mlflow.log_metrics(flatten_metrics(metrics) | dict(extra_metrics or {}))

        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            title = f"{strategy_name}: equity"
            if broken:
                title = f"BROKEN RUNNER (look-ahead leak), {title}"
            _render_equity_curve(equity_curve, drawdown, title, tmp_dir / "equity_curve.png")
            trades.write_csv(tmp_dir / "trades.csv")
            equity_curve.write_csv(tmp_dir / "equity_curve.csv")
            for name in ("equity_curve.png", "trades.csv", "equity_curve.csv"):
                mlflow.log_artifact(str(tmp_dir / name))

        return run.info.run_id


def flatten_metrics(metrics: Metrics, prefix: str = "") -> dict[str, float]:
    """Every numeric field of ``Metrics``. The ``Drawdown`` contributes its depth
    as ``max_drawdown``, and its dates go to tags. New fields added to
    ``Metrics`` get logged automatically. NaN (undefined) values are logged as NaN.
    """
    out: dict[str, float] = {}
    for field in dataclasses.fields(metrics):
        value = getattr(metrics, field.name)
        out[prefix + field.name] = float(value.depth if isinstance(value, Drawdown) else value)
    return out


def _iso(ts: datetime | None) -> str:
    return ts.date().isoformat() if ts is not None else "none"


# --- Equity chart -------------------------------------------------------------
# Single-series line chart, styled after the project's dataviz rules: one 2px
# line in categorical slot 1, text in text colors (never the series color),
# recessive hairline grid, and no legend (the title names the only series).
# The max-drawdown span is a neutral gray band, not a status color.

_SURFACE = "#fcfcfb"
_SERIES = "#2a78d6"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"
_BAND = "#f0efec"


def _render_equity_curve(equity_curve: pl.DataFrame, drawdown: Drawdown, title: str, path: Path) -> None:
    """Draw equity over time to a PNG at ``path``, shading the max-drawdown span.

    Uses matplotlib's object-oriented ``Figure`` (no pyplot), so it needs no
    display and leaves no global state behind.
    """
    ts = equity_curve["timestamp"].to_list()
    equity = equity_curve["equity"].to_list()

    fig = Figure(figsize=(10, 4.5), dpi=150, facecolor=_SURFACE)
    ax = fig.add_subplot()
    ax.set_facecolor(_SURFACE)

    if drawdown.peak is not None and drawdown.trough is not None:
        ax.axvspan(drawdown.peak, drawdown.trough, color=_BAND, zorder=0, linewidth=0)
        ax.annotate(
            f"Max drawdown {drawdown.depth:.1%}",
            xy=(drawdown.peak, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(4, -4),
            textcoords="offset points",
            va="top",
            fontsize=8,
            color=_TEXT_SECONDARY,
        )

    ax.plot(ts, equity, color=_SERIES, linewidth=2, solid_joinstyle="round", solid_capstyle="round", zorder=2)
    ax.annotate(  # value at the line's end, in text color
        f"{equity[-1]:,.0f}",
        xy=(ts[-1], equity[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        va="center",
        fontsize=9,
        color=_TEXT_PRIMARY,
    )

    ax.set_title(title, loc="left", fontsize=11, color=_TEXT_PRIMARY, pad=12)
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(ConciseDateFormatter(locator))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.grid(axis="y", color=_GRID, linewidth=0.8, linestyle="-")
    ax.tick_params(colors=_TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(_GRID)
    ax.margins(x=0.01)

    fig.tight_layout()
    fig.savefig(path, facecolor=_SURFACE)
