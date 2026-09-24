"""Run one backtest on stored data, compute metrics, and log everything to MLflow.

Usage (from the project root). ``--data-root data/sample`` uses the committed
BTCUSDT 2020-2024 sample; otherwise run ``python -m scripts.fetch_data ...`` first:
    python -m scripts.run_backtest --source binance --symbol BTCUSDT \\
        --start 2020-01-01 --end 2024-12-31 --strategy moving_average_crossover \\
        --param short_window=20 --param long_window=50 --data-root data/sample

    # Look-ahead demo: the same run through the DELIBERATELY BROKEN runner,
    # logged with the MLflow tag runner=broken (real runs: runner=correct).
    python -m scripts.run_backtest ... --runner broken

This never touches the network. It reads only what ``fetch_data`` stored.
The full results (params, metrics, equity chart, trade log) go to MLflow, and
the console just gets the run id and a one-line summary. View them with:
    mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
"""

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any, NamedTuple

from config import (
    DEFAULT_FEE_RATE,
    DEFAULT_IMPACT_COEFFICIENT,
    DEFAULT_INITIAL_CAPITAL,
    DEFAULT_MAX_PARTICIPATION,
    DEFAULT_SLIPPAGE_RATE,
    PERIODS_PER_YEAR,
    PROCESSED_DATA_DIR,
)
from data_loader import FETCHERS, LoadedDataset, load
from metrics import Metrics, compute_metrics, excess_return, split_metrics
from mlflow_logger import flatten_metrics, log_backtest_run
from runner import BacktestResult, CostModel, run_backtest
from strategies import STRATEGIES, BuyAndHold

UI_COMMAND = "mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns"


def parse_param(text: str) -> tuple[str, Any]:
    """Parse ``KEY=VALUE`` into ``(key, value)``. The value becomes an int, else a float, else a str."""
    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE, got {text!r}")
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    return key, raw


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one backtest and log it to MLflow.")
    parser.add_argument("--source", required=True, choices=sorted(FETCHERS))
    parser.add_argument("--symbol", required=True, help="Binance spot pair, e.g. BTCUSDT")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--strategy", required=True, choices=sorted(STRATEGIES))
    parser.add_argument(
        "--param",
        action="append",
        default=[],
        type=parse_param,
        metavar="KEY=VALUE",
        help="strategy parameter, repeatable (e.g. --param short_window=20)",
    )
    parser.add_argument(
        "--runner",
        choices=("correct", "broken"),
        default="correct",
        help="'broken' uses the DELIBERATELY BROKEN look-ahead runner (demo only)",
    )
    parser.add_argument("--fee-rate", type=float, default=DEFAULT_FEE_RATE, help="fraction of notional, > 0")
    parser.add_argument("--slippage-rate", type=float, default=DEFAULT_SLIPPAGE_RATE, help="fraction of price, > 0")
    parser.add_argument("--capital", type=float, default=DEFAULT_INITIAL_CAPITAL, help="starting cash")
    parser.add_argument(
        "--impact-coefficient",
        type=float,
        default=DEFAULT_IMPACT_COEFFICIENT,
        help="square-root market impact constant, 0 disables (default: %(default)s)",
    )
    parser.add_argument(
        "--max-participation",
        type=float,
        default=DEFAULT_MAX_PARTICIPATION,
        help="max fill per bar as a fraction of average daily dollar volume (default: %(default)s)",
    )
    parser.add_argument(
        "--split-date",
        type=date.fromisoformat,
        help="also report metrics before (in-sample) and from (out-of-sample) this YYYY-MM-DD",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="where stored bars live; use data/sample for the committed offline sample (default: data/processed)",
    )
    return parser.parse_args(argv)


class Outcome(NamedTuple):
    run_id: str
    dataset: LoadedDataset
    metrics: Metrics
    benchmark: Metrics  # buy-and-hold on the same data, through the correct runner
    split: tuple[Metrics, Metrics] | None  # (in-sample, out-of-sample) if --split-date
    result: BacktestResult  # equity curve + trade log
    benchmark_result: BacktestResult


def backtest(args: argparse.Namespace, extra_params: dict[str, Any] | None = None) -> Outcome:
    """Run one backtest from parsed ``args`` and log it to MLflow.

    Raises:
        FileNotFoundError: no stored data for the request.
        ValueError / TypeError: invalid strategy params, costs or split date.
    """
    dataset = load(args.symbol, args.source, args.start, args.end, root=args.data_root)
    strategy = STRATEGIES[args.strategy](**dict(args.param))
    costs = CostModel(args.fee_rate, args.slippage_rate, args.impact_coefficient, args.max_participation)

    if args.runner == "broken":
        # Imported only here, so the normal path never loads the broken runner.
        from broken_runner import run_backtest_leaky

        result = run_backtest_leaky(dataset.df, strategy, costs, args.capital)
    else:
        result = run_backtest(dataset.df, strategy, costs, args.capital)

    periods = PERIODS_PER_YEAR[args.source]
    metrics = compute_metrics(result.equity_curve, result.trades, periods)
    split = split_metrics(result.equity_curve, result.trades, args.split_date, periods) if args.split_date else None
    # The benchmark always uses the correct runner, even for a broken-runner run.
    bench = run_backtest(dataset.df, BuyAndHold(), costs, args.capital)
    benchmark = compute_metrics(bench.equity_curve, bench.trades, periods)

    extra = flatten_metrics(benchmark, "benchmark_") | {"excess_return": excess_return(metrics, benchmark)}
    if split:
        extra |= flatten_metrics(split[0], "in_sample_") | flatten_metrics(split[1], "out_of_sample_")
    run_id = log_backtest_run(
        strategy_name=strategy.name,
        strategy_params=strategy.params,
        dataset_id=dataset.dataset_id,
        metrics=metrics,
        equity_curve=result.equity_curve,
        trades=result.trades,
        runner=args.runner,
        extra_metrics=extra,
        backtest_params={
            "symbol": args.symbol,
            "source": args.source,
            "start": str(dataset.start),
            "end": str(dataset.end),
            "initial_capital": args.capital,
            "fee_rate": costs.fee_rate,
            "slippage_rate": costs.slippage_rate,
            "impact_coefficient": costs.impact_coefficient,
            "max_participation": costs.max_participation,
        }
        | ({"split_date": str(args.split_date)} if split else {})
        | (extra_params or {}),
    )
    return Outcome(run_id, dataset, metrics, benchmark, split, result, bench)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        run_id, dataset, metrics, benchmark, split, *_ = backtest(args)
    except FileNotFoundError as e:
        print(
            f"error: {e}\nFetch it first:\n  python -m scripts.fetch_data --source {args.source} "
            f"--symbol {args.symbol} --start {args.start} --end {args.end} --data-root {args.data_root}",
            file=sys.stderr,
        )
        return 2
    except (TypeError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    label = "BROKEN RUNNER (look-ahead leak)" if args.runner == "broken" else "correct runner"
    lines = [
        f"[{label}] {args.strategy} on {args.symbol} {dataset.start} → {dataset.end}",
        f"  total return {metrics.total_return:+.1%} | CAGR {metrics.cagr:+.1%} | "
        f"Sharpe {metrics.sharpe_ratio:.2f} | max drawdown {metrics.max_drawdown.depth:.1%} | "
        f"{metrics.num_trades} trades",
        f"  vs buy-and-hold: return {benchmark.total_return:+.1%}, Sharpe {benchmark.sharpe_ratio:.2f}"
        f" | excess return {excess_return(metrics, benchmark):+.1%}",
    ]
    if split:
        ins, oos = split
        lines.append(
            f"  in-sample (before {args.split_date}): return {ins.total_return:+.1%}, Sharpe {ins.sharpe_ratio:.2f}"
            f" | out-of-sample: return {oos.total_return:+.1%}, Sharpe {oos.sharpe_ratio:.2f}"
        )
    lines += [f"  dataset_snapshot_id {dataset.dataset_id} | MLflow run {run_id}", f"  View: {UI_COMMAND}"]
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
