"""Grid-search moving-average windows, honestly: pick in-sample, judge out-of-sample.

Usage (from the project root):
    python -m scripts.sweep --source binance --symbol BTCUSDT \\
        --start 2020-01-01 --end 2024-12-31 --data-root data/sample \\
        --split-date 2023-01-01 --short 5,10,20 --long 30,50,100

Every (short, long) pair becomes one MLflow run, tagged with a shared
``sweep_id`` param (filter with ``params.sweep_id = '<id>'``). The winner is
chosen by IN-SAMPLE Sharpe only, then its OUT-OF-SAMPLE Sharpe is reported
next to its rank among all pairs out of sample. A split date is required:
a sweep without one just finds the setting that best fits the past.
"""

import argparse
import uuid
from collections.abc import Sequence

from scripts.run_backtest import backtest
from scripts.run_backtest import parse_args as parse_backtest_args


def _ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",")]


def run_sweep(shorts: list[int], longs: list[int], split_date: str, backtest_argv: list[str]) -> tuple[str, list[dict]]:
    """Run every (short, long) pair with short < long and log each to MLflow.

    ``backtest_argv`` holds the other ``run_backtest`` flags (source, symbol, dates, costs, data root).

    Returns:
        ``(sweep_id, rows)``, with rows sorted by in-sample Sharpe (best first).
        Each row has ``short``, ``long``, ``in_sample_sharpe``, ``out_of_sample_sharpe``
        and ``oos_rank`` (1 = best out of sample).
    """
    sweep_id = uuid.uuid4().hex[:8]
    rows = []
    for short, long in [(s, lg) for s in shorts for lg in longs if s < lg]:
        ns = parse_backtest_args(
            [*backtest_argv, "--strategy", "moving_average_crossover", "--split-date", split_date,
             "--param", f"short_window={short}", "--param", f"long_window={long}"]
        )  # fmt: skip
        ins, oos = backtest(ns, extra_params={"sweep_id": sweep_id}).split
        rows.append(
            {
                "short": short,
                "long": long,
                "in_sample_sharpe": ins.sharpe_ratio,
                "out_of_sample_sharpe": oos.sharpe_ratio,
            }
        )

    for rank, row in enumerate(sorted(rows, key=lambda r: r["out_of_sample_sharpe"], reverse=True), 1):
        row["oos_rank"] = rank
    rows.sort(key=lambda r: r["in_sample_sharpe"], reverse=True)  # by in-sample Sharpe: the only fair way to choose
    return sweep_id, rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--short", required=True, type=_ints, help="comma-separated short windows, e.g. 5,10,20")
    parser.add_argument("--long", required=True, type=_ints, help="comma-separated long windows, e.g. 30,50,100")
    parser.add_argument("--split-date", required=True, help="YYYY-MM-DD: choose before it, judge from it")
    args, rest = parser.parse_known_args(argv)  # everything else goes straight to run_backtest

    sweep_id, rows = run_sweep(args.short, args.long, args.split_date, rest)
    n = len(rows)
    print(f"sweep {sweep_id}: {n} runs, split at {args.split_date}")
    print(f"{'short':>5} {'long':>5} {'in-sample Sharpe':>17} {'out-of-sample Sharpe':>21} {'OOS rank':>9}")
    for r in rows:
        print(
            f"{r['short']:>5} {r['long']:>5} {r['in_sample_sharpe']:>17.3f} "
            f"{r['out_of_sample_sharpe']:>21.3f} {r['oos_rank']:>6}/{n}"
        )
    pick = rows[0]
    print(
        f"\nPicked by in-sample Sharpe: {pick['short']}/{pick['long']}. Out of sample it ranked "
        f"{pick['oos_rank']} of {n} (Sharpe {pick['out_of_sample_sharpe']:.2f}). "
        "That out-of-sample number is the honest one."
        f"\nMLflow filter: params.sweep_id = '{sweep_id}'"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
