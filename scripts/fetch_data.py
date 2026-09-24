"""Download daily bars once and store them locally as Parquet.

Usage (from the project root):
    python -m scripts.fetch_data --source binance --symbol BTCUSDT --start 2020-01-01 --end 2024-12-31

Fetching is a separate step from backtesting on purpose: ``run_backtest``
never touches the network. Providers can revise history over time, so if
every backtest fetched its own data, two runs meant to use "the same data"
could quietly differ. Fetch once, then every run reads the same stored
snapshot and logs the same ``dataset_snapshot_id``.
"""

import argparse
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from config import PROCESSED_DATA_DIR
from data_loader import FETCHERS, compute_dataset_id, write_partitioned


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", required=True, choices=sorted(FETCHERS))
    parser.add_argument("--symbol", required=True, help="Binance spot pair, e.g. BTCUSDT")
    parser.add_argument("--start", required=True, type=date.fromisoformat, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--end", required=True, type=date.fromisoformat, help="YYYY-MM-DD, inclusive")
    parser.add_argument("--data-root", type=Path, default=PROCESSED_DATA_DIR, help="default: data/processed")
    args = parser.parse_args(argv)

    bars = FETCHERS[args.source](args.symbol, args.start, args.end)
    files = write_partitioned(bars, args.source, root=args.data_root)
    first, last = bars["timestamp"][0].date(), bars["timestamp"][-1].date()
    print(
        f"Stored {bars.height} bars for {args.source}:{args.symbol} ({first} → {last}) "
        f"in {len(files)} partition(s).\n"
        f"dataset_snapshot_id = {compute_dataset_id(bars, args.source, args.symbol)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
