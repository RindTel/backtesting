"""Shared test fixtures: synthetic bars, a standard cost model, a scripted strategy."""

from datetime import date, datetime, timedelta

import polars as pl

from data_loader import CANONICAL_SCHEMA
from runner import CostModel
from strategies.base_strategy import HOLD, BaseStrategy, Signal

COSTS = CostModel(fee_rate=0.001, slippage_rate=0.0005)

# A price path with several SMA(2)/SMA(4) crossovers.
CLOSES = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6, 7, 8, 9, 10, 11, 10, 9, 8]


def make_bars(
    closes: list[float],
    opens: list[float] | None = None,
    symbol: str = "TEST",
    start: date = date(2024, 1, 1),
) -> pl.DataFrame:
    """Canonical daily bars from ``start``. ``opens`` defaults to ``closes``.

    high/low are set to the max/min of open and close, ±1, so the bars pass validation.
    """
    opens = opens if opens is not None else closes
    t0 = datetime(start.year, start.month, start.day)
    return pl.DataFrame(
        {
            "timestamp": [t0 + timedelta(days=i) for i in range(len(closes))],
            "open": [float(o) for o in opens],
            "high": [max(o, c) + 1.0 for o, c in zip(opens, closes, strict=True)],
            "low": [min(o, c) - 1.0 for o, c in zip(opens, closes, strict=True)],
            "close": [float(c) for c in closes],
            "volume": [1000.0] * len(closes),
            "symbol": [symbol] * len(closes),
        }
    ).with_columns(pl.col("timestamp").dt.replace_time_zone("UTC").cast(CANONICAL_SCHEMA["timestamp"]))


class Scripted(BaseStrategy):
    """Returns a pre-set signal per bar index. Any bar not in the script is HOLD."""

    name = "scripted"

    def __init__(self, script: dict[int, Signal]) -> None:
        super().__init__()
        self.script = script

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        return self.script.get(historical_data_up_to_now.height - 1, HOLD)
