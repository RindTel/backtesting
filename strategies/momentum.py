"""Time-series momentum (long-only)."""

import polars as pl

from strategies.base_strategy import HOLD, BaseStrategy, Side, Signal, require_positive_ints


class Momentum(BaseStrategy):
    """Hold while the price is above its level ``lookback`` bars ago.

    Buys on the bar the close rises above the close ``lookback`` bars earlier,
    sells on the bar it falls back below. Every other bar is a HOLD. Detecting
    that flip needs the comparison at bar t and bar t-1, so the first signal
    can come at ``lookback + 2`` bars.

    Params:
        lookback: How many bars back to compare against (default 90).
    """

    name = "momentum"
    label = "Momentum"
    summary = "Buys when the price is above where it was N days ago (an uptrend), sells when it drops back below."

    def __init__(self, lookback: int = 90) -> None:
        require_positive_ints(lookback=lookback)
        super().__init__(lookback=lookback)
        self.lookback = lookback

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        n = self.lookback
        if historical_data_up_to_now.height < n + 2:
            return HOLD
        # LOOK-AHEAD SAFETY: only the tail of the slice we were given (closes t-n-1 .. t).
        closes = historical_data_up_to_now["close"].tail(n + 2).to_list()
        up_now = closes[-1] > closes[-1 - n]
        up_before = closes[-2] > closes[-2 - n]
        if up_now and not up_before:
            return Signal(Side.BUY)
        if up_before and not up_now:
            return Signal(Side.SELL)
        return HOLD
