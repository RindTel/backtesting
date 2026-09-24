"""Moving-average crossover strategy (long-only)."""

import polars as pl

from strategies.base_strategy import HOLD, BaseStrategy, Side, Signal, require_positive_ints


class MovingAverageCrossover(BaseStrategy):
    """Buy when the short moving average crosses above the long one, sell when it crosses back below.

    Signals fire only on the bar where the cross happens. Every other bar is
    a HOLD. The strategy keeps no state between calls, so each signal depends
    only on the slice it was given.

    Params:
        short_window: Short simple-moving-average length in bars (default 20).
        long_window: Long simple-moving-average length in bars (default 50).
            Must be greater than ``short_window``.
    """

    name = "moving_average_crossover"
    label = "MA crossover"
    summary = (
        "Buys when the short-term average price crosses above the long-term one "
        "(an uptrend starting), sells on the reverse cross."
    )

    def __init__(self, short_window: int = 20, long_window: int = 50) -> None:
        require_positive_ints(short_window=short_window, long_window=long_window)
        if short_window >= long_window:
            raise ValueError(f"short_window ({short_window}) must be less than long_window ({long_window})")
        super().__init__(short_window=short_window, long_window=long_window)
        self.short_window = short_window
        self.long_window = long_window

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        # Detecting a cross needs both averages at the current bar t AND the
        # previous bar t-1, so the long average must be defined at t-1:
        # that takes long_window + 1 bars.
        if historical_data_up_to_now.height < self.long_window + 1:
            return HOLD

        # LOOK-AHEAD SAFETY: the averages are computed right here, from the
        # slice we were given, which ends at the current bar. Nothing is
        # precomputed over the full dataset. Only the last long_window + 1
        # closes are needed, so take just that tail.
        closes = historical_data_up_to_now["close"].tail(self.long_window + 1)
        short_ma = closes.rolling_mean(self.short_window)
        long_ma = closes.rolling_mean(self.long_window)
        short_prev, short_now = short_ma[-2], short_ma[-1]
        long_prev, long_now = long_ma[-2], long_ma[-1]

        if short_prev <= long_prev and short_now > long_now:
            return Signal(Side.BUY)
        if short_prev >= long_prev and short_now < long_now:
            return Signal(Side.SELL)
        return HOLD
