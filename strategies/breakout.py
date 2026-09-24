"""Channel breakout, the classic "Turtle" rule (long-only)."""

import polars as pl

from strategies.base_strategy import HOLD, BaseStrategy, Side, Signal, require_positive_ints


class Breakout(BaseStrategy):
    """Buy a new high, sell a new low (a Donchian channel breakout).

    * BUY when today's close is above the highest high of the previous
      ``entry_window`` bars.
    * SELL when today's close is below the lowest low of the previous
      ``exit_window`` bars.

    "Previous" excludes today, so today's own high and low never set the level
    it is compared against. Entry and exit are levels rather than one-bar
    events, so a signal can repeat on consecutive bars; the runner ignores a
    BUY with no cash and a SELL with nothing to sell.

    Params:
        entry_window: Bars in the breakout channel (default 20).
        exit_window: Bars in the exit channel (default 10).
    """

    name = "breakout"
    label = "Breakout"
    summary = (
        "Buys when the price closes above its highest high of the last N days, "
        "sells below the lowest low of the last M days."
    )

    def __init__(self, entry_window: int = 20, exit_window: int = 10) -> None:
        require_positive_ints(entry_window=entry_window, exit_window=exit_window)
        super().__init__(entry_window=entry_window, exit_window=exit_window)
        self.entry_window = entry_window
        self.exit_window = exit_window

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        n = max(self.entry_window, self.exit_window)
        if historical_data_up_to_now.height < n + 1:
            return HOLD
        # LOOK-AHEAD SAFETY: the channels come from the n bars BEFORE today, all inside the slice.
        prior = historical_data_up_to_now.tail(n + 1).head(n)
        close = current_bar["close"]
        if close > prior["high"].tail(self.entry_window).max():
            return Signal(Side.BUY)
        if close < prior["low"].tail(self.exit_window).min():
            return Signal(Side.SELL)
        return HOLD
