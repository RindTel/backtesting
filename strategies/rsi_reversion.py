"""RSI mean reversion: buy the dip, sell the rip (long-only)."""

import polars as pl

from strategies.base_strategy import HOLD, BaseStrategy, Side, Signal, require_positive_ints


def rsi(closes: list[float]) -> float:
    """Relative Strength Index of a close series, with simple averages (Cutler's RSI).

    Over the changes between consecutive closes: ``RSI = 100 - 100 / (1 + avg_gain / avg_loss)``.
    100 when there were no losses, 50 when the price didn't move at all.
    """
    changes = [b - a for a, b in zip(closes, closes[1:], strict=False)]
    gain = sum(max(c, 0.0) for c in changes) / len(changes)
    loss = sum(max(-c, 0.0) for c in changes) / len(changes)
    if loss == 0:
        return 50.0 if gain == 0 else 100.0
    return 100 - 100 / (1 + gain / loss)


class RsiReversion(BaseStrategy):
    """Buy when the RSI drops into oversold territory, sell when it rises into overbought.

    Signals fire on the bar the RSI crosses the threshold (below ``oversold``:
    BUY; above ``overbought``: SELL); every other bar is a HOLD. The RSI at bar
    t uses the last ``period`` price changes, and a cross compares bar t with
    bar t-1, so the first signal can come at ``period + 2`` bars.

    Params:
        period: Price changes in the RSI window (default 14).
        oversold: Buy threshold, 1-99 (default 30).
        overbought: Sell threshold, above ``oversold`` (default 70).
    """

    name = "rsi_reversion"
    label = "RSI mean reversion"
    summary = "Buys dips: when the RSI falls below 30 (oversold). Sells when it climbs above 70 (overbought)."

    def __init__(self, period: int = 14, oversold: int = 30, overbought: int = 70) -> None:
        require_positive_ints(period=period, oversold=oversold, overbought=overbought)
        if not oversold < overbought < 100:
            raise ValueError(f"need oversold < overbought < 100, got {oversold} and {overbought}")
        super().__init__(period=period, oversold=oversold, overbought=overbought)
        self.period, self.oversold, self.overbought = period, oversold, overbought

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        n = self.period
        if historical_data_up_to_now.height < n + 2:
            return HOLD
        # LOOK-AHEAD SAFETY: only the tail of the slice we were given (closes t-n-1 .. t).
        closes = historical_data_up_to_now["close"].tail(n + 2).to_list()
        rsi_now, rsi_before = rsi(closes[1:]), rsi(closes[:-1])
        if rsi_now < self.oversold <= rsi_before:
            return Signal(Side.BUY)
        if rsi_now > self.overbought >= rsi_before:
            return Signal(Side.SELL)
        return HOLD
