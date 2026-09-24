"""Buy-and-hold baseline: every other strategy should be compared against this."""

import polars as pl

from strategies.base_strategy import BaseStrategy, Side, Signal


class BuyAndHold(BaseStrategy):
    """Buy on the first bar, then hold to the end.

    Has no parameters. It exists so that each run has a cost-inclusive
    benchmark in MLflow. It says BUY on every bar: the runner fills the first
    one with all the cash, and every later BUY finds no cash and does nothing.
    That keeps it stateless without depending on how many rows a slice holds
    (an earlier "buy when the slice has one row" version never traded under
    the broken runner, which always hands over one extra row).
    """

    name = "buy_and_hold"
    label = "Buy & hold"
    summary = "Buys on day one and never sells. The benchmark every other strategy has to beat."

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        return Signal(Side.BUY)
