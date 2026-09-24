"""Trading strategies. Each one implements ``BaseStrategy.on_bar``.

``STRATEGIES`` maps the CLI name (``--strategy``) to the strategy class, in UI order.
"""

from strategies.base_strategy import BaseStrategy
from strategies.breakout import Breakout
from strategies.buy_and_hold import BuyAndHold
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.rsi_reversion import RsiReversion

STRATEGIES: dict[str, type[BaseStrategy]] = {
    cls.name: cls for cls in (MovingAverageCrossover, Momentum, Breakout, RsiReversion, BuyAndHold)
}
