"""Unit tests for the strategy interface and the concrete strategies."""

import polars as pl
import pytest

from strategies import STRATEGIES
from strategies.base_strategy import HOLD, BaseStrategy, Side, Signal
from strategies.breakout import Breakout
from strategies.buy_and_hold import BuyAndHold
from strategies.momentum import Momentum
from strategies.moving_average_crossover import MovingAverageCrossover
from strategies.rsi_reversion import RsiReversion, rsi
from tests.helpers import make_bars

BUY = Signal(Side.BUY)
SELL = Signal(Side.SELL)


def _run(strategy: BaseStrategy, df: pl.DataFrame) -> list[Signal]:
    """Feed bars one at a time, the way the runner does: a slice of rows 0..t plus row t."""
    signals = []
    for t in range(df.height):
        history = df.slice(0, t + 1)
        signals.append(strategy.on_bar(history, history.row(-1, named=True)))
    return signals


# --- Moving-average crossover ------------------------------------------------


def test_crossover_signals_on_exact_bars() -> None:
    """Hand-computed SMA(2) vs SMA(4):

    bar  close  SMA2  SMA4
     3     7    7.5   8.5
     4     6    6.5   7.5
     5     7    6.5   7.0    short <= long
     6     8    7.5   7.0    short >  long  -> BUY  (crossed above)
     7     9    8.5   7.5
     8    10    9.5   8.5
     9     9    9.5   9.0    short >= long
    10     8    8.5   9.0    short <  long  -> SELL (crossed below)
    11     7    7.5   8.5
    12     6    6.5   7.5

    Bars 0-3 are warm-up (a cross needs SMA4 at both t and t-1). Bar 4 is the
    first bar that can signal, and it doesn't cross.
    """
    closes = [10, 9, 8, 7, 6, 7, 8, 9, 10, 9, 8, 7, 6]
    signals = _run(MovingAverageCrossover(short_window=2, long_window=4), make_bars([float(c) for c in closes]))

    expected = [HOLD] * len(closes)
    expected[6] = BUY
    expected[10] = SELL
    assert signals == expected


def test_crossover_holds_during_warmup() -> None:
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    # A sharp rise that would be a crossover, but only 4 bars exist (5 are needed).
    assert _run(strategy, make_bars([1.0, 1.0, 1.0, 50.0])) == [HOLD] * 4


def test_crossover_params_are_logged() -> None:
    assert MovingAverageCrossover(3, 7).params == {"short_window": 3, "long_window": 7}


@pytest.mark.parametrize(
    ("short", "long"),
    [(5, 5), (10, 5), (0, 5), (-1, 5), (2.5, 5), (True, 5)],
)
def test_crossover_rejects_invalid_windows(short, long) -> None:
    with pytest.raises(ValueError):
        MovingAverageCrossover(short_window=short, long_window=long)


# --- Buy and hold ------------------------------------------------------------


def test_buy_and_hold_always_says_buy() -> None:
    """The runner fills the first BUY with all the cash; later ones find no cash (see test_runner)."""
    assert _run(BuyAndHold(), make_bars([10.0, 11.0, 9.0, 12.0])) == [BUY] * 4


# --- Signal ------------------------------------------------------------------


def test_signal_size_defaults_to_none() -> None:
    assert Signal(Side.BUY).size is None


@pytest.mark.parametrize("size", [0.0, -0.5, 1.01])
def test_signal_rejects_size_outside_unit_interval(size: float) -> None:
    with pytest.raises(ValueError, match="size"):
        Signal(Side.BUY, size)


def test_hold_signal_cannot_have_size() -> None:
    with pytest.raises(ValueError, match="HOLD"):
        Signal(Side.HOLD, 0.5)


# --- Momentum ------------------------------------------------------------------


def test_momentum_signals_when_the_trend_flips() -> None:
    """lookback=2: 'up' means close[t] > close[t-2].

    bar  close  vs t-2   up?
     2    12    10        yes
     3    11    11        no    -> SELL (was up, now not)
     4    10    12        no
     5     9    11        no
     6    10    10        no
     7    12     9        yes   -> BUY (was not up, now up)
     8    13    10        yes
    Bars 0-2 are warm-up (a flip needs the comparison at t and t-1).
    """
    closes = [10.0, 11, 12, 11, 10, 9, 10, 12, 13]
    expected = [HOLD] * len(closes)
    expected[3], expected[7] = SELL, BUY
    assert _run(Momentum(lookback=2), make_bars(closes)) == expected


# --- Breakout ----------------------------------------------------------------------


def test_breakout_buys_new_highs_and_sells_new_lows() -> None:
    """entry=3, exit=2. make_bars sets high = close + 1 and low = close - 1.

    bar  close  prior-3 highs  prior-2 lows  signal
     3    12    11, 11, 11     9, 9          BUY  (12 > 11)
     4    11    11, 11, 13     9, 11         HOLD
     5    10    11, 13, 12     11, 10        HOLD (10 is not < 10)
     6     8    13, 12, 11     10, 9         SELL (8 < 9)
     7     9    12, 11, 9      9, 7          HOLD
    """
    closes = [10.0, 10, 10, 12, 11, 10, 8, 9]
    expected = [HOLD] * len(closes)
    expected[3], expected[6] = BUY, SELL
    assert _run(Breakout(entry_window=3, exit_window=2), make_bars(closes)) == expected


# --- RSI mean reversion --------------------------------------------------------------


def test_rsi_values() -> None:
    assert rsi([10.0, 11, 12]) == 100.0  # no losses
    assert rsi([10.0, 10, 10]) == 50.0  # no movement
    assert rsi([11.0, 12, 10]) == pytest.approx(100 - 100 / 1.5)  # gain 0.5, loss 1.0 per change


def test_rsi_reversion_buys_oversold_and_sells_overbought() -> None:
    """period=2, thresholds 30/70.

    bar  closes (t-2..t)  RSI
     3   11, 12, 10       33.3
     4   12, 10,  8        0.0   -> BUY  (crossed below 30)
     5   10,  8,  9       33.3
     6    8,  9, 12      100.0   -> SELL (crossed above 70)
     7    9, 12, 13      100.0
    """
    closes = [10.0, 11, 12, 10, 8, 9, 12, 13]
    expected = [HOLD] * len(closes)
    expected[4], expected[6] = BUY, SELL
    assert _run(RsiReversion(period=2), make_bars(closes)) == expected


@pytest.mark.parametrize(
    "make",
    [
        lambda: Momentum(lookback=0),
        lambda: Breakout(entry_window=0),
        lambda: Breakout(exit_window=2.5),
        lambda: RsiReversion(period=0),
        lambda: RsiReversion(oversold=70, overbought=30),
        lambda: RsiReversion(overbought=100),
    ],
)
def test_new_strategies_reject_invalid_params(make) -> None:
    with pytest.raises(ValueError):
        make()


def test_every_registered_strategy_describes_itself() -> None:
    for name, cls in STRATEGIES.items():
        assert cls.name == name and cls.label and cls.summary
