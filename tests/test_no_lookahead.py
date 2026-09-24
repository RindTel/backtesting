"""Proof of the no-look-ahead guarantee.

The future-poisoning test: record a strategy's signals on clean data, then
again with every row after bar t replaced by garbage. If the runner is
look-ahead-safe, every decision made at bars <= t is the same in both runs.
The same test against ``broken_runner`` must FAIL, which shows the test can
actually detect leakage. (The only other test that touches the broken runner
is ``test_cli.py``, and only through the ``--runner broken`` flag.)
"""

import polars as pl
import pytest

from broken_runner import run_backtest_leaky
from runner import CostModel, history_up_to, run_backtest
from strategies import STRATEGIES
from strategies.base_strategy import HOLD, BaseStrategy, Side, Signal
from strategies.moving_average_crossover import MovingAverageCrossover
from tests.helpers import CLOSES, COSTS, Scripted, make_bars


def _signals_via_runner_slices(strategy: BaseStrategy, data: pl.DataFrame) -> list[Signal]:
    """Drive a strategy exactly as ``run_backtest`` does, and record every signal."""
    out = []
    for t in range(data.height):
        history = history_up_to(data, t)
        out.append(strategy.on_bar(history, history.row(-1, named=True)))
    return out


def _poison_after(data: pl.DataFrame, t: int) -> pl.DataFrame:
    """A completely different future after bar t: every later bar's prices scaled by
    a repeating x1.8 / x0.55 / x1.25 pattern and its volume tripled (timestamps kept
    in order). The uneven 3-step pattern changes later decisions at every cut used here.

    Wrong but physically possible, on purpose: a future of 1e9 prices makes the
    impact model's volatility explode, and the engine rightly refuses such trades.
    """
    idx = pl.int_range(pl.len())
    factor = pl.when(idx <= t).then(1.0).when(idx % 3 == 0).then(1.8).when(idx % 3 == 1).then(0.55).otherwise(1.25)
    return data.with_columns(
        [(pl.col(c) * factor).alias(c) for c in ("open", "high", "low", "close")]
        + [pl.when(idx > t).then(pl.col("volume") * 3).otherwise(pl.col("volume")).alias("volume")]
    )


@pytest.mark.parametrize("cut", [4, 6, 10, 15])
def test_runner_decisions_unchanged_when_future_is_poisoned(cut: int) -> None:
    """Signals returned at bars <= cut are identical between clean and poisoned data."""
    data = make_bars(CLOSES)
    strategy = MovingAverageCrossover(short_window=2, long_window=4)

    clean = _signals_via_runner_slices(strategy, data)
    poisoned = _signals_via_runner_slices(strategy, _poison_after(data, cut))

    assert poisoned[: cut + 1] == clean[: cut + 1]
    # Non-vacuity: the poison does change decisions after the cut, so the
    # equality above is a real check and not a coincidence.
    assert poisoned != clean


def _equity_through(run, data: pl.DataFrame, costs: CostModel) -> list[float]:
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    return run(data, strategy, costs, 10_000).equity_curve["equity"].to_list()


# Impact and the participation cap read past volume and volatility, so they get poisoned too.
REALISTIC = CostModel(fee_rate=0.001, slippage_rate=0.0005, impact_coefficient=1.0, max_participation=0.1)


@pytest.mark.parametrize("costs", [COSTS, REALISTIC], ids=["flat-costs", "with-impact"])
def test_end_to_end_poisoning_passes_correct_runner_and_catches_broken_runner(costs: CostModel) -> None:
    """The poisoning test through both real runners, end to end.

    For every cut, run on clean data and on data whose rows after the cut are
    replaced with absurd prices and volume, then compare the equity for bars <= cut.
    ``runner.run_backtest`` must match at EVERY cut. ``broken_runner`` must
    differ at SOME cut: its off-by-one slice lets bar cut+1 (poisoned) change
    a decision made at bar <= cut, and its same-close fill books that trade immediately.
    """
    data = make_bars(CLOSES)
    cuts = range(data.height - 1)

    def leaked_at(run) -> list[int]:
        clean = _equity_through(run, data, costs)
        return [c for c in cuts if _equity_through(run, _poison_after(data, c), costs)[: c + 1] != clean[: c + 1]]

    assert leaked_at(run_backtest) == []
    assert leaked_at(run_backtest_leaky), "poisoning failed to catch the deliberately broken runner"


def _poison_from_fill_bar(data: pl.DataFrame, t: int) -> pl.DataFrame:
    """Poison bar t except its OPEN (the one thing a fill at bar t may use), and every bar after it."""
    idx = pl.int_range(pl.len())
    at, after = idx == t, idx > t
    return data.with_columns(
        pl.when(after).then(pl.lit(1e9)).otherwise(pl.col("open")).alias("open"),
        pl.when(at | after).then(pl.lit(1e9)).otherwise(pl.col("high")).alias("high"),
        pl.when(after).then(pl.lit(1e9)).otherwise(pl.col("low")).alias("low"),
        pl.when(at | after).then(pl.lit(1e9)).otherwise(pl.col("close")).alias("close"),
        pl.when(at | after).then(pl.lit(1e12)).otherwise(pl.col("volume")).alias("volume"),
    )


def fills_up_to(run, data: pl.DataFrame, t: int) -> pl.DataFrame:
    strategy = MovingAverageCrossover(short_window=2, long_window=4)
    trades = run(data, strategy, REALISTIC, 10_000).trades
    return trades.filter(pl.col("timestamp") <= data["timestamp"][t])


def test_fill_prices_use_only_the_fill_bars_open_and_earlier_bars() -> None:
    """Finer than the equity check: a fill at bar t may use bar t's OPEN, but not its
    close or volume (unknown at the open). Poison everything else from bar t on;
    every fill up to and including bar t must come out identical."""
    data = make_bars(CLOSES, opens=[c + 0.5 for c in CLOSES])
    clean = run_backtest(data, MovingAverageCrossover(short_window=2, long_window=4), REALISTIC, 10_000).trades
    assert clean.height > 0
    for t in range(1, data.height):
        expected = clean.filter(pl.col("timestamp") <= data["timestamp"][t])
        assert fills_up_to(run_backtest, _poison_from_fill_bar(data, t), t).equals(expected), f"leak at bar {t}"


class Spy(BaseStrategy):
    """Records exactly what the runner shows it on every call."""

    name = "spy"

    def __init__(self) -> None:
        super().__init__()
        self.seen: list[tuple[int, object, dict, dict]] = []

    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        h = historical_data_up_to_now
        self.seen.append((h.height, h["timestamp"].max(), h.row(-1, named=True), current_bar))
        return HOLD


def test_strategy_never_sees_future_timestamps() -> None:
    """On bar t the slice has t+1 rows, its max timestamp is bar t's, and
    ``current_bar`` equals its last row.
    """
    data = make_bars(CLOSES)
    spy = Spy()
    run_backtest(data, spy, COSTS, 10_000)

    assert len(spy.seen) == data.height
    for t, (height, max_ts, last_row, current_bar) in enumerate(spy.seen):
        assert height == t + 1
        assert max_ts == data["timestamp"][t]
        assert current_bar == last_row == data.row(t, named=True)


def test_signals_fill_at_next_bar_open() -> None:
    """A signal at bar t fills at bar t+1's open (with slippage), never at bar t's
    close. A signal on the final bar is dropped.
    """

    # Opens and closes differ on every bar, so a same-bar fill would be obvious.
    data = make_bars(closes=[100, 110, 120, 130], opens=[95, 105, 115, 125])
    trades = run_backtest(data, Scripted({1: Signal(Side.BUY), 3: Signal(Side.SELL)}), COSTS, 10_000).trades

    # Only the BUY fills. The SELL came on the last bar and is dropped.
    assert trades.height == 1
    assert trades["signal_timestamp"][0] == data["timestamp"][1]
    assert trades["timestamp"][0] == data["timestamp"][2]
    assert trades["price"][0] == pytest.approx(115 * (1 + COSTS.slippage_rate))  # bar 2 OPEN


def test_strategy_cannot_tamper_with_runner_data() -> None:
    """Mutating the slice it receives does not change the runner's frame (Polars is copy-on-write)."""

    class Tamper(BaseStrategy):
        name = "tamper"

        def on_bar(self, historical_data_up_to_now, current_bar):
            historical_data_up_to_now[-1, "close"] = 1e9
            return HOLD

    data = make_bars(CLOSES)
    before = data.clone()
    result = run_backtest(data, Tamper(), COSTS, 10_000)
    assert data.equals(before)
    assert result.equity_curve["close"].to_list() == [float(c) for c in CLOSES]


def test_history_up_to_rejects_out_of_range_index() -> None:
    """A negative index would slice from the END of the frame (the future), so it must be rejected."""
    data = make_bars(CLOSES)
    for bad in (-1, data.height):
        with pytest.raises(IndexError):
            history_up_to(data, bad)


# Every registered strategy, with windows small enough to trade on the short test series.
SMALL_PARAMS = {
    "moving_average_crossover": {"short_window": 2, "long_window": 4},
    "momentum": {"lookback": 3},
    "breakout": {"entry_window": 3, "exit_window": 2},
    "rsi_reversion": {"period": 3},
    "buy_and_hold": {},
}


@pytest.mark.parametrize("costs", [COSTS, REALISTIC], ids=["flat-costs", "with-impact"])
@pytest.mark.parametrize("name", sorted(STRATEGIES))
def test_every_strategy_is_look_ahead_safe_through_the_runner(name: str, costs: CostModel) -> None:
    """The end-to-end poisoning check for each strategy: at every cut, equity for
    bars <= cut is identical whether the future is real or garbage."""
    assert set(SMALL_PARAMS) == set(STRATEGIES), "add new strategies to SMALL_PARAMS"
    # Scaled x3 so daily moves exceed make_bars' +/-1 high/low padding and breakouts can fire.
    data = make_bars([3.0 * c for c in CLOSES])

    def equity(d: pl.DataFrame) -> list[float]:
        strategy = STRATEGIES[name](**SMALL_PARAMS[name])
        return run_backtest(d, strategy, costs, 10_000).equity_curve["equity"].to_list()

    clean = equity(data)
    assert [c for c in range(data.height - 1) if equity(_poison_after(data, c))[: c + 1] != clean[: c + 1]] == []
    # Non-vacuity: the strategy actually trades on this series, so the check above tested something.
    assert run_backtest(data, STRATEGIES[name](**SMALL_PARAMS[name]), costs, 10_000).trades.height > 0
