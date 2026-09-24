"""Fees and slippage are always applied. There is no frictionless mode."""

import math
import statistics

import polars as pl
import pytest

from runner import CostModel, _market_conditions, run_backtest
from strategies.base_strategy import Side, Signal
from tests.helpers import COSTS, Scripted, make_bars

BUY_THEN_SELL = {0: Signal(Side.BUY), 1: Signal(Side.SELL)}


@pytest.mark.parametrize(
    ("fee", "slippage"),
    [(0.0, 0.0005), (0.001, 0.0), (0.0, 0.0), (-0.001, 0.0005), (0.001, -0.0005), (1.0, 0.0005)],
)
def test_cost_model_rejects_non_positive_or_absurd_rates(fee: float, slippage: float) -> None:
    with pytest.raises(ValueError, match="frictionless"):
        CostModel(fee_rate=fee, slippage_rate=slippage)


def test_slippage_always_moves_price_against_trader() -> None:
    data = make_bars([100, 100, 100])
    trades = run_backtest(data, Scripted(BUY_THEN_SELL), COSTS, 10_000).trades
    buy, sell = trades.row(0, named=True), trades.row(1, named=True)
    assert buy["price"] == pytest.approx(100 * 1.0005)
    assert sell["price"] == pytest.approx(100 * 0.9995)
    assert buy["slippage_cost"] > 0 and sell["slippage_cost"] > 0


def test_fee_charged_on_every_fill() -> None:
    data = make_bars([100, 100, 100])
    trades = run_backtest(data, Scripted(BUY_THEN_SELL), COSTS, 10_000).trades
    assert trades.height == 2
    for row in trades.iter_rows(named=True):
        assert row["fee"] == pytest.approx(row["notional"] * COSTS.fee_rate)
        assert row["fee"] > 0


def test_round_trip_at_flat_price_loses_money() -> None:
    data = make_bars([100, 100, 100])
    result = run_backtest(data, Scripted(BUY_THEN_SELL), COSTS, 10_000)
    assert result.equity_curve["equity"][-1] < 10_000


# --- Market impact and participation cap ------------------------------------


def _varied_bars(n: int = 30):
    """Closes that zig-zag upward, and volume that changes every day, so σ and ADV are non-trivial."""
    closes = [100.0 + i + (3.0 if i % 2 else -2.0) for i in range(n)]
    return make_bars(closes).with_columns(pl.Series("volume", [1_000.0 + 37.0 * i for i in range(n)]))


def test_impact_matches_square_root_law_from_prior_bars_only() -> None:
    """BUY signalled on bar 25 fills at bar 26's open. σ and ADV must come from the 20 bars
    BEFORE bar 26 (bars 6..25), computed here independently with the statistics module."""
    data = _varied_bars()
    costs = CostModel(fee_rate=0.001, slippage_rate=0.0005, impact_coefficient=1.0)
    trades = run_backtest(data, Scripted({25: Signal(Side.BUY)}), costs, 10_000).trades

    closes, volumes = data["close"].to_list(), data["volume"].to_list()
    returns = [closes[i] / closes[i - 1] - 1 for i in range(6, 26)]
    sigma = statistics.stdev(returns)
    adv = statistics.fmean(closes[i] * volumes[i] for i in range(6, 26))
    notional = 10_000 / 1.001
    impact = 1.0 * sigma * math.sqrt(notional / adv)

    fill = trades.row(0, named=True)
    assert fill["price"] == pytest.approx(data["open"][26] * (1 + 0.0005 + impact), rel=1e-12)
    assert fill["impact_cost"] == pytest.approx(data["open"][26] * impact * fill["quantity"], rel=1e-12)


def test_bigger_orders_pay_more_impact() -> None:
    data = _varied_bars()
    costs = CostModel(fee_rate=0.001, slippage_rate=0.0005, impact_coefficient=1.0)
    prices = [
        run_backtest(data, Scripted({25: Signal(Side.BUY)}), costs, capital).trades["price"][0]
        for capital in (1_000, 10_000, 100_000)
    ]
    assert prices[0] < prices[1] < prices[2]


def test_participation_cap_splits_a_large_order_across_bars() -> None:
    """A BUY far above 10% of ADV fills in slices of at most 10% of ADV per bar,
    all tracing back to the same signal, until the cash is spent or the data ends."""
    data = _varied_bars()
    costs = CostModel(fee_rate=0.001, slippage_rate=0.0005, max_participation=0.1)
    trades = run_backtest(data, Scripted({20: Signal(Side.BUY)}), costs, 1e9).trades

    assert trades.height > 1
    assert trades["signal_timestamp"].unique().to_list() == [data["timestamp"][20]]
    _, advs = _market_conditions(data)
    for row in trades.iter_rows(named=True):
        t = data["timestamp"].to_list().index(row["timestamp"])
        assert row["notional"] == pytest.approx(0.1 * advs[t], rel=1e-12)


@pytest.mark.parametrize(("coef", "participation"), [(-0.1, 0.1), (1.0, 0.0), (1.0, 1.5)])
def test_cost_model_rejects_bad_impact_settings(coef: float, participation: float) -> None:
    with pytest.raises(ValueError):
        CostModel(fee_rate=0.001, slippage_rate=0.0005, impact_coefficient=coef, max_participation=participation)


def test_impact_that_would_make_the_sell_price_negative_is_refused() -> None:
    """Reproduces a review finding: with k=50 a sale executed at a negative price
    (selling cost money). The engine now raises instead of booking it."""
    data = _varied_bars()
    costs = CostModel(fee_rate=0.001, slippage_rate=0.0005, impact_coefficient=50.0)
    strategy = Scripted({22: Signal(Side.BUY), 25: Signal(Side.SELL)})
    with pytest.raises(ValueError, match="sell price to zero or below"):
        run_backtest(data, strategy, costs, 10_000_000)
