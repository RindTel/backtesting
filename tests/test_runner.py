"""Tests for ``runner.run_backtest``: hand-calculated P&L and execution rules."""

import polars as pl
import pytest

from runner import run_backtest
from strategies.base_strategy import BaseStrategy, Side, Signal
from tests.helpers import COSTS, Scripted, make_bars


def test_round_trip_matches_hand_calculation() -> None:
    """Buy on day 1, sell on the last day, and check every number by hand.

    Bars (open / close):  0: 100/100   1: 100/105   2: 110/110   3: 120/125
    Capital 10,000, fee 0.1%, slippage 5 bps. Signals: BUY on bar 0, SELL on bar 2.
    Signals fill at the NEXT bar's open, so the BUY fills on bar 1 and the
    SELL fills on bar 3 (the last day).

    BUY  @ bar 1 open: price    = 100 × 1.0005                  = 100.05
                       notional = 10,000 / 1.001                = 9,990.00999000999…
                       fee      = 10,000 − notional             = 9.99000999000999…
                       quantity = notional / 100.05             = 99.85017481269355…
                       cash     = 0
    bar 1 equity = quantity × 105                               = 10,484.268355332822…
    bar 2 equity = quantity × 110                               = 10,983.51922939629…
    SELL @ bar 3 open: price    = 120 × 0.9995                  = 119.94
                       notional = quantity × 119.94             = 11,976.029967034465…
                       fee      = notional × 0.001              = 11.976029967034465…
                       cash     = notional − fee                = 11,964.05393706743…

    Closed form: final = 10,000 × (0.999 × 119.94) / (1.001 × 100.05)
    """
    data = make_bars(closes=[100, 105, 110, 125], opens=[100, 100, 110, 120])
    strategy = Scripted({0: Signal(Side.BUY), 2: Signal(Side.SELL)})

    result = run_backtest(data, strategy, COSTS, initial_capital=10_000)

    expected_final = 11_964.05393706743
    assert expected_final == pytest.approx(10_000 * (0.999 * 119.94) / (1.001 * 100.05), rel=1e-15)
    exact = pytest.approx  # float arithmetic: equal to ~12 significant digits
    rel = 1e-12

    eq = result.equity_curve
    assert eq["equity"].to_list() == [
        exact(10_000.0, rel=rel),
        exact(10_484.268355332822, rel=rel),
        exact(10_983.51922939629, rel=rel),
        exact(expected_final, rel=rel),
    ]
    assert eq["position"][-1] == 0.0
    assert eq["cash"][-1] == exact(expected_final, rel=rel)

    trades = result.trades
    assert trades["side"].to_list() == ["buy", "sell"]
    assert trades["price"].to_list() == [exact(100.05, rel=rel), exact(119.94, rel=rel)]
    assert trades["quantity"].to_list() == [exact(99.85017481269355, rel=rel)] * 2
    assert trades["fee"].to_list() == [exact(9.99000999000999, rel=rel), exact(11.976029967034465, rel=rel)]
    # Next-bar rule: fills happen one bar after the signal.
    ts = data["timestamp"]
    assert trades["timestamp"].to_list() == [ts[1], ts[3]]
    assert trades["signal_timestamp"].to_list() == [ts[0], ts[2]]


def test_full_buy_leaves_cash_at_exactly_zero() -> None:
    data = make_bars([100, 101, 102])
    result = run_backtest(data, Scripted({0: Signal(Side.BUY)}), COSTS, 10_000)
    assert result.equity_curve["cash"].to_list() == [10_000.0, 0.0, 0.0]


def test_partial_sizes_are_respected() -> None:
    data = make_bars([100, 100, 100, 100])
    strategy = Scripted({0: Signal(Side.BUY, 0.5), 1: Signal(Side.SELL, 0.25)})
    trades = run_backtest(data, strategy, COSTS, 10_000).trades

    assert trades["notional"][0] + trades["fee"][0] == pytest.approx(5_000.0)
    assert trades["cash_after"][0] == pytest.approx(5_000.0)
    assert trades["quantity"][1] == pytest.approx(trades["quantity"][0] * 0.25)
    assert trades["position_after"][1] == pytest.approx(trades["quantity"][0] * 0.75)


def test_sell_without_position_and_buy_without_cash_do_nothing() -> None:
    data = make_bars([100, 100, 100, 100])
    # SELL while flat; then BUY all; then BUY again with no cash left.
    strategy = Scripted({0: Signal(Side.SELL), 1: Signal(Side.BUY), 2: Signal(Side.BUY)})
    trades = run_backtest(data, strategy, COSTS, 10_000).trades
    assert trades["side"].to_list() == ["buy"]


def test_rejects_unsorted_data() -> None:
    data = make_bars([100, 101, 102])[[0, 2, 1]]
    with pytest.raises(ValueError, match="not sorted"):
        run_backtest(data, Scripted({}), COSTS, 10_000)


def test_rejects_multiple_symbols() -> None:
    data = pl.concat([make_bars([100, 101], symbol="A"), make_bars([100, 101], symbol="B")])
    with pytest.raises(ValueError, match="one symbol"):
        run_backtest(data, Scripted({}), COSTS, 10_000)


def test_rejects_non_signal_return() -> None:
    class Bad(BaseStrategy):
        name = "bad"

        def on_bar(self, historical_data_up_to_now, current_bar):
            return "buy"

    with pytest.raises(TypeError, match="must return a Signal"):
        run_backtest(make_bars([100, 101]), Bad(), COSTS, 10_000)


def test_buy_and_hold_fills_exactly_once() -> None:
    from strategies.buy_and_hold import BuyAndHold

    result = run_backtest(make_bars([100, 101, 102, 103]), BuyAndHold(), COSTS, 10_000)
    assert result.trades["side"].to_list() == ["buy"]
    assert result.trades["timestamp"][0] == make_bars([100, 101])["timestamp"][1]  # next bar's open


def test_empty_trade_log_has_schema() -> None:
    result = run_backtest(make_bars([100, 101, 102]), Scripted({}), COSTS, 10_000)
    assert result.trades.is_empty()
    assert "fee" in result.trades.columns
    assert result.equity_curve["equity"].to_list() == [10_000.0] * 3
