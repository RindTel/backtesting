"""Each metric in ``metrics.py`` checked against hand-computed values.

Where the arithmetic is exact in binary floating point, the tests use ``==``.
Otherwise they use ``pytest.approx(rel=1e-12)``: equal to about 12
significant digits, which is as exact as float arithmetic allows.
"""

import math
from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from metrics import (
    Drawdown,
    cagr,
    compute_metrics,
    max_drawdown,
    round_trip_pnls,
    sharpe_ratio,
    split_metrics,
    total_return,
    win_rate,
)
from runner import TRADE_LOG_SCHEMA, run_backtest
from strategies.base_strategy import Side, Signal
from tests.helpers import COSTS, Scripted, make_bars

T0 = datetime(2024, 1, 1, tzinfo=UTC)
REL = 1e-12


def _series(values: list[float]) -> pl.Series:
    return pl.Series("equity", [float(v) for v in values])


def _days(n: int) -> pl.Series:
    return pl.Series("timestamp", [T0 + timedelta(days=i) for i in range(n)])


def _day(i: int) -> datetime:
    return T0 + timedelta(days=i)


# --- Total return ------------------------------------------------------------


def test_total_return() -> None:
    """100 → 200: 200 / 100 − 1 = 1.0 (the path in between doesn't matter)."""
    assert total_return(_series([100, 150, 120, 200])) == 1.0


def test_total_return_loss() -> None:
    """100 → 75: 75 / 100 − 1 = −0.25."""
    assert total_return(_series([100, 90, 75])) == -0.25


# --- CAGR --------------------------------------------------------------------


def test_cagr_uses_calendar_time() -> None:
    """100 → 400 over 730.5 days = exactly 2 years of 365.25 days.
    CAGR = 4 ** (1/2) − 1 = 1.0 (100% a year: 100 → 200 → 400).
    """
    ts = pl.Series([T0, T0 + timedelta(days=730.5)])
    assert cagr(_series([100, 400]), ts) == 1.0


def test_cagr_loss() -> None:
    """100 → 25 over 2 years: 0.25 ** (1/2) − 1 = −0.5."""
    ts = pl.Series([T0, T0 + timedelta(days=730.5)])
    assert cagr(_series([100, 25]), ts) == -0.5


def test_cagr_less_than_a_year() -> None:
    """100 → 110 over a quarter year (91.3125 days): 1.1 ** 4 − 1 = 0.4641."""
    ts = pl.Series([T0, T0 + timedelta(days=365.25 / 4)])
    assert cagr(_series([100, 110]), ts) == pytest.approx(0.4641, rel=REL)


def test_cagr_zero_span_is_nan() -> None:
    assert math.isnan(cagr(_series([100, 110]), pl.Series([T0, T0])))


# --- Max drawdown ------------------------------------------------------------


def test_max_drawdown_depth_and_dates() -> None:
    """
    day  equity  running max  drawdown
     0    100       100          0
     1    120       120          0        <- peak
     2     90       120        −0.25
     3     60       120        −0.5       <- trough (max drawdown)
     4     80       120       −0.333…
     5    130       130          0        <- recovery (first day >= 120 after the trough)
     6    104       130        −0.2       (a smaller, later dip)
    """
    dd = max_drawdown(_series([100, 120, 90, 60, 80, 130, 104]), _days(7))
    assert dd == Drawdown(depth=-0.5, peak=_day(1), trough=_day(3), recovery=_day(5))


def test_max_drawdown_starts_at_end_of_plateau_and_may_not_recover() -> None:
    """Equity sits at the 120 high on days 1–2, so the drawdown starts on day 2.
    Trough: 90/120 − 1 = −0.25 on day 3. It never gets back to 120, so recovery is None.
    """
    dd = max_drawdown(_series([100, 120, 120, 90, 100]), _days(5))
    assert dd == Drawdown(depth=-0.25, peak=_day(2), trough=_day(3), recovery=None)


def test_max_drawdown_monotonic_curve_is_zero() -> None:
    assert max_drawdown(_series([100, 100, 105, 110]), _days(4)) == Drawdown(0.0, None, None, None)


# --- Sharpe ------------------------------------------------------------------


def test_sharpe_ratio() -> None:
    """Equity 100 → 110 → 99 → 108.9 gives returns +0.1, −0.1, +0.1.

    mean = 0.1/3 = 1/30
    deviations: 2/30, −4/30, 2/30, so the sum of squares is 24/900 = 2/75
    sample variance = (2/75) / 2 = 1/75, and std = 1/(5√3)
    per-bar Sharpe = (1/30) · 5√3 = √3/6
    annualized (252) = √3/6 · √252 = √(3·252/36) = √21 ≈ 4.58258
    """
    assert sharpe_ratio(_series([100, 110, 99, 108.9]), periods_per_year=252) == pytest.approx(math.sqrt(21), rel=REL)


def test_sharpe_ratio_with_risk_free_rate() -> None:
    """rf = 2.52% a year gives 0.0252/252 = 0.0001 per bar. It shifts the mean, not the std:
    Sharpe = (1/30 − 0.0001) · 5√3 · √252 = (1/30 − 0.0001) · 30√21 = 0.997 · √21.
    """
    sharpe = sharpe_ratio(_series([100, 110, 99, 108.9]), periods_per_year=252, risk_free_rate=0.0252)
    assert sharpe == pytest.approx(0.997 * math.sqrt(21), rel=REL)


def test_sharpe_ratio_annualization_factor() -> None:
    """The same returns at 365 bars/year give √3/6 · √365."""
    sharpe = sharpe_ratio(_series([100, 110, 99, 108.9]), periods_per_year=365)
    assert sharpe == pytest.approx(math.sqrt(3) / 6 * math.sqrt(365), rel=REL)


def test_sharpe_ratio_of_flat_equity_is_nan() -> None:
    """Zero volatility means the Sharpe ratio is undefined, not 0."""
    assert math.isnan(sharpe_ratio(_series([100, 100, 100, 100]), periods_per_year=252))


# --- Win rate ----------------------------------------------------------------


def _trade_log(fills: list[tuple[str, float, float, float]]) -> pl.DataFrame:
    """Full-schema trade log from ``(side, notional, fee, position_after)`` tuples."""
    rows = [
        {
            "timestamp": _day(i + 1),
            "signal_timestamp": _day(i),
            "symbol": "TEST",
            "side": side,
            "price": 100.0,
            "quantity": notional / 100.0,
            "notional": notional,
            "fee": fee,
            "slippage_cost": 0.05,
            "position_after": position_after,
            "cash_after": 0.0,
        }
        for i, (side, notional, fee, position_after) in enumerate(fills)
    ]
    return pl.DataFrame(rows, schema=TRADE_LOG_SCHEMA)


# Four trips. Net P&L = Σ(sell notional − fee) − Σ(buy notional + fee).
FOUR_TRIPS = _trade_log(
    [
        # A: −(1000 + 1) + (1100 − 1.1) = +97.9     win
        ("buy", 1000.0, 1.0, 10.0),
        ("sell", 1100.0, 1.1, 0.0),
        # B: −(1000 + 1) + (1001 − 1.001) = −1.001   LOSS: up 1 before fees, down after them
        ("buy", 1000.0, 1.0, 10.0),
        ("sell", 1001.0, 1.001, 0.0),
        # C: −(1000 + 1) + (600 − 0.6) + (600 − 0.6) = +197.8   win, with a partial exit
        ("buy", 1000.0, 1.0, 10.0),
        ("sell", 600.0, 0.6, 5.0),
        ("sell", 600.0, 0.6, 0.0),
        # D: still open at the end, so not counted
        ("buy", 1000.0, 1.0, 10.0),
    ]
)


def test_round_trip_pnls_are_net_of_fees() -> None:
    assert round_trip_pnls(FOUR_TRIPS) == [
        pytest.approx(97.9, rel=REL),
        pytest.approx(-1.001, rel=REL),
        pytest.approx(197.8, rel=REL),
    ]


def test_win_rate() -> None:
    """2 winners (A, C) out of 3 closed trips. The open trip D is ignored."""
    assert win_rate(round_trip_pnls(FOUR_TRIPS)) == 2 / 3


def test_win_rate_without_closed_trips_is_nan() -> None:
    assert math.isnan(win_rate(round_trip_pnls(_trade_log([]))))
    assert math.isnan(win_rate(round_trip_pnls(_trade_log([("buy", 1000.0, 1.0, 10.0)]))))


# --- compute_metrics: end to end on the runner's output ----------------------


def test_compute_metrics_on_hand_calculated_backtest() -> None:
    """The round trip from tests/test_runner.py: buy fills at 100.05, sell at 119.94,
    fee 0.1%, starting cash 10,000.

    total return  = 0.999 × 119.94 / (1.001 × 100.05) − 1 ≈ 0.196405
    equity        = 10,000 → 10,484.27 → 10,983.52 → 11,964.05, rising every bar, so no drawdown
    round trips   = 1, and it's a winner, so the win rate is 1.0. 2 fills.
    CAGR          = (1 + total return) ** (1 / years) − 1, with years = 3 days / 365.25
    """
    data = make_bars(closes=[100, 105, 110, 125], opens=[100, 100, 110, 120])
    result = run_backtest(data, Scripted({0: Signal(Side.BUY), 2: Signal(Side.SELL)}), COSTS, 10_000)

    m = compute_metrics(result.equity_curve, result.trades, periods_per_year=252)

    growth = 0.999 * 119.94 / (1.001 * 100.05)
    assert m.total_return == pytest.approx(growth - 1, rel=REL)
    assert m.cagr == pytest.approx(growth ** (365.25 / 3) - 1, rel=1e-9)
    assert m.max_drawdown == Drawdown(0.0, None, None, None)
    assert m.win_rate == 1.0
    assert (m.num_round_trips, m.num_trades) == (1, 2)
    assert m.sharpe_ratio > 0


# --- Input validation --------------------------------------------------------


def test_rejects_too_short_equity_curve() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        total_return(_series([100]))


# --- In-sample / out-of-sample split -----------------------------------------


def test_split_metrics_by_hand() -> None:
    """Equity on days 0-4: 100, 110, 121, 110, 132. Split at day 2.

    In-sample:  days 0-1, 100 → 110, so total return 0.1.
    Out-of-sample: starts from the last in-sample bar (day 1, 110), so the
                first return spans the split: 110 → 132, total return 0.2.
    Trades: BUY on day 1 (in-sample), SELL on day 2 (out-of-sample). The trip
    closes after the split, so it counts out-of-sample (P&L +97.9, a win), and
    in-sample has no closed trip (win rate NaN). Each side has 1 fill.
    """
    equity = pl.DataFrame({"timestamp": _days(5), "equity": [100.0, 110.0, 121.0, 110.0, 132.0]})
    trades = _trade_log([("buy", 1000.0, 1.0, 10.0), ("sell", 1100.0, 1.1, 0.0)])  # fills on days 1, 2

    ins, oos = split_metrics(equity, trades, _day(2).date(), periods_per_year=252)

    assert ins.total_return == pytest.approx(0.1, rel=REL)
    assert math.isnan(ins.win_rate)
    assert (ins.num_round_trips, ins.num_trades) == (0, 1)

    assert oos.total_return == pytest.approx(0.2, rel=REL)
    assert oos.win_rate == 1.0
    assert (oos.num_round_trips, oos.num_trades) == (1, 1)


@pytest.mark.parametrize("split_day", [0, 1, 4, 30])
def test_split_needs_two_bars_each_side(split_day: int) -> None:
    equity = pl.DataFrame({"timestamp": _days(4), "equity": [100.0, 101.0, 102.0, 103.0]})
    with pytest.raises(ValueError, match="at least 2 bars"):
        split_metrics(equity, _trade_log([]), _day(split_day).date(), periods_per_year=252)
