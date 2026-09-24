"""The single source of truth for every performance number.

No strategy, runner, script, notebook or test may compute its own version of
any metric. Import it from here. Each function's docstring gives its exact
definition, so runs are always comparable.

Inputs are the runner's outputs:
    * equity curve: one row per bar, with at least ``timestamp`` and ``equity``
      (cash + position marked at that bar's close). Row 0 is the initial
      capital, because nothing can fill before bar 1.
    * trade log: one row per fill, with at least ``side``, ``notional``, ``fee``
      and ``position_after`` (see ``runner.TRADE_LOG_SCHEMA``).

Undefined values are ``NaN``, never a misleading 0: e.g. Sharpe of a flat
equity curve, or win rate with no closed trades.
"""

import math
from dataclasses import dataclass
from datetime import UTC, date, datetime

import polars as pl

from config import RISK_FREE_RATE

_SECONDS_PER_YEAR = 365.25 * 86_400


@dataclass(frozen=True)
class Drawdown:
    """The largest peak-to-trough decline, and when it happened.

    Attributes:
        depth: ``trough / peak - 1``, a fraction <= 0 (e.g. -0.35). 0.0 if equity never declined.
        peak: When equity last stood at the high the decline started from.
        trough: When the decline bottomed out.
        recovery: First time after the trough that equity got back to the
            peak value, or ``None`` if it never did.
    """

    depth: float
    peak: datetime | None
    trough: datetime | None
    recovery: datetime | None


@dataclass(frozen=True)
class Metrics:
    """Every performance number reported for a run.

    Attributes:
        total_return: See ``total_return``.
        cagr: See ``cagr``.
        sharpe_ratio: See ``sharpe_ratio``.
        max_drawdown: See ``max_drawdown``.
        win_rate: See ``win_rate``, as a fraction in [0, 1].
        num_round_trips: Closed round trips counted by ``win_rate``.
        num_trades: Total fills in the trade log.
    """

    total_return: float
    cagr: float
    sharpe_ratio: float
    max_drawdown: Drawdown
    win_rate: float
    num_round_trips: int
    num_trades: int


def compute_metrics(
    equity_curve: pl.DataFrame,
    trades: pl.DataFrame,
    periods_per_year: int,
    risk_free_rate: float = RISK_FREE_RATE,
    since: datetime | None = None,
) -> Metrics:
    """Compute every metric for one run (or one period of it, see ``split_metrics``).

    Args:
        equity_curve: The runner's equity curve (needs ``timestamp``, ``equity``).
        trades: The runner's trade log (needs ``side``, ``notional``, ``fee``, ``position_after``).
        periods_per_year: Bars per year, used to annualize Sharpe. Use
            ``config.PERIODS_PER_YEAR`` (365 for 24/7 crypto; 252 would suit exchange-traded assets).
        risk_free_rate: Annual risk-free rate for Sharpe (default ``config.RISK_FREE_RATE`` = 0).
        since: If set, only fills at or after it count as trades, and only round
            trips that close at or after it count for win rate. ``trades`` must
            still hold the full log, so trips opened earlier are priced correctly.
    """
    equity = equity_curve["equity"]
    timestamps = equity_curve["timestamp"]
    pnls = round_trip_pnls(trades, closed_from=since)
    return Metrics(
        total_return=total_return(equity),
        cagr=cagr(equity, timestamps),
        sharpe_ratio=sharpe_ratio(equity, periods_per_year, risk_free_rate),
        max_drawdown=max_drawdown(equity, timestamps),
        win_rate=win_rate(pnls),
        num_round_trips=len(pnls),
        num_trades=trades.height if since is None else trades.filter(pl.col("timestamp") >= since).height,
    )


def excess_return(strategy: Metrics, benchmark: Metrics) -> float:
    """Strategy total return minus benchmark total return over the same period (e.g. 0.25 = 25 points better)."""
    return strategy.total_return - benchmark.total_return


def split_metrics(
    equity_curve: pl.DataFrame,
    trades: pl.DataFrame,
    split: date,
    periods_per_year: int,
    risk_free_rate: float = RISK_FREE_RATE,
) -> tuple[Metrics, Metrics]:
    """Metrics before ``split`` (in-sample) and from ``split`` on (out-of-sample).

    Tune parameters on the in-sample period only. The out-of-sample numbers
    are the honest estimate: the strategy was never chosen by looking at them.

    * In-sample: equity bars before ``split``, and fills before ``split``.
      A trip still open at the split is not counted.
    * Out-of-sample: the equity curve from the last in-sample bar on, so the
      first return spans the split and total return is measured from the
      equity held at the split. Fills from ``split`` on. A round trip counts
      in the period in which it closes, so a trip opened before the split and
      closed after it is out-of-sample.

    Raises:
        ValueError: if either side would have fewer than 2 equity points.
    """
    split_ts = datetime(split.year, split.month, split.day, tzinfo=UTC)
    before = equity_curve.filter(pl.col("timestamp") < split_ts)
    if before.height < 2 or before.height == equity_curve.height:
        raise ValueError(f"split date {split} must leave at least 2 bars on each side of the equity curve")
    in_sample = compute_metrics(before, trades.filter(pl.col("timestamp") < split_ts), periods_per_year, risk_free_rate)
    out_of_sample = compute_metrics(
        equity_curve.slice(before.height - 1), trades, periods_per_year, risk_free_rate, since=split_ts
    )
    return in_sample, out_of_sample


# --- Return metrics ----------------------------------------------------------


def total_return(equity: pl.Series) -> float:
    """``equity[-1] / equity[0] - 1``. E.g. 10,000 → 12,500 gives 0.25."""
    _check_equity(equity)
    return equity[-1] / equity[0] - 1


def cagr(equity: pl.Series, timestamps: pl.Series) -> float:
    """Compound annual growth rate over the actual calendar span of the curve.

    ``(equity[-1] / equity[0]) ** (1 / years) - 1``, where
    ``years = (timestamps[-1] - timestamps[0]) / 365.25 days``.

    Calendar time is used (not bar count), so the result is the same for
    daily stock bars (which skip weekends) and 24/7 crypto bars. ``NaN`` if
    the curve spans zero time.
    """
    _check_equity(equity)
    years = (timestamps[-1] - timestamps[0]).total_seconds() / _SECONDS_PER_YEAR
    if years <= 0:
        return math.nan
    return (equity[-1] / equity[0]) ** (1 / years) - 1


# --- Risk metrics ------------------------------------------------------------


def max_drawdown(equity: pl.Series, timestamps: pl.Series) -> Drawdown:
    """Largest peak-to-trough decline of the equity curve.

    ``depth = min over t of (equity[t] / max(equity[0..t]) - 1)``.
      * trough: the bar with that minimum (the first, if tied).
      * peak: the LAST bar at or before the trough where equity equaled the
        running max. On a plateau, the drawdown starts when equity last stood
        at the high.
      * recovery: the first bar after the trough with equity >= the peak value, else ``None``.
    If equity never declines: ``Drawdown(0.0, None, None, None)``.
    """
    _check_equity(equity)
    depths = equity / equity.cum_max() - 1
    trough = depths.arg_min()  # first index of the minimum
    if depths[trough] >= 0:
        return Drawdown(0.0, None, None, None)

    values = equity.to_list()
    ts = timestamps.to_list()
    peak_value = max(values[: trough + 1])
    peak = max(i for i in range(trough + 1) if values[i] == peak_value)
    recovery = next((i for i in range(trough + 1, len(values)) if values[i] >= peak_value), None)
    return Drawdown(
        depth=depths[trough],
        peak=ts[peak],
        trough=ts[trough],
        recovery=ts[recovery] if recovery is not None else None,
    )


def sharpe_ratio(equity: pl.Series, periods_per_year: int, risk_free_rate: float = RISK_FREE_RATE) -> float:
    """Annualized Sharpe ratio of per-bar returns.

    * per-bar simple returns: ``r[i] = equity[i] / equity[i-1] - 1``
    * excess returns: ``x = r - risk_free_rate / periods_per_year``
    * ``sharpe = sqrt(periods_per_year) * mean(x) / std(x)``, with sample std (ddof=1)

    Returns are derived here from the equity curve, so no caller ever
    computes returns its own way. ``NaN`` if the std is zero or there are
    fewer than 2 returns (e.g. a strategy that never trades has a flat curve,
    and its Sharpe is undefined, not 0).
    """
    _check_equity(equity)
    if periods_per_year <= 0:
        raise ValueError(f"periods_per_year must be > 0, got {periods_per_year}")
    returns = equity.slice(1) / equity.slice(0, equity.len() - 1) - 1
    excess = returns - risk_free_rate / periods_per_year
    std = excess.std(ddof=1)
    if std is None or std == 0:
        return math.nan
    return math.sqrt(periods_per_year) * excess.mean() / std


# --- Trade metrics -----------------------------------------------------------


def round_trip_pnls(trades: pl.DataFrame, closed_from: datetime | None = None) -> list[float]:
    """Net P&L of each closed round trip, in order (only trips closing at or after ``closed_from``, if set).

    A round trip starts with the fill that opens a position from flat and
    ends with the fill that brings ``position_after`` back to exactly 0.0.
    The runner sells ``position * 1.0`` on a full exit, so the zero is exact.
    Partial exits belong to the same trip.

    ``pnl = Σ over SELLs (notional - fee) - Σ over BUYs (notional + fee)``.
    Fees are included explicitly, and slippage is already in the fill prices,
    so this is the P&L after all costs. A trip still open at the end is not included.
    """
    pnls: list[float] = []
    pnl = 0.0
    for row in trades.iter_rows(named=True):
        if row["side"] == "buy":
            pnl -= row["notional"] + row["fee"]
        elif row["side"] == "sell":
            pnl += row["notional"] - row["fee"]
        else:
            raise ValueError(f"unexpected trade side {row['side']!r}")
        if row["position_after"] == 0.0:
            if closed_from is None or row["timestamp"] >= closed_from:
                pnls.append(pnl)
            pnl = 0.0
    return pnls


def win_rate(pnls: list[float]) -> float:
    """Fraction of closed round trips with P&L > 0 after all costs, in [0, 1] (0.6 = 60%).

    ``pnls`` comes from ``round_trip_pnls``. A trip that is profitable before
    fees but not after counts as a loss. ``NaN`` if there are no closed round trips.
    """
    if not pnls:
        return math.nan
    return sum(p > 0 for p in pnls) / len(pnls)


# --- Validation --------------------------------------------------------------


def _check_equity(equity: pl.Series) -> None:
    if equity.len() < 2:
        raise ValueError(f"need at least 2 equity points, got {equity.len()}")
    if equity.null_count() or (equity <= 0).any():
        raise ValueError("equity must be strictly positive with no nulls")
