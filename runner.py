"""The correct, look-ahead-safe, bar-by-bar backtest runner.

This module enforces the core rule: a strategy never receives any row whose
timestamp is later than the current simulated bar. The guarantee is
structural, and each piece below names the part of it that it implements:

    1. Full-dataset isolation: ``run_backtest`` owns the full frame. It is never
       passed to, stored on, or reachable from the strategy.
    2. One choke point: ``history_up_to`` is the ONLY place that builds what
       the strategy sees. It slices rows 0..t and verifies the slice ends at
       bar t, raising ``LookAheadError`` otherwise. Reviewers: start there.
    3. Narrow interface: the strategy gets only that slice and
       ``current_bar``, which is taken from the slice's own last row, and
       returns a ``Signal``.
    4. Next-bar execution: a signal from bar t is filled at bar t+1's open,
       never at bar t's close (which the strategy used to decide). A signal
       from the last bar is dropped.

v1 scope: one symbol per run, long-only (position >= 0), fractional quantities.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from config import DEFAULT_BUY_FRACTION, IMPACT_LOOKBACK
from data_loader import validate_canonical
from strategies.base_strategy import BaseStrategy, Side, Signal


class LookAheadError(RuntimeError):
    """The slice about to be handed to a strategy would contain future rows."""


@dataclass(frozen=True)
class CostModel:
    """Trading costs applied to every fill.

    There is no frictionless mode: fee and slippage must be strictly positive.

    Every fill's price moves against the trader by ``slippage_rate + impact``, where
    ``impact = impact_coefficient * sigma * sqrt(order notional / ADV)``: the
    empirical square-root law of market impact. ``sigma`` (daily-return volatility)
    and ``ADV`` (average daily dollar volume) come from the bars BEFORE the fill bar
    (see ``_market_conditions``). An order is capped at ``max_participation * ADV``
    of notional per bar; the rest is carried to the next bar.

    Attributes:
        fee_rate: Fraction of fill notional charged as a fee (0.001 = 10 bps).
        slippage_rate: Flat price move against the trader, i.e. half the bid-ask spread (0.0005 = 5 bps).
        impact_coefficient: The square-root law's constant, empirically of order 1. 0 disables impact.
        max_participation: Largest fraction of ADV one bar's fill may be, in (0, 1]. 1 = a full day's average volume.
    """

    fee_rate: float
    slippage_rate: float
    impact_coefficient: float = 0.0
    max_participation: float = 1.0

    def __post_init__(self) -> None:
        for label, rate in (("fee_rate", self.fee_rate), ("slippage_rate", self.slippage_rate)):
            if not 0 < rate < 1:
                raise ValueError(f"{label} must be in (0, 1), got {rate}. There is no frictionless mode.")
        if self.impact_coefficient < 0:
            raise ValueError(f"impact_coefficient must be >= 0, got {self.impact_coefficient}")
        if not 0 < self.max_participation <= 1:
            raise ValueError(f"max_participation must be in (0, 1], got {self.max_participation}")


_TS = pl.Datetime("us", "UTC")

# One row per executed trade.
TRADE_LOG_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": _TS,  # the bar whose open the trade filled at (bar t+1)
    "signal_timestamp": _TS,  # the bar whose signal caused it (bar t): makes the next-bar rule visible
    "symbol": pl.String(),
    "side": pl.String(),  # "buy" | "sell" (HOLD never trades)
    "price": pl.Float64(),  # the bar's open, moved against the trader by slippage
    "quantity": pl.Float64(),  # units traded, > 0
    "notional": pl.Float64(),  # price * quantity
    "fee": pl.Float64(),  # notional * fee_rate
    "slippage_cost": pl.Float64(),  # abs(price - open) * quantity: spread + impact
    "impact_cost": pl.Float64(),  # the market-impact part of slippage_cost
    "position_after": pl.Float64(),
    "cash_after": pl.Float64(),
}

EQUITY_CURVE_SCHEMA: dict[str, pl.DataType] = {
    "timestamp": _TS,
    "close": pl.Float64(),
    "cash": pl.Float64(),
    "position": pl.Float64(),
    "equity": pl.Float64(),
}


@dataclass(frozen=True)
class BacktestResult:
    """Output of a backtest run. It is the input to ``metrics.compute_metrics``.

    Attributes:
        equity_curve: One row per bar (``EQUITY_CURVE_SCHEMA``). ``equity`` is
            cash + position marked at that bar's close.
        trades: One row per fill (``TRADE_LOG_SCHEMA``). May be empty.
    """

    equity_curve: pl.DataFrame
    trades: pl.DataFrame


@dataclass
class _Portfolio:
    """Mutable portfolio state, private to the runner. Never handed to a strategy."""

    cash: float
    position: float


# =============================================================================
# THE LOOK-AHEAD CHOKE POINT
# =============================================================================


def history_up_to(data: pl.DataFrame, t: int) -> pl.DataFrame:
    """Return rows 0..t of ``data``, and verify the result holds nothing after bar t.

    This is the only function that builds the data a strategy sees. If the
    look-ahead guarantee is broken anywhere, it's broken here, so this is the
    function to review.

    Args:
        data: The full, validated bar frame (sorted, unique timestamps).
        t: Index of the current bar.

    Returns:
        A new DataFrame whose last row is bar t.

    Raises:
        IndexError: if ``t`` is outside ``data``.
        LookAheadError: if the slice does not end exactly at bar t.
    """
    # A negative t would slice from the END of the frame (Python-style
    # negative indexing), which is the last rows of the dataset, i.e. the
    # future. Reject anything out of range explicitly.
    if not 0 <= t < data.height:
        raise IndexError(f"bar index {t} out of range for {data.height} bars")

    # The slice itself: `offset=0, length=t + 1` gives rows 0, 1, ..., t.
    #   * `t + 1` rather than `t`: the current bar IS included. The strategy
    #     decides after bar t's close, so it may see that close. That is safe
    #     because the resulting signal is only filled at bar t+1's open
    #     (see `run_backtest`).
    #   * The object returned is a new DataFrame containing exactly those
    #     rows. There is no index, cursor or reference back to `data`, so the
    #     strategy has no way to reach row t+1 or later.
    #   * Polars slices are zero-copy but copy-on-write: if a strategy mutates
    #     its slice, Polars copies first, and `data` is untouched.
    history = data.slice(0, t + 1)

    # Defense in depth: check the result, rather than trusting the slice call.
    # These run on every bar and use `raise` rather than `assert`, because
    # `python -O` strips asserts. Since `data` was validated as strictly
    # increasing, "the last row is bar t" also means "no row is later than bar t".
    if history.height != t + 1:
        raise LookAheadError(f"history for bar {t} has {history.height} rows, expected {t + 1}")
    last_ts = history["timestamp"][-1]
    current_ts = data["timestamp"][t]
    if last_ts != current_ts:
        raise LookAheadError(f"history for bar {t} ends at {last_ts}, but the current bar is {current_ts}")

    return history


def validate_inputs(data: pl.DataFrame, initial_capital: float) -> None:
    """Shared by both runners: canonical, non-empty, one symbol, positive capital."""
    validate_canonical(data)
    if data.is_empty():
        raise ValueError("data is empty")
    if data["symbol"].n_unique() != 1:
        raise ValueError(f"runner handles one symbol per run, got {data['symbol'].unique().to_list()}")
    if not initial_capital > 0:
        raise ValueError(f"initial_capital must be > 0, got {initial_capital}")


# =============================================================================
# THE LOOP
# =============================================================================


def run_backtest(
    data: pl.DataFrame,
    strategy: BaseStrategy,
    costs: CostModel,
    initial_capital: float,
) -> BacktestResult:
    """Simulate ``strategy`` over ``data`` one bar at a time, without look-ahead.

    For each bar t, in time order:
        1. If bar t-1 produced a signal, execute it at bar t's OPEN (with
           slippage and fees). This is the first price after the decision.
        2. Mark the portfolio to bar t's CLOSE and record equity.
        3. Hand the strategy ``history_up_to(data, t)`` and collect its signal,
           to be executed at bar t+1's open.
    A signal returned on the last bar is dropped, since there is no later bar
    to fill it. An open position at the end stays marked at the last close.

    Args:
        data: Canonical bars for one symbol (see ``data_loader``). Validated here.
        strategy: The strategy. It only ever sees ``history_up_to`` slices.
        costs: Fee/slippage model, applied to every fill.
        initial_capital: Starting cash, > 0.

    Returns:
        The equity curve (one row per bar) and the trade log (one row per fill).
    """
    validate_inputs(data, initial_capital)

    # The runner reads prices from these plain lists. They belong to the
    # runner and are never passed to the strategy.
    timestamps = data["timestamp"].to_list()
    opens = data["open"].to_list()
    closes = data["close"].to_list()
    symbol = data["symbol"][0]
    sigmas, advs = _market_conditions(data)

    state = _Portfolio(cash=float(initial_capital), position=0.0)
    pending: Signal | None = None  # signal to fill at this bar's open
    pending_ts = None  # the bar that signal came from
    fills: list[dict] = []
    equity_rows: list[dict] = []

    # An explicit loop, deliberately not vectorized. Vectorized backtests
    # (e.g. `signal = sma_short > sma_long` over the whole column) are where
    # off-by-one shifts silently leak the future. Here, time only moves
    # forward one bar at a time, and each step can only use what it's handed.
    for t in range(data.height):
        # 1. Execute the previous bar's decision at this bar's open.
        if pending is not None:
            fill, capped = _execute(
                pending,
                bar_open=opens[t],
                sigma=sigmas[t],
                adv=advs[t],
                timestamp=timestamps[t],
                signal_timestamp=pending_ts,
                symbol=symbol,
                costs=costs,
                state=state,
            )
            if fill is not None:
                fills.append(fill)
            if not capped:  # a capped order keeps working next bar, unless a new signal replaces it
                pending = None

        # 2. Mark to market at this bar's close.
        equity_rows.append(_mark(state, timestamps[t], closes[t]))

        # 3. Ask the strategy for a decision, showing it rows 0..t only.
        history = history_up_to(data, t)
        signal = strategy.on_bar(history, history.row(-1, named=True))
        if not isinstance(signal, Signal):
            raise TypeError(f"{type(strategy).__name__}.on_bar must return a Signal, got {signal!r}")
        if signal.side is not Side.HOLD:
            pending, pending_ts = signal, timestamps[t]

    # A `pending` signal left over here came from the final bar (or is an unfilled remainder). It is dropped.

    return BacktestResult(
        equity_curve=pl.DataFrame(equity_rows, schema=EQUITY_CURVE_SCHEMA),
        trades=pl.DataFrame(fills, schema=TRADE_LOG_SCHEMA),
    )


def _mark(state: _Portfolio, timestamp: datetime, close: float) -> dict:
    """One equity-curve row: the portfolio valued at ``close``."""
    return {
        "timestamp": timestamp,
        "close": close,
        "cash": state.cash,
        "position": state.position,
        "equity": state.cash + state.position * close,
    }


def _market_conditions(data: pl.DataFrame) -> tuple[list[float | None], list[float | None]]:
    """Per bar t: daily-return volatility and average dollar volume over the
    ``IMPACT_LOOKBACK`` bars strictly BEFORE t.

    Execution-side only; never shown to a strategy. The ``shift(1)`` is what keeps
    it causal: a fill at bar t's open can't know bar t's volume or close yet. Both
    are ``None`` until enough history exists (then impact is 0 and nothing is capped).
    """
    df = data.select(
        pl.col("close").pct_change().rolling_std(IMPACT_LOOKBACK, min_samples=2).shift(1).alias("sigma"),
        (pl.col("close") * pl.col("volume")).rolling_mean(IMPACT_LOOKBACK, min_samples=1).shift(1).alias("adv"),
    )
    return df["sigma"].to_list(), df["adv"].to_list()


def _impact(costs: CostModel, sigma: float | None, adv: float | None, notional: float) -> float:
    """Square-root market impact as a fraction of price. 0 without enough history."""
    if not (costs.impact_coefficient and sigma and adv):
        return 0.0
    return costs.impact_coefficient * sigma * math.sqrt(notional / adv)


def _execute(
    signal: Signal,
    *,
    bar_open: float,
    sigma: float | None,
    adv: float | None,
    timestamp: datetime,
    signal_timestamp: datetime,
    symbol: str,
    costs: CostModel,
    state: _Portfolio,
) -> tuple[dict | None, bool]:
    """Execute ``signal`` at ``bar_open`` with spread, impact and fees. Updates ``state`` in place.

    * BUY spends ``size`` (default ``config.DEFAULT_BUY_FRACTION``) of cash.
      The budget covers notional AND fee, so ``notional + fee == budget`` and
      cash can never go negative.
    * SELL sells ``size`` (default all) of the position. Long-only: it never goes short.
    * Either is capped at ``costs.max_participation * adv`` of notional.

    Returns:
        ``(row, capped)``: one trade-log row (``TRADE_LOG_SCHEMA``), or ``None`` if there
        was nothing to do; and whether the participation cap cut the order short.
    """
    cap = costs.max_participation * adv if adv else math.inf
    if signal.side is Side.BUY:
        fraction = signal.size if signal.size is not None else DEFAULT_BUY_FRACTION
        budget = state.cash * fraction
        if budget <= 0:
            return None, False
        capped = budget > cap * (1 + costs.fee_rate)
        if capped:
            budget = cap * (1 + costs.fee_rate)
        notional = budget / (1 + costs.fee_rate)  # so notional + notional * fee_rate == budget
        fee = budget - notional
        impact = _impact(costs, sigma, adv, notional)
        price = bar_open * (1 + costs.slippage_rate + impact)  # pay up: spread + impact against us
        quantity = notional / price
        state.cash -= budget
        state.position += quantity

    elif signal.side is Side.SELL:
        fraction = signal.size if signal.size is not None else 1.0
        quantity = state.position * fraction
        if quantity <= 0:
            return None, False
        capped = quantity * bar_open > cap
        if capped:
            quantity = cap / bar_open
        impact = _impact(costs, sigma, adv, quantity * bar_open)
        factor = 1 - costs.slippage_rate - impact  # receive less: spread + impact against us
        if factor <= 0:
            # A price at or below zero would make selling cost money. Refuse rather than book nonsense.
            raise ValueError(
                f"market impact of {impact:.0%} would push the sell price to zero or below; "
                "lower the impact coefficient or the volume cap"
            )
        price = bar_open * factor
        notional = quantity * price
        fee = notional * costs.fee_rate
        state.cash += notional - fee
        state.position -= quantity

    else:
        return None, False

    return {
        "timestamp": timestamp,
        "signal_timestamp": signal_timestamp,
        "symbol": symbol,
        "side": signal.side.value,
        "price": price,
        "quantity": quantity,
        "notional": notional,
        "fee": fee,
        "slippage_cost": abs(price - bar_open) * quantity,
        "impact_cost": bar_open * impact * quantity,
        "position_after": state.position,
        "cash_after": state.cash,
    }, capped
