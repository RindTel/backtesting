"""
################################################################################
#                                                                              #
#        DELIBERATELY BROKEN — LEAKS LOOK-AHEAD BIAS — DEMO ONLY               #
#                                                                              #
#   Do NOT use this for real results. Do NOT "fix" it. Its bug is its point.  #
#                                                                              #
################################################################################

A copy of the runner loop with two classic look-ahead mistakes built in, kept
so the demo can show side by side how leakage inflates backtest results:

    LEAK 1: an off-by-one slice. On bar t the strategy receives rows 0..t+1
            (``data.slice(0, t + 2)``), not ``runner.history_up_to(data, t)``,
            so its indicators already include tomorrow's bar. This is the
            classic, easy-to-miss version of the bug: the results look
            plausible, just better than they should be.
    LEAK 2: signals fill at the SAME bar's close that generated them, not at
            bar t+1's open as in ``runner.run_backtest``.

Everything else is shared with ``runner.run_backtest``: the same input
validation, the same equity marking (``runner._mark``) and the same order
execution (``runner._execute``), so fees, slippage and impact still apply.
One non-leak difference: when the volume cap cuts an order
short, the broken runner drops the remainder instead of carrying it to the
next bar. That only matters for accounts large enough to hit the cap
(about $12M+ on the sample); below that, every difference is the leaks.

Isolation rules:
    * Only ``scripts/run_backtest.py --runner broken`` and the leakage-detection
      test in ``tests/test_no_lookahead.py`` may import this module.
    * ``runner.py`` and every strategy must never import it.
    * Its runs are logged with ``mlflow_logger.log_backtest_run(runner="broken")``,
      which tags them ``runner=broken``, names them ``BROKEN-<strategy>`` and
      adds a warning to the run description.
"""

import polars as pl

from runner import (
    EQUITY_CURVE_SCHEMA,
    TRADE_LOG_SCHEMA,
    BacktestResult,
    CostModel,
    _execute,
    _mark,
    _market_conditions,
    _Portfolio,
    validate_inputs,
)
from strategies.base_strategy import BaseStrategy, Side, Signal


def run_backtest_leaky(
    data: pl.DataFrame,
    strategy: BaseStrategy,
    costs: CostModel,
    initial_capital: float,
) -> BacktestResult:
    """DELIBERATELY BROKEN: same signature as ``runner.run_backtest``, but it leaks.

    For each bar t it passes rows 0..t+1 (LEAK 1: one bar from the future) to
    ``strategy.on_bar``, and fills the resulting signal at bar t's close
    (LEAK 2), with fees and slippage.

    Returns:
        A ``BacktestResult`` whose metrics are expected to beat the honest
        runner's on the same data, which is the point of the demo.
    """
    validate_inputs(data, initial_capital)

    timestamps = data["timestamp"].to_list()
    closes = data["close"].to_list()
    symbol = data["symbol"][0]
    sigmas, advs = _market_conditions(data)

    state = _Portfolio(cash=float(initial_capital), position=0.0)
    fills: list[dict] = []
    equity_rows: list[dict] = []

    for t in range(data.height):
        # ======================================================================
        # LEAK 1: OFF-BY-ONE SLICE, THE STRATEGY SEES TOMORROW'S BAR.
        #   Correct (runner.py):  history = history_up_to(data, t)  # rows 0..t
        #   Broken (here):        rows 0..t+1, with no check at all.
        # `current_bar` is then the slice's last row, which is tomorrow.
        # ======================================================================
        history = data.slice(0, t + 2)
        signal = strategy.on_bar(history, history.row(-1, named=True))
        if not isinstance(signal, Signal):
            raise TypeError(f"{type(strategy).__name__}.on_bar must return a Signal, got {signal!r}")

        # ======================================================================
        # LEAK 2: SAME-BAR FILL, TRADING AT THE PRICE THE DECISION WAS MADE FROM.
        #   Correct (runner.py):  the signal from bar t waits and fills at bar t+1's OPEN.
        #   Broken (here):        it fills right now, at bar t's CLOSE.
        # ======================================================================
        if signal.side is not Side.HOLD:
            # (A capped order's remainder is simply dropped here; only matters for huge accounts.)
            fill, _capped = _execute(
                signal,
                bar_open=closes[t],  # LEAK 2: bar t's close used as the execution price
                sigma=sigmas[t],
                adv=advs[t],
                timestamp=timestamps[t],
                signal_timestamp=timestamps[t],
                symbol=symbol,
                costs=costs,
                state=state,
            )
            if fill is not None:
                fills.append(fill)

        equity_rows.append(_mark(state, timestamps[t], closes[t]))

    return BacktestResult(
        equity_curve=pl.DataFrame(equity_rows, schema=EQUITY_CURVE_SCHEMA),
        trades=pl.DataFrame(fills, schema=TRADE_LOG_SCHEMA),
    )
