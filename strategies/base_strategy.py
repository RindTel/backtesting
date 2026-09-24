"""The strategy interface: ``BaseStrategy.on_bar`` and the ``Signal`` it returns.

WHY THIS SIGNATURE PREVENTS LOOK-AHEAD BIAS
===========================================

Look-ahead bias means a strategy's decision at some bar is influenced by
prices from later bars, which a real trader could never have seen. It is the
easiest way to make a backtest look far better than reality.

This interface is designed so that a strategy has no way to see the future,
even by accident:

1. **A strategy's only view of the market is the arguments of ``on_bar``.**
   It is never given the full dataset, the runner, the data loader or a file
   path, so there is nothing else it could read from.

2. **``historical_data_up_to_now`` ends at the current bar.** On bar t the
   runner hands over a DataFrame holding rows 0..t and nothing after t. The
   future rows are not hidden or "off limits"; they are simply not in the
   object the strategy holds. Any calculation the strategy does, whether a
   rolling mean, a max or an index lookup, can only ever touch the past and
   the present.

3. **``current_bar`` is just that same last row,** given as a dict for
   convenience. The runner builds it from the slice itself, so the two can
   never disagree, and it is never a peek at the next bar.

4. **The strategy returns a decision, not a trade.** A ``Signal`` issued
   after seeing bar t's close is filled by the runner at bar t+1's open, with
   fees and slippage. The strategy never chooses a fill price, so it cannot
   trade at a price it used to make the decision.

**The runner is responsible for making the slice correct.** It builds a fresh
``rows 0..t`` slice every bar and asserts that the slice's last timestamp is
the current bar's timestamp. This file defines the contract, and ``runner.py``
enforces it. ``tests/test_no_lookahead.py`` proves it.

Rules for strategy authors:
    * Compute everything from the arguments you're given, on every call.
      Don't cache DataFrames between calls or precompute indicators over a
      full dataset in ``__init__``.
    * Don't import ``runner``, ``broken_runner``, ``data_loader``,
      ``mlflow_logger`` or ``metrics``, and don't read files.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

import polars as pl


class Side(Enum):
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"


@dataclass(frozen=True)
class Signal:
    """A strategy's decision for the current bar.

    The runner acts on it at the NEXT bar's open, with fees and slippage.

    Attributes:
        side: BUY, SELL or HOLD.
        size: Optional fraction in (0, 1]. Its meaning depends on ``side``:
            * BUY: fraction of available cash to spend. ``None`` means
              ``config.DEFAULT_BUY_FRACTION``.
            * SELL: fraction of the current position to sell. ``None`` means
              the whole position.
            * HOLD: must be ``None``.
    """

    side: Side
    size: float | None = None

    def __post_init__(self) -> None:
        if self.side is Side.HOLD and self.size is not None:
            raise ValueError("A HOLD signal cannot have a size")
        if self.size is not None and not 0 < self.size <= 1:
            raise ValueError(f"Signal size must be in (0, 1], got {self.size}")


HOLD = Signal(Side.HOLD)


class BaseStrategy(ABC):
    """Abstract base for all strategies.

    Subclasses set ``name`` (CLI id), ``label`` (display name) and ``summary``
    (one plain-English sentence for the UI), pass their tunable parameters to
    ``super().__init__`` (so they get logged to MLflow), and implement ``on_bar``.
    Tunable parameters are keyword arguments of ``__init__`` with int defaults;
    the UI builds its inputs from that signature.
    """

    name: ClassVar[str]
    label: ClassVar[str]
    summary: ClassVar[str]

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)

    @property
    def params(self) -> dict[str, Any]:
        """All tunable parameters, as a flat dict of MLflow-loggable values."""
        return dict(self._params)

    @abstractmethod
    def on_bar(self, historical_data_up_to_now: pl.DataFrame, current_bar: dict) -> Signal:
        """Decide what to do after observing the current bar t.

        This is the only method the runner calls, once per bar.

        Args:
            historical_data_up_to_now: Canonical bars for rows 0..t, in time
                order. The last row IS the current bar, and there are no rows after it.
            current_bar: The last row of ``historical_data_up_to_now`` as a
                dict (``timestamp``, ``open``, ``high``, ``low``, ``close``,
                ``volume``, ``symbol``).

        Returns:
            A ``Signal``, which the runner acts on at bar t+1's open.
        """


def require_positive_ints(**values: Any) -> None:
    """Raise ``ValueError`` unless every value is a positive ``int`` (``bool`` excluded)."""
    for label, value in values.items():
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{label} must be a positive int, got {value!r}")
