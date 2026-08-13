"""Abstract Strategy interface.

The one-bar-at-a-time contract on ``on_market_event`` is itself the
no-look-ahead enforcement mechanism at this layer: a Strategy is never
handed a DataFrame or any other back-channel to history or future bars, so
any indicator it computes (a moving average, say) must be built from its own
internal rolling state as bars arrive one at a time -- exactly like it would
have to in a live/streaming system. There is nothing in this base class that
lets a subclass reach around that constraint.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from engine.events import MarketEvent, SignalEvent


class Strategy(ABC):
    """Base class for strategies. Subclasses hold their own rolling state."""

    def __init__(self, strategy_id: str) -> None:
        self.strategy_id = strategy_id

    @abstractmethod
    def on_market_event(self, event: MarketEvent) -> SignalEvent | None:
        """Process one bar and optionally emit a directional call.

        Called once per MarketEvent, in chronological order, for every
        symbol the strategy is meant to track. Must not raise on an
        undersized history -- return None until enough bars have arrived.
        """
        ...
