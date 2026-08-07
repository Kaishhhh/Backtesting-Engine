"""Event types for the event-driven backtesting engine.

All events are immutable (frozen, slotted) dataclasses. Every event carries a
``timestamp`` because the EventQueue enforces strictly chronological
processing on that field -- see the no-look-ahead invariant in CLAUDE.md.

The four event types mirror the pipeline described in CLAUDE.md:

    MarketEvent -> Strategy   -> SignalEvent
    SignalEvent -> Portfolio  -> OrderEvent
    OrderEvent  -> Execution  -> FillEvent
    FillEvent   -> Portfolio updates cash/positions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Literal


class EventType(Enum):
    """Discriminant for dispatching events without isinstance chains."""

    MARKET = auto()
    SIGNAL = auto()
    ORDER = auto()
    FILL = auto()


@dataclass(frozen=True, slots=True)
class MarketEvent:
    """New bar of market data for a symbol has arrived.

    This is the only event type produced from outside the system (by a data
    handler feeding historical bars in time order); every other event is
    derived, directly or indirectly, from a MarketEvent.
    """

    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    event_type: EventType = field(default=EventType.MARKET, init=False)

    def __post_init__(self) -> None:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open ({self.open}) outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close ({self.close}) outside [low, high]")
        if self.volume < 0:
            raise ValueError(f"volume ({self.volume}) is negative")


@dataclass(frozen=True, slots=True)
class SignalEvent:
    """A Strategy's directional call for a symbol, derived from a MarketEvent.

    Carries no size/quantity -- sizing is the Portfolio's job (it turns a
    SignalEvent into an OrderEvent), keeping strategy logic decoupled from
    position sizing and risk management.
    """

    timestamp: datetime
    symbol: str
    direction: Literal["LONG", "SHORT", "EXIT"]
    strategy_id: str
    strength: float = 1.0
    event_type: EventType = field(default=EventType.SIGNAL, init=False)


@dataclass(frozen=True, slots=True)
class OrderEvent:
    """An order the Portfolio wants the Execution handler to fill."""

    timestamp: datetime
    symbol: str
    order_type: Literal["MARKET", "LIMIT"]
    quantity: int
    direction: Literal["BUY", "SELL"]
    limit_price: float | None = None
    event_type: EventType = field(default=EventType.ORDER, init=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity ({self.quantity}) must be positive")
        if self.order_type == "LIMIT" and self.limit_price is None:
            raise ValueError("LIMIT orders require a limit_price")


@dataclass(frozen=True, slots=True)
class FillEvent:
    """Confirmation that an OrderEvent was executed, including cost of trading."""

    timestamp: datetime
    symbol: str
    quantity: int
    direction: Literal["BUY", "SELL"]
    fill_price: float
    commission: float
    slippage: float = 0.0
    event_type: EventType = field(default=EventType.FILL, init=False)

    def __post_init__(self) -> None:
        if self.quantity <= 0:
            raise ValueError(f"quantity ({self.quantity}) must be positive")
        if self.fill_price < 0:
            raise ValueError(f"fill_price ({self.fill_price}) is negative")
        if self.commission < 0:
            raise ValueError(f"commission ({self.commission}) is negative")


# Union used for type-hinting anything that flows through the EventQueue.
Event = MarketEvent | SignalEvent | OrderEvent | FillEvent
