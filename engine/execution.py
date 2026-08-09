"""Fill simulation: turns pending OrderEvents into FillEvents.

Core rule -- fills happen on the NEXT bar's open, never the bar that produced
the order (see CLAUDE.md's no-look-ahead invariant): a SignalEvent derived
from bar T's data cascades into an OrderEvent timestamped T in the same
event-queue pass. If that order filled against bar T's own close, the
backtest would be trading on a price it "already knew" when the decision was
made -- a same-bar fill is look-ahead bias wearing a trenchcoat. Instead,
ExecutionHandler holds every order pending until it observes a MarketEvent
for that symbol strictly *after* the order's timestamp, and fills against
that bar's open. This mirrors EventQueue's own philosophy: the invariant is a
runtime-checked guard (see the timestamp check in on_market_event), not just
a convention callers are trusted to respect.

Order types:
- MARKET: always fills on the next bar, at that bar's open plus slippage.
- LIMIT: fills only if the next bar's [low, high] range would have crossed
  the limit price; otherwise the order stays pending for a future bar. No
  slippage is applied to LIMIT fills -- a limit order's whole point is a
  price guarantee, and slippage would violate it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from engine.events import FillEvent, MarketEvent, OrderEvent


class SlippageModel(Protocol):
    def apply(self, price: float, direction: Literal["BUY", "SELL"]) -> tuple[float, float]:
        """Return (slipped_price, slippage_amount_per_share)."""
        ...


class CommissionModel(Protocol):
    def compute(self, quantity: int, fill_price: float) -> float:
        """Return total commission owed for a fill of this size/price."""
        ...


@dataclass(frozen=True)
class FixedBpsSlippage:
    """Slippage as a fixed number of basis points of price, working against
    the trader: BUYs fill higher, SELLs fill lower."""

    bps: float = 5.0

    def apply(self, price: float, direction: Literal["BUY", "SELL"]) -> tuple[float, float]:
        adjustment = price * (self.bps / 10_000)
        if direction == "BUY":
            return price + adjustment, adjustment
        return price - adjustment, adjustment


@dataclass(frozen=True)
class PerTradeCommission:
    """Fixed commission per trade plus an optional per-share component."""

    fixed: float = 1.0
    per_share: float = 0.0

    def compute(self, quantity: int, fill_price: float) -> float:
        return self.fixed + self.per_share * quantity


class ExecutionHandler:
    """Holds pending orders per symbol and fills them against later bars."""

    def __init__(
        self,
        slippage_model: SlippageModel | None = None,
        commission_model: CommissionModel | None = None,
    ) -> None:
        self._slippage_model = slippage_model or FixedBpsSlippage()
        self._commission_model = commission_model or PerTradeCommission()
        self._pending_orders: dict[str, list[OrderEvent]] = {}

    def submit_order(self, order: OrderEvent) -> None:
        self._pending_orders.setdefault(order.symbol, []).append(order)

    def pending_orders(self, symbol: str) -> list[OrderEvent]:
        return list(self._pending_orders.get(symbol, []))

    def on_market_event(self, market_event: MarketEvent) -> list[FillEvent]:
        """Attempt to fill every pending order for this bar's symbol.

        MARKET orders always fill here. LIMIT orders fill only if this bar's
        range crosses the limit price; otherwise they remain pending for a
        future bar. Returns the FillEvents produced, if any (order may be
        zero, e.g. an unfilled LIMIT order).
        """
        orders = self._pending_orders.get(market_event.symbol, [])
        if not orders:
            return []

        fills: list[FillEvent] = []
        still_pending: list[OrderEvent] = []
        for order in orders:
            if order.timestamp >= market_event.timestamp:
                # This bar is not strictly after the order -- most likely the
                # very bar that produced the order. Refuse to fill against it.
                still_pending.append(order)
                continue

            fill = self._attempt_fill(order, market_event)
            if fill is not None:
                fills.append(fill)
            else:
                still_pending.append(order)

        self._pending_orders[market_event.symbol] = still_pending
        return fills

    def _attempt_fill(self, order: OrderEvent, bar: MarketEvent) -> FillEvent | None:
        if order.order_type == "MARKET":
            fill_price, slippage_amount = self._slippage_model.apply(bar.open, order.direction)
        else:
            crossed_price = self._limit_cross_price(order, bar)
            if crossed_price is None:
                return None
            fill_price, slippage_amount = crossed_price, 0.0

        commission = self._commission_model.compute(order.quantity, fill_price)
        return FillEvent(
            timestamp=bar.timestamp,
            symbol=order.symbol,
            quantity=order.quantity,
            direction=order.direction,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount,
        )

    @staticmethod
    def _limit_cross_price(order: OrderEvent, bar: MarketEvent) -> float | None:
        assert order.limit_price is not None  # enforced by OrderEvent.__post_init__
        if order.direction == "BUY":
            if bar.low > order.limit_price:
                return None
            # A gap-down open below the limit is more favorable than the
            # limit itself -- the trader pays the better (lower) price.
            return min(bar.open, order.limit_price)
        else:
            if bar.high < order.limit_price:
                return None
            # A gap-up open above the limit is more favorable -- the trader
            # receives the better (higher) price.
            return max(bar.open, order.limit_price)
