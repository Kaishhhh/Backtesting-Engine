"""Portfolio: cash accounting, position tracking, PnL, and the equity curve.

Scope, deliberately: long-only, cash account, no margin/leverage, no
shorting. A BUY fill that would take cash below zero, or a SELL fill that
would take a position below zero, is rejected outright (raises
PortfolioError) rather than silently allowed -- this project models
leverage or short positions only if that scope is deliberately expanded
later (see CLAUDE.md's Design Decisions).

Cost-basis convention: a position's avg_cost is the volume-weighted average
*fill price* only -- commission is not folded into it. Commission still
reduces cash immediately on every fill (buy or sell), and is subtracted
again from realized PnL at the moment a position is sold. This keeps
avg_cost a pure "what did I pay for the shares" number while commission's
drag on performance still shows up in both cash and realized PnL.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from engine.events import FillEvent, MarketEvent

_CASH_EPSILON = 1e-9  # tolerance for floating-point noise, not a leverage allowance


class PortfolioError(RuntimeError):
    """Raised when applying a fill would violate the long-only, no-margin
    invariant, or when portfolio state is queried before it can be answered."""


@dataclass
class Position:
    symbol: str
    quantity: int = 0
    avg_cost: float = 0.0
    realized_pnl: float = 0.0


@dataclass(frozen=True)
class PortfolioSnapshot:
    timestamp: datetime
    cash: float
    positions_value: float

    @property
    def total_value(self) -> float:
        return self.cash + self.positions_value


class Portfolio:
    def __init__(self, initial_cash: float) -> None:
        if initial_cash < 0:
            raise ValueError(f"initial_cash ({initial_cash}) cannot be negative")
        self.cash = initial_cash
        self.positions: dict[str, Position] = {}
        self.trade_log: list[FillEvent] = []
        self.equity_curve: list[PortfolioSnapshot] = []
        self._latest_prices: dict[str, float] = {}

    def update_market_price(self, market_event: MarketEvent) -> None:
        """Record the latest known price for a symbol, used for
        mark-to-market valuation. Uses the bar's close."""
        self._latest_prices[market_event.symbol] = market_event.close

    def on_fill(self, fill: FillEvent) -> None:
        if fill.direction == "BUY":
            self._apply_buy(fill)
        else:
            self._apply_sell(fill)
        self.trade_log.append(fill)

    def _apply_buy(self, fill: FillEvent) -> None:
        cost = fill.quantity * fill.fill_price + fill.commission
        if cost > self.cash + _CASH_EPSILON:
            raise PortfolioError(
                f"BUY fill for {fill.quantity} {fill.symbol} @ {fill.fill_price} "
                f"(+ {fill.commission:.2f} commission) costs {cost:.2f}, exceeding "
                f"available cash {self.cash:.2f}. No margin/leverage is modeled."
            )
        position = self.positions.setdefault(fill.symbol, Position(fill.symbol))
        new_quantity = position.quantity + fill.quantity
        position.avg_cost = (
            position.quantity * position.avg_cost + fill.quantity * fill.fill_price
        ) / new_quantity
        position.quantity = new_quantity
        self.cash -= cost

    def _apply_sell(self, fill: FillEvent) -> None:
        position = self.positions.get(fill.symbol)
        held = position.quantity if position is not None else 0
        if fill.quantity > held:
            raise PortfolioError(
                f"SELL fill for {fill.quantity} {fill.symbol} exceeds the "
                f"{held} share(s) held. Shorting is not supported (long-only)."
            )
        proceeds = fill.quantity * fill.fill_price - fill.commission
        realized = (fill.fill_price - position.avg_cost) * fill.quantity - fill.commission
        position.realized_pnl += realized
        position.quantity -= fill.quantity
        if position.quantity == 0:
            position.avg_cost = 0.0
        self.cash += proceeds

    def _price_for(self, symbol: str) -> float:
        price = self._latest_prices.get(symbol)
        if price is None:
            raise PortfolioError(
                f"no market price known for {symbol}; call update_market_price() "
                f"before requesting mark-to-market value."
            )
        return price

    def market_value(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        if position is None or position.quantity == 0:
            return 0.0
        return position.quantity * self._price_for(symbol)

    def unrealized_pnl(self, symbol: str) -> float:
        position = self.positions.get(symbol)
        if position is None or position.quantity == 0:
            return 0.0
        return (self._price_for(symbol) - position.avg_cost) * position.quantity

    def total_positions_value(self) -> float:
        return sum(self.market_value(symbol) for symbol in self.positions)

    def total_unrealized_pnl(self) -> float:
        return sum(self.unrealized_pnl(symbol) for symbol in self.positions)

    def total_realized_pnl(self) -> float:
        return sum(position.realized_pnl for position in self.positions.values())

    def snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Mark-to-market portfolio value at `timestamp`, without recording
        it to the equity curve."""
        return PortfolioSnapshot(
            timestamp=timestamp,
            cash=self.cash,
            positions_value=self.total_positions_value(),
        )

    def record_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """Snapshot and append it to the equity curve. The caller decides
        cadence (e.g. once per bar, after that bar's fills and price updates
        have both been applied) -- Portfolio stays agnostic about how many
        symbols or bars make up a single point in time."""
        snap = self.snapshot(timestamp)
        self.equity_curve.append(snap)
        return snap
