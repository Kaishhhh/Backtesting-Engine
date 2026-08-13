"""The reusable per-bar event-drain loop that wires Strategy, Portfolio, and
ExecutionHandler together through an EventQueue.

Pulled out of main.py (rather than living only in the script) so an
integration test can drive it directly with synthetic bars.

Per bar: push the MarketEvent, then fully drain the queue -- dispatching by
event_type -- before the next bar is pushed. This mirrors the main-loop
pattern established by engine/event_queue.py's own tests.

Ordering inside the MARKET handler is load-bearing, not incidental: fills
for orders placed on a *prior* bar are resolved and applied to the Portfolio
(execution.on_market_event -> queue.put -> drained as FILL) before this
bar's Strategy call runs. Because EventQueue is strict FIFO, putting the
fill onto the queue first guarantees it dequeues -- and updates
Portfolio.positions -- before any SignalEvent this bar's strategy call
produces reaches Portfolio.generate_order()'s "already holding?" check.
Reversing that order would let generate_order see stale, pre-fill state.
"""

from __future__ import annotations

from collections.abc import Iterable

from engine.event_queue import EventQueue
from engine.events import EventType, MarketEvent
from engine.execution import ExecutionHandler
from engine.portfolio import Portfolio
from strategies.base import Strategy


def run_backtest(
    bars: Iterable[MarketEvent],
    strategy: Strategy,
    portfolio: Portfolio,
    execution: ExecutionHandler,
    queue: EventQueue | None = None,
) -> None:
    """Feed `bars` through the full event pipeline, in order.

    Calls portfolio.record_snapshot() once per bar, after that bar's fills,
    price update, and any resulting signal/order have all been processed --
    giving one equity-curve point per bar (daily cadence, for daily OHLCV).
    """
    queue = queue if queue is not None else EventQueue()

    for bar in bars:
        queue.put(bar)

        while not queue.empty():
            event = queue.get()

            if event.event_type is EventType.MARKET:
                for fill in execution.on_market_event(event):
                    queue.put(fill)
                portfolio.update_market_price(event)
                signal = strategy.on_market_event(event)
                if signal is not None:
                    queue.put(signal)

            elif event.event_type is EventType.SIGNAL:
                order = portfolio.generate_order(event)
                if order is not None:
                    queue.put(order)

            elif event.event_type is EventType.ORDER:
                execution.submit_order(event)

            elif event.event_type is EventType.FILL:
                portfolio.on_fill(event)

        portfolio.record_snapshot(bar.timestamp)
