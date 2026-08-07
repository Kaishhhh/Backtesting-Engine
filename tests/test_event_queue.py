"""Tests for engine/event_queue.py.

The queue's core job is to guarantee strict chronological (and, among
same-timestamp events, causal) processing order -- this is the plumbing-level
enforcement of the no-look-ahead invariant described in CLAUDE.md.
"""

from datetime import datetime, timedelta

import pytest

from engine.event_queue import EventQueue, EventQueueError
from engine.events import FillEvent, MarketEvent, OrderEvent, SignalEvent

T0 = datetime(2020, 1, 2, 9, 30)
T1 = T0 + timedelta(minutes=1)
T2 = T0 + timedelta(minutes=2)


def market_event(ts: datetime, symbol: str = "AAPL") -> MarketEvent:
    return MarketEvent(
        timestamp=ts, symbol=symbol, open=100.0, high=101.0, low=99.0,
        close=100.5, volume=1_000,
    )


def signal_event(ts: datetime) -> SignalEvent:
    return SignalEvent(
        timestamp=ts, symbol="AAPL", direction="LONG", strategy_id="test",
    )


def order_event(ts: datetime) -> OrderEvent:
    return OrderEvent(
        timestamp=ts, symbol="AAPL", order_type="MARKET", quantity=10,
        direction="BUY",
    )


def fill_event(ts: datetime) -> FillEvent:
    return FillEvent(
        timestamp=ts, symbol="AAPL", quantity=10, direction="BUY",
        fill_price=100.5, commission=1.0,
    )


class TestBasicFIFOBehavior:
    def test_empty_queue(self):
        queue = EventQueue()
        assert queue.empty()
        assert len(queue) == 0

    def test_put_then_get_returns_same_event(self):
        queue = EventQueue()
        event = market_event(T0)
        queue.put(event)
        assert queue.get() is event

    def test_fifo_order_preserved(self):
        queue = EventQueue()
        events = [market_event(T0), market_event(T1), market_event(T2)]
        for event in events:
            queue.put(event)
        dequeued = [queue.get() for _ in range(3)]
        assert dequeued == events

    def test_len_tracks_queue_size(self):
        queue = EventQueue()
        assert len(queue) == 0
        queue.put(market_event(T0))
        queue.put(market_event(T1))
        assert len(queue) == 2
        queue.get()
        assert len(queue) == 1

    def test_get_on_empty_queue_raises(self):
        queue = EventQueue()
        with pytest.raises(EventQueueError):
            queue.get()


class TestChronologicalOrderEnforcement:
    def test_put_rejects_event_earlier_than_last_dequeued(self):
        queue = EventQueue()
        queue.put(market_event(T1))
        queue.get()  # last_dequeued_timestamp is now T1
        with pytest.raises(EventQueueError):
            queue.put(market_event(T0))  # T0 < T1: would be look-ahead-ish

    def test_put_allows_event_at_same_timestamp_as_last_dequeued(self):
        # Same-timestamp causal cascade (Market -> Signal -> Order -> Fill)
        # must remain legal.
        queue = EventQueue()
        queue.put(market_event(T0))
        queue.get()
        queue.put(signal_event(T0))
        assert queue.get() == signal_event(T0)

    def test_put_allows_event_later_than_last_dequeued(self):
        queue = EventQueue()
        queue.put(market_event(T0))
        queue.get()
        queue.put(market_event(T1))
        assert queue.get() == market_event(T1)

    def test_full_causal_cascade_across_two_bars_stays_in_order(self):
        """Simulates the main loop: drain everything spawned by bar T0's
        MarketEvent (Signal -> Order -> Fill) before bar T1 arrives."""
        queue = EventQueue()
        queue.put(market_event(T0))

        processed = []
        # Bar 1 cascade
        processed.append(queue.get())  # MarketEvent @ T0
        queue.put(signal_event(T0))
        processed.append(queue.get())  # SignalEvent @ T0
        queue.put(order_event(T0))
        processed.append(queue.get())  # OrderEvent @ T0
        queue.put(fill_event(T0))
        processed.append(queue.get())  # FillEvent @ T0

        # Next bar arrives only once the T0 cascade is fully drained.
        queue.put(market_event(T1))
        processed.append(queue.get())  # MarketEvent @ T1

        timestamps = [event.timestamp for event in processed]
        assert timestamps == sorted(timestamps)
        assert [type(event).__name__ for event in processed] == [
            "MarketEvent", "SignalEvent", "OrderEvent", "FillEvent",
            "MarketEvent",
        ]
