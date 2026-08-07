"""FIFO event queue that drives the main backtest loop.

Design note (why FIFO, not a timestamp-sorted heap):
Events are not independent -- a MarketEvent at time t causally produces a
SignalEvent at time t, which produces an OrderEvent at time t (or t, depending
on the execution model), which produces a FillEvent. In the classic
event-driven backtest design, the main loop pulls one MarketEvent at a time
(in chronological order, from the data handler) and fully drains every event
that cascades from it -- Signal -> Order -> Fill -- before pulling the next
MarketEvent. Because insertion order already matches both chronological order
*and* causal order among same-timestamp events, plain FIFO is correct and
simpler than a priority queue, which would need a tiebreaker to avoid
reordering causally dependent same-timestamp events.

The queue still enforces the invariant rather than merely assuming callers get
it right: put() rejects any event timestamped earlier than the most recently
dequeued event. This makes an accidental attempt to feed the strategy a
stale/future-relative-to-processing event fail loudly, in keeping with the
no-look-ahead invariant described in CLAUDE.md.
"""

from __future__ import annotations

from collections import deque
from datetime import datetime

from engine.events import Event


class EventQueueError(RuntimeError):
    """Raised when the EventQueue detects an ordering violation or misuse."""


class EventQueue:
    """A FIFO queue of Events, processed in strict chronological order."""

    def __init__(self) -> None:
        self._queue: deque[Event] = deque()
        self._last_dequeued_timestamp: datetime | None = None

    def put(self, event: Event) -> None:
        """Enqueue an event.

        Raises EventQueueError if the event's timestamp is earlier than the
        most recently dequeued event's timestamp -- such an event could only
        be processed by "going back in time" relative to what has already
        been handled, which would violate no-look-ahead.
        """
        if (
            self._last_dequeued_timestamp is not None
            and event.timestamp < self._last_dequeued_timestamp
        ):
            raise EventQueueError(
                f"Refusing to enqueue {type(event).__name__} timestamped "
                f"{event.timestamp!r}, which is earlier than the most "
                f"recently processed event at "
                f"{self._last_dequeued_timestamp!r}."
            )
        self._queue.append(event)

    def get(self) -> Event:
        """Dequeue and return the next event in FIFO order.

        Raises EventQueueError if the queue is empty.
        """
        if not self._queue:
            raise EventQueueError("get() called on an empty EventQueue.")
        event = self._queue.popleft()
        self._last_dequeued_timestamp = event.timestamp
        return event

    def empty(self) -> bool:
        return len(self._queue) == 0

    def __len__(self) -> int:
        return len(self._queue)
