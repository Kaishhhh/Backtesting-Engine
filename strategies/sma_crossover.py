"""Dual simple-moving-average crossover strategy.

Deliberately simple: this strategy exists to validate the full event-driven
pipeline end-to-end (MarketEvent -> Signal -> Order -> Fill -> Portfolio),
not to generate alpha. Emits a LONG signal when the fast SMA crosses above
the slow SMA, and an EXIT signal on the reverse crossover.

Rolling-window state, not a stored history: each bar only ever appends to a
fixed-size per-symbol deque (maxlen=slow_window) and recomputes both SMAs
from it. There is no dataframe of "all bars so far" kept anywhere -- the
deque IS the strategy's entire memory, matching the one-bar-at-a-time
contract in strategies/base.py. Multiple symbols are supported (independent
deque + crossover state per symbol) even though this project currently only
ever backtests one symbol at a time.

No signal on the first bar a window becomes full. That bar only establishes
a baseline fast-vs-slow relationship -- no crossover was actually observed,
so emitting a trade there would mean acting on the trend as it stood before
the strategy started watching, which is not something a real point-in-time
strategy could have done. A tie (fast == slow) is treated as "not above" and
does not by itself flip the tracked state.
"""

from __future__ import annotations

from collections import deque

from engine.events import MarketEvent, SignalEvent
from strategies.base import Strategy


class SMACrossoverStrategy(Strategy):
    def __init__(
        self,
        fast_window: int,
        slow_window: int,
        strategy_id: str = "sma_crossover",
    ) -> None:
        if fast_window <= 0:
            raise ValueError(f"fast_window ({fast_window}) must be positive")
        if slow_window <= 0:
            raise ValueError(f"slow_window ({slow_window}) must be positive")
        if fast_window >= slow_window:
            raise ValueError(
                f"fast_window ({fast_window}) must be < slow_window ({slow_window})"
            )
        super().__init__(strategy_id)
        self.fast_window = fast_window
        self.slow_window = slow_window
        self._closes: dict[str, deque[float]] = {}
        self._fast_above_slow: dict[str, bool] = {}

    def on_market_event(self, event: MarketEvent) -> SignalEvent | None:
        closes = self._closes.setdefault(event.symbol, deque(maxlen=self.slow_window))
        closes.append(event.close)
        if len(closes) < self.slow_window:
            return None

        fast_sma = sum(list(closes)[-self.fast_window :]) / self.fast_window
        slow_sma = sum(closes) / self.slow_window
        currently_above = fast_sma > slow_sma

        previously_above = self._fast_above_slow.get(event.symbol)
        self._fast_above_slow[event.symbol] = currently_above

        if previously_above is None or currently_above == previously_above:
            return None

        direction = "LONG" if currently_above else "EXIT"
        return SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            direction=direction,
            strategy_id=self.strategy_id,
        )
