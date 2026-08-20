"""RSI mean-reversion strategy.

Conceptually the opposite bet from ``sma_crossover``: instead of trend-
following (a moving-average crossover assumes a trend, once established,
continues), this strategy bets on reversion -- that a price move extreme
enough to push RSI into oversold/overbought territory is likely to snap
back. Emits LONG when RSI crosses down into oversold (< 30) and EXIT when
RSI crosses up into overbought (> 70).

RSI variant: a plain/"Cutler's" RSI -- average gain and average loss over
the lookback period are simple (unweighted) means of the period's
bar-over-bar changes, recomputed fresh from the rolling window on every
bar. Deliberately not Wilder's original smoothed RSI (an exponential
moving average of gains/losses carried forward indefinitely), because that
would mean the indicator's value depends on the entire history since the
strategy started rather than just the current window -- more path-dependent
state to reason about for no benefit here, given this project's stated goal
is proving the engine/Strategy interface generalizes, not tuning an
indicator. Cutler's variant is a standard, well-documented alternative, not
a made-up simplification.

Rolling-window state, not a stored history: mirrors sma_crossover.py's
approach exactly. A fixed-size deque (maxlen=period + 1, since computing
`period` bar-over-bar changes needs period+1 closes) is the strategy's
entire memory; RSI is recomputed from scratch from that window every bar.

No signal until the window is full, and no signal on the bar the window
first becomes full: that bar only establishes a baseline oversold/
overbought/neutral zone, not an observed crossing into one -- same
discipline as sma_crossover.py's baseline-bar handling, for the same
reason (trading on it would mean acting on history from before the
strategy started watching).
"""

from __future__ import annotations

from collections import deque
from typing import Literal

from engine.events import MarketEvent, SignalEvent
from strategies.base import Strategy

Zone = Literal["low", "mid", "high"]


class RSIMeanReversionStrategy(Strategy):
    def __init__(
        self,
        period: int = 14,
        oversold: float = 30.0,
        overbought: float = 70.0,
        strategy_id: str = "rsi_mean_reversion",
    ) -> None:
        if period <= 0:
            raise ValueError(f"period ({period}) must be positive")
        if not (0.0 <= oversold < overbought <= 100.0):
            raise ValueError(
                f"require 0 <= oversold ({oversold}) < overbought ({overbought}) <= 100"
            )
        super().__init__(strategy_id)
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
        self._closes: dict[str, deque[float]] = {}
        self._zone: dict[str, Zone] = {}

    def _rsi(self, closes: deque[float]) -> float:
        gains = 0.0
        losses = 0.0
        prev = closes[0]
        for close in list(closes)[1:]:
            change = close - prev
            if change > 0:
                gains += change
            else:
                losses += -change
            prev = close
        avg_gain = gains / self.period
        avg_loss = losses / self.period
        if avg_loss == 0.0:
            return 100.0
        if avg_gain == 0.0:
            return 0.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    def _classify(self, rsi: float) -> Zone:
        if rsi < self.oversold:
            return "low"
        if rsi > self.overbought:
            return "high"
        return "mid"

    def on_market_event(self, event: MarketEvent) -> SignalEvent | None:
        closes = self._closes.setdefault(event.symbol, deque(maxlen=self.period + 1))
        closes.append(event.close)
        if len(closes) < self.period + 1:
            return None

        current_zone = self._classify(self._rsi(closes))
        previous_zone = self._zone.get(event.symbol)
        self._zone[event.symbol] = current_zone

        if previous_zone is None or current_zone == previous_zone:
            return None

        if current_zone == "low":
            direction = "LONG"
        elif current_zone == "high":
            direction = "EXIT"
        else:
            # Left the zone it was in without reaching the opposite
            # extreme (e.g. low -> mid, or high -> mid): the reversion (or
            # its exit) hasn't actually happened yet.
            return None

        return SignalEvent(
            timestamp=event.timestamp,
            symbol=event.symbol,
            direction=direction,
            strategy_id=self.strategy_id,
        )
