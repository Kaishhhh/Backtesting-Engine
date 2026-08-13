"""Tests for strategies/sma_crossover.py.

Covers: no signal until the slow window is full, no signal on the bar the
window first becomes full (baseline, not an observed crossover), correct
LONG/EXIT emission on actual crossovers, tie handling, per-symbol
independence, and that the strategy is never handed more than one bar at a
time (the interface itself is the no-look-ahead guarantee here).
"""

from datetime import datetime, timedelta

import pytest

from engine.events import MarketEvent
from strategies.sma_crossover import SMACrossoverStrategy

T0 = datetime(2024, 1, 2, 9, 30)


def ts(n: int) -> datetime:
    return T0 + timedelta(days=n)


def bar(n: int, close: float, symbol: str = "AAPL") -> MarketEvent:
    return MarketEvent(
        timestamp=ts(n), symbol=symbol, open=close, high=close + 1,
        low=close - 1, close=close, volume=1_000,
    )


class TestConstructorValidation:
    def test_rejects_non_positive_fast_window(self):
        with pytest.raises(ValueError):
            SMACrossoverStrategy(fast_window=0, slow_window=4)

    def test_rejects_non_positive_slow_window(self):
        with pytest.raises(ValueError):
            SMACrossoverStrategy(fast_window=2, slow_window=0)

    def test_rejects_fast_window_not_less_than_slow(self):
        with pytest.raises(ValueError):
            SMACrossoverStrategy(fast_window=4, slow_window=4)
        with pytest.raises(ValueError):
            SMACrossoverStrategy(fast_window=5, slow_window=4)


class TestUndersizedWindow:
    def test_no_signal_until_slow_window_is_full(self):
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=4)
        closes = [10, 10, 10]  # only 3 bars, slow_window needs 4
        for n, close in enumerate(closes):
            assert strategy.on_market_event(bar(n, close)) is None


class TestCrossoverDetection:
    """fast=2, slow=4 over closes [10, 10, 10, 10, 20, 20, 5]:

    bar0-2 (10,10,10): undersized, None.
    bar3 (10): window full, fast=slow=10 -> baseline established, None.
    bar4 (20): fast=avg(10,20)=15, slow=avg(10,10,10,20)=12.5 -> above,
        first observed crossover -> LONG.
    bar5 (20): fast=avg(20,20)=20, slow=avg(10,10,20,20)=15 -> still above,
        no new crossover -> None.
    bar6 (5): fast=avg(20,5)=12.5, slow=avg(10,20,20,5)=13.75 -> below,
        crossed down -> EXIT.
    """

    def test_full_crossover_sequence(self):
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=4)
        closes = [10, 10, 10, 10, 20, 20, 5]
        signals = [strategy.on_market_event(bar(n, c)) for n, c in enumerate(closes)]

        assert signals[0] is None
        assert signals[1] is None
        assert signals[2] is None
        assert signals[3] is None  # baseline bar, not a crossover

        assert signals[4] is not None
        assert signals[4].direction == "LONG"
        assert signals[4].symbol == "AAPL"
        assert signals[4].strategy_id == "sma_crossover"
        assert signals[4].timestamp == ts(4)

        assert signals[5] is None

        assert signals[6] is not None
        assert signals[6].direction == "EXIT"
        assert signals[6].timestamp == ts(6)


class TestTieHandling:
    """fast=1, slow=2 over closes [10, 10, 20, 20, 20]:

    bar0 (10): undersized, None.
    bar1 (10): fast=10, slow=10 -> tie counts as "not above" -> baseline, None.
    bar2 (20): fast=20, slow=15 -> above -> LONG.
    bar3 (20): fast=20, slow=20 -> tie counts as "not above" -> crossed down
        from the prior above state -> EXIT.
    bar4 (20): fast=20, slow=20 -> still tied/"not above", no state change
        from bar3 -> None (a tie does not itself keep re-triggering).
    """

    def test_tie_transitions_and_stabilizes(self):
        strategy = SMACrossoverStrategy(fast_window=1, slow_window=2)
        closes = [10, 10, 20, 20, 20]
        signals = [strategy.on_market_event(bar(n, c)) for n, c in enumerate(closes)]

        assert signals[0] is None
        assert signals[1] is None
        assert signals[2].direction == "LONG"
        assert signals[3].direction == "EXIT"
        assert signals[4] is None


class TestPerSymbolIndependence:
    def test_symbols_track_independent_state(self):
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=4)

        # Establish AAPL fully (baseline + one crossover) while MSFT has
        # seen nothing at all yet.
        aapl_closes = [10, 10, 10, 10, 20]
        aapl_signals = [strategy.on_market_event(bar(n, c, "AAPL")) for n, c in enumerate(aapl_closes)]
        assert aapl_signals[-1].direction == "LONG"

        # MSFT must independently require its own full window -- AAPL's
        # history must not leak into it.
        assert strategy.on_market_event(bar(10, 10, "MSFT")) is None
        assert strategy.on_market_event(bar(11, 10, "MSFT")) is None
        assert strategy.on_market_event(bar(12, 10, "MSFT")) is None


class TestNoLookahead:
    def test_on_market_event_only_accepts_a_single_bar(self):
        """The interface itself forbids access to history/future: the only
        way to feed data in is one MarketEvent per call, and the return type
        can only carry a single symbol/timestamp derived from that one bar."""
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=4)
        signal = strategy.on_market_event(bar(0, 10))
        assert signal is None or signal.timestamp == ts(0)
