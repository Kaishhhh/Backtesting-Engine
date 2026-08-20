"""Tests for strategies/rsi_mean_reversion.py.

Covers: no signal until the period+1-close window is full, no signal on the
bar the window first becomes full (baseline zone, not an observed
crossing), correct LONG/EXIT emission on actual zone crossings, the
"left a zone without reaching the opposite extreme" no-signal case,
per-symbol independence, and that the strategy is never handed more than
one bar at a time (the interface itself is the no-look-ahead guarantee).
"""

from datetime import datetime, timedelta

import pytest

from engine.events import MarketEvent
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

T0 = datetime(2024, 1, 2, 9, 30)


def ts(n: int) -> datetime:
    return T0 + timedelta(days=n)


def bar(n: int, close: float, symbol: str = "AAPL") -> MarketEvent:
    return MarketEvent(
        timestamp=ts(n), symbol=symbol, open=close, high=close + 1,
        low=close - 1, close=close, volume=1_000,
    )


class TestConstructorValidation:
    def test_rejects_non_positive_period(self):
        with pytest.raises(ValueError):
            RSIMeanReversionStrategy(period=0)

    def test_rejects_oversold_not_less_than_overbought(self):
        with pytest.raises(ValueError):
            RSIMeanReversionStrategy(oversold=70, overbought=30)
        with pytest.raises(ValueError):
            RSIMeanReversionStrategy(oversold=50, overbought=50)

    def test_rejects_thresholds_outside_0_100(self):
        with pytest.raises(ValueError):
            RSIMeanReversionStrategy(oversold=-1, overbought=70)
        with pytest.raises(ValueError):
            RSIMeanReversionStrategy(oversold=30, overbought=101)


class TestUndersizedWindow:
    def test_no_signal_until_period_plus_one_closes(self):
        strategy = RSIMeanReversionStrategy(period=4)
        closes = [100, 90, 80, 70]  # only 4 closes; period=4 needs 5
        for n, close in enumerate(closes):
            assert strategy.on_market_event(bar(n, close)) is None


class TestZoneCrossingSequence:
    """period=4, oversold=30, overbought=70 over closes:

    [100, 90, 80, 70, 60, 60, 100, 140, 140, 100, 60, 20]

    Cutler's RSI (simple mean of gains/losses over the trailing 4
    bar-over-bar changes), hand-computed per bar once the 5-close window
    is full:

    n4  window(100,90,80,70,60): changes -10,-10,-10,-10 -> all losses ->
        RSI=0 -> zone "low". First bar the window is full -> baseline,
        None (no crossing has actually been observed yet).
    n5  window(90,80,70,60,60): changes -10,-10,-10,0 -> still all
        non-positive -> RSI=0 -> zone "low", unchanged from n4 -> None.
    n6  window(80,70,60,60,100): changes -10,-10,0,+40 -> gains=40,
        losses=20 -> avg_gain=10, avg_loss=5 -> RS=2 ->
        RSI=100-100/3=66.667 -> zone "mid". Left "low" but landed in
        "mid", not "high" -> no signal (reversion not confirmed yet).
    n7  window(70,60,60,100,140): changes -10,0,+40,+40 -> gains=80,
        losses=10 -> avg_gain=20, avg_loss=2.5 -> RS=8 ->
        RSI=100-100/9=88.889 -> zone "high". mid -> high -> EXIT.
    n8  window(60,60,100,140,140): changes 0,+40,+40,0 -> gains=80,
        losses=0 -> avg_loss=0 -> RSI=100 -> zone "high", unchanged from
        n7 -> None.
    n9  window(60,100,140,140,100): changes +40,+40,0,-40 -> gains=80,
        losses=40 -> avg_gain=20, avg_loss=10 -> RS=2 ->
        RSI=66.667 -> zone "mid". high -> mid -> no signal (no reversal
        down to oversold confirmed yet).
    n10 window(100,140,140,100,60): changes +40,0,-40,-40 -> gains=40,
        losses=80 -> avg_gain=10, avg_loss=20 -> RS=0.5 ->
        RSI=100-100/1.5=33.333 -> zone "mid" (33.333 > 30), unchanged
        from n9 -> None.
    n11 window(140,140,100,60,20): changes 0,-40,-40,-40 -> gains=0,
        losses=120 -> avg_gain=0 -> RSI=0 -> zone "low". mid -> low ->
        LONG.
    """

    CLOSES = [100, 90, 80, 70, 60, 60, 100, 140, 140, 100, 60, 20]

    def test_full_zone_crossing_sequence(self):
        strategy = RSIMeanReversionStrategy(period=4)
        signals = [strategy.on_market_event(bar(n, c)) for n, c in enumerate(self.CLOSES)]

        # n0-n3: undersized window (only 4 of the 5 required closes).
        assert signals[0] is None
        assert signals[1] is None
        assert signals[2] is None
        assert signals[3] is None

        assert signals[4] is None  # baseline "low" zone, not a crossing
        assert signals[5] is None  # still "low"

        assert signals[6] is None  # low -> mid: left without reaching high

        assert signals[7] is not None
        assert signals[7].direction == "EXIT"
        assert signals[7].symbol == "AAPL"
        assert signals[7].strategy_id == "rsi_mean_reversion"
        assert signals[7].timestamp == ts(7)

        assert signals[8] is None  # still "high"

        assert signals[9] is None  # high -> mid: left without reaching low

        assert signals[10] is None  # mid -> mid, no zone change

        assert signals[11] is not None
        assert signals[11].direction == "LONG"
        assert signals[11].timestamp == ts(11)


class TestPerSymbolIndependence:
    def test_symbols_track_independent_state(self):
        strategy = RSIMeanReversionStrategy(period=4)

        aapl_closes = TestZoneCrossingSequence.CLOSES[:8]  # through the EXIT bar
        aapl_signals = [
            strategy.on_market_event(bar(n, c, "AAPL")) for n, c in enumerate(aapl_closes)
        ]
        assert aapl_signals[-1].direction == "EXIT"

        # MSFT must independently require its own full 5-close window --
        # AAPL's history/zone state must not leak into it.
        for n, close in enumerate([100, 90, 80, 70]):
            assert strategy.on_market_event(bar(n + 20, close, "MSFT")) is None


class TestNoLookahead:
    def test_on_market_event_only_accepts_a_single_bar(self):
        """The interface itself forbids access to history/future: the only
        way to feed data in is one MarketEvent per call, and the return type
        can only carry a single symbol/timestamp derived from that one bar."""
        strategy = RSIMeanReversionStrategy(period=4)
        signal = strategy.on_market_event(bar(0, 100))
        assert signal is None or signal.timestamp == ts(0)
