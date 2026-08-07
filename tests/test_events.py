"""Tests for engine/events.py."""

from dataclasses import FrozenInstanceError
from datetime import datetime

import pytest

from engine.events import EventType, FillEvent, MarketEvent, OrderEvent, SignalEvent

TS = datetime(2020, 1, 2, 9, 30)


def make_market_event(**overrides) -> MarketEvent:
    fields = dict(
        timestamp=TS, symbol="AAPL", open=100.0, high=101.0, low=99.0,
        close=100.5, volume=1_000,
    )
    fields.update(overrides)
    return MarketEvent(**fields)


class TestMarketEvent:
    def test_fields_stored(self):
        event = make_market_event()
        assert event.timestamp == TS
        assert event.symbol == "AAPL"
        assert event.open == 100.0
        assert event.high == 101.0
        assert event.low == 99.0
        assert event.close == 100.5
        assert event.volume == 1_000
        assert event.event_type is EventType.MARKET

    def test_frozen(self):
        event = make_market_event()
        with pytest.raises(FrozenInstanceError):
            event.close = 200.0  # type: ignore[misc]

    def test_high_below_low_rejected(self):
        with pytest.raises(ValueError):
            make_market_event(high=98.0, low=99.0)

    def test_open_outside_range_rejected(self):
        with pytest.raises(ValueError):
            make_market_event(open=105.0)

    def test_close_outside_range_rejected(self):
        with pytest.raises(ValueError):
            make_market_event(close=105.0)

    def test_negative_volume_rejected(self):
        with pytest.raises(ValueError):
            make_market_event(volume=-1)


class TestSignalEvent:
    def test_fields_stored(self):
        event = SignalEvent(
            timestamp=TS, symbol="AAPL", direction="LONG",
            strategy_id="sma_crossover", strength=0.8,
        )
        assert event.direction == "LONG"
        assert event.strategy_id == "sma_crossover"
        assert event.strength == 0.8
        assert event.event_type is EventType.SIGNAL

    def test_strength_defaults_to_one(self):
        event = SignalEvent(
            timestamp=TS, symbol="AAPL", direction="EXIT", strategy_id="s1",
        )
        assert event.strength == 1.0

    def test_frozen(self):
        event = SignalEvent(
            timestamp=TS, symbol="AAPL", direction="LONG", strategy_id="s1",
        )
        with pytest.raises(FrozenInstanceError):
            event.direction = "SHORT"  # type: ignore[misc]


class TestOrderEvent:
    def test_fields_stored(self):
        event = OrderEvent(
            timestamp=TS, symbol="AAPL", order_type="MARKET", quantity=10,
            direction="BUY",
        )
        assert event.order_type == "MARKET"
        assert event.quantity == 10
        assert event.direction == "BUY"
        assert event.limit_price is None
        assert event.event_type is EventType.ORDER

    def test_nonpositive_quantity_rejected(self):
        with pytest.raises(ValueError):
            OrderEvent(
                timestamp=TS, symbol="AAPL", order_type="MARKET", quantity=0,
                direction="BUY",
            )

    def test_limit_order_requires_limit_price(self):
        with pytest.raises(ValueError):
            OrderEvent(
                timestamp=TS, symbol="AAPL", order_type="LIMIT", quantity=10,
                direction="BUY",
            )

    def test_limit_order_with_price_ok(self):
        event = OrderEvent(
            timestamp=TS, symbol="AAPL", order_type="LIMIT", quantity=10,
            direction="BUY", limit_price=99.5,
        )
        assert event.limit_price == 99.5


class TestFillEvent:
    def test_fields_stored(self):
        event = FillEvent(
            timestamp=TS, symbol="AAPL", quantity=10, direction="BUY",
            fill_price=100.1, commission=1.0, slippage=0.05,
        )
        assert event.quantity == 10
        assert event.fill_price == 100.1
        assert event.commission == 1.0
        assert event.slippage == 0.05
        assert event.event_type is EventType.FILL

    def test_slippage_defaults_to_zero(self):
        event = FillEvent(
            timestamp=TS, symbol="AAPL", quantity=10, direction="BUY",
            fill_price=100.1, commission=1.0,
        )
        assert event.slippage == 0.0

    def test_nonpositive_quantity_rejected(self):
        with pytest.raises(ValueError):
            FillEvent(
                timestamp=TS, symbol="AAPL", quantity=0, direction="BUY",
                fill_price=100.1, commission=1.0,
            )

    def test_negative_fill_price_rejected(self):
        with pytest.raises(ValueError):
            FillEvent(
                timestamp=TS, symbol="AAPL", quantity=10, direction="BUY",
                fill_price=-1.0, commission=1.0,
            )

    def test_negative_commission_rejected(self):
        with pytest.raises(ValueError):
            FillEvent(
                timestamp=TS, symbol="AAPL", quantity=10, direction="BUY",
                fill_price=100.1, commission=-1.0,
            )
