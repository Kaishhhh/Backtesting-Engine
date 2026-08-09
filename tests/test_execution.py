"""Tests for engine/execution.py.

The most important property tested here is the one CLAUDE.md calls out
explicitly: an order must never fill against the same bar that produced it,
only a strictly later one. test_order_does_not_fill_against_its_own_bar
checks that directly, not just indirectly via "the fill price looks right."
"""

from datetime import datetime, timedelta

import pytest

from engine.events import MarketEvent, OrderEvent
from engine.execution import ExecutionHandler, FixedBpsSlippage, PerTradeCommission

T0 = datetime(2024, 1, 2, 9, 30)
T1 = T0 + timedelta(days=1)
T2 = T0 + timedelta(days=2)


def bar(ts: datetime, open_: float, high: float, low: float, close: float, symbol="AAPL") -> MarketEvent:
    return MarketEvent(
        timestamp=ts, symbol=symbol, open=open_, high=high, low=low, close=close, volume=1_000
    )


def market_order(ts: datetime, direction: str, quantity: int = 10, symbol="AAPL") -> OrderEvent:
    return OrderEvent(
        timestamp=ts, symbol=symbol, order_type="MARKET", quantity=quantity, direction=direction
    )


def limit_order(ts: datetime, direction: str, limit_price: float, quantity: int = 10, symbol="AAPL") -> OrderEvent:
    return OrderEvent(
        timestamp=ts, symbol=symbol, order_type="LIMIT", quantity=quantity,
        direction=direction, limit_price=limit_price,
    )


class TestNextBarFillTiming:
    def test_order_does_not_fill_against_its_own_bar(self):
        handler = ExecutionHandler()
        order = market_order(T0, "BUY")
        handler.submit_order(order)

        # The very bar that produced this order (same timestamp) must not
        # fill it -- that would be a same-bar look-ahead fill.
        fills = handler.on_market_event(bar(T0, open_=100, high=101, low=99, close=100.5))

        assert fills == []
        assert order in handler.pending_orders("AAPL")

    def test_order_fills_on_the_next_strictly_later_bar(self):
        handler = ExecutionHandler(slippage_model=FixedBpsSlippage(bps=0))
        order = market_order(T0, "BUY")
        handler.submit_order(order)

        handler.on_market_event(bar(T0, open_=100, high=101, low=99, close=100.5))  # no fill
        fills = handler.on_market_event(bar(T1, open_=105, high=106, low=104, close=105.5))

        assert len(fills) == 1
        assert fills[0].fill_price == 105  # next bar's open, zero slippage
        assert fills[0].timestamp == T1
        assert handler.pending_orders("AAPL") == []

    def test_market_events_for_other_symbols_do_not_trigger_a_fill(self):
        handler = ExecutionHandler()
        order = market_order(T0, "BUY", symbol="AAPL")
        handler.submit_order(order)

        fills = handler.on_market_event(bar(T1, open_=50, high=51, low=49, close=50.5, symbol="MSFT"))

        assert fills == []
        assert order in handler.pending_orders("AAPL")


class TestMarketOrders:
    def test_buy_fills_at_next_open_plus_slippage(self):
        handler = ExecutionHandler(slippage_model=FixedBpsSlippage(bps=10))  # 0.10%
        handler.submit_order(market_order(T0, "BUY", quantity=100))
        handler.on_market_event(bar(T0, 100, 101, 99, 100.5))

        [fill] = handler.on_market_event(bar(T1, open_=200, high=201, low=199, close=200.5))

        assert fill.fill_price == pytest.approx(200.2)  # 200 + 0.10% = 200.2
        assert fill.slippage == pytest.approx(0.2)
        assert fill.direction == "BUY"
        assert fill.quantity == 100

    def test_sell_fills_at_next_open_minus_slippage(self):
        handler = ExecutionHandler(slippage_model=FixedBpsSlippage(bps=10))
        handler.submit_order(market_order(T0, "SELL", quantity=100))
        handler.on_market_event(bar(T0, 100, 101, 99, 100.5))

        [fill] = handler.on_market_event(bar(T1, open_=200, high=201, low=199, close=200.5))

        assert fill.fill_price == pytest.approx(199.8)  # 200 - 0.10%
        assert fill.slippage == pytest.approx(0.2)
        assert fill.direction == "SELL"


class TestCommission:
    def test_fixed_plus_per_share_commission(self):
        handler = ExecutionHandler(
            slippage_model=FixedBpsSlippage(bps=0),
            commission_model=PerTradeCommission(fixed=1.0, per_share=0.005),
        )
        handler.submit_order(market_order(T0, "BUY", quantity=200))
        handler.on_market_event(bar(T0, 100, 101, 99, 100.5))

        [fill] = handler.on_market_event(bar(T1, 150, 151, 149, 150.5))

        assert fill.commission == pytest.approx(1.0 + 0.005 * 200)  # 2.0

    def test_limit_fill_has_no_slippage(self):
        handler = ExecutionHandler(slippage_model=FixedBpsSlippage(bps=50))
        handler.submit_order(limit_order(T0, "BUY", limit_price=100))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        [fill] = handler.on_market_event(bar(T1, open_=102, high=103, low=99, close=100.5))

        assert fill.slippage == 0.0


class TestLimitOrders:
    def test_buy_limit_does_not_fill_when_range_never_reaches_limit(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "BUY", limit_price=90))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        fills = handler.on_market_event(bar(T1, open_=100, high=101, low=95, close=100.5))

        assert fills == []
        assert len(handler.pending_orders("AAPL")) == 1

    def test_buy_limit_fills_when_low_touches_limit(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "BUY", limit_price=98))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        [fill] = handler.on_market_event(bar(T1, open_=100, high=101, low=97, close=99))

        # Open (100) is worse than the limit (98) for a buyer, so fill at
        # the limit price -- the trader's protected price.
        assert fill.fill_price == 98

    def test_buy_limit_gap_down_open_fills_at_more_favorable_open(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "BUY", limit_price=98))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        [fill] = handler.on_market_event(bar(T1, open_=95, high=96, low=94, close=95.5))

        # Market gapped below the limit -- buyer gets the better (lower) open.
        assert fill.fill_price == 95

    def test_sell_limit_does_not_fill_when_range_never_reaches_limit(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "SELL", limit_price=120))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        fills = handler.on_market_event(bar(T1, open_=110, high=115, low=108, close=110.5))

        assert fills == []

    def test_sell_limit_fills_when_high_touches_limit(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "SELL", limit_price=120))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        [fill] = handler.on_market_event(bar(T1, open_=115, high=121, low=114, close=119))

        assert fill.fill_price == 120

    def test_sell_limit_gap_up_open_fills_at_more_favorable_open(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "SELL", limit_price=120))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        [fill] = handler.on_market_event(bar(T1, open_=125, high=126, low=124, close=125.5))

        assert fill.fill_price == 125

    def test_unfilled_limit_order_stays_pending_across_multiple_bars(self):
        handler = ExecutionHandler()
        handler.submit_order(limit_order(T0, "BUY", limit_price=50))
        handler.on_market_event(bar(T0, 105, 106, 104, 105.5))

        handler.on_market_event(bar(T1, open_=100, high=101, low=95, close=100.5))
        fills = handler.on_market_event(bar(T2, open_=98, high=99, low=90, close=95))

        assert fills == []
        assert len(handler.pending_orders("AAPL")) == 1
