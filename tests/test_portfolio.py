"""Tests for engine/portfolio.py: cash accounting, position averaging,
realized/unrealized PnL, and equity curve accumulation over a fill sequence.
"""

from datetime import datetime, timedelta

import pytest

from engine.events import FillEvent, MarketEvent, SignalEvent
from engine.portfolio import Portfolio, PortfolioError

T0 = datetime(2024, 1, 2, 9, 30)
T1 = T0 + timedelta(days=1)
T2 = T0 + timedelta(days=2)


def fill(ts, direction, quantity, price, commission=1.0, symbol="AAPL") -> FillEvent:
    return FillEvent(
        timestamp=ts, symbol=symbol, quantity=quantity, direction=direction,
        fill_price=price, commission=commission,
    )


def bar(ts, close, symbol="AAPL", open_=None) -> MarketEvent:
    o = open_ if open_ is not None else close
    return MarketEvent(
        timestamp=ts, symbol=symbol, open=o, high=max(o, close) + 1,
        low=min(o, close) - 1, close=close, volume=1_000,
    )


def signal(ts, direction, symbol="AAPL", strength=1.0, strategy_id="test") -> SignalEvent:
    return SignalEvent(
        timestamp=ts, symbol=symbol, direction=direction,
        strategy_id=strategy_id, strength=strength,
    )


class TestInitialization:
    def test_negative_initial_cash_rejected(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=-100)

    def test_starts_with_no_positions(self):
        portfolio = Portfolio(initial_cash=10_000)
        assert portfolio.positions == {}
        assert portfolio.cash == 10_000


class TestBuyFills:
    def test_buy_deducts_cash_and_creates_position(self):
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=1.0))

        assert portfolio.cash == pytest.approx(10_000 - 1_000 - 1.0)
        assert portfolio.positions["AAPL"].quantity == 10
        assert portfolio.positions["AAPL"].avg_cost == pytest.approx(100.0)

    def test_position_averaging_across_multiple_buys(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0))
        portfolio.on_fill(fill(T1, "BUY", quantity=10, price=200.0))

        position = portfolio.positions["AAPL"]
        assert position.quantity == 20
        assert position.avg_cost == pytest.approx(150.0)  # (10*100 + 10*200) / 20

    def test_buy_exceeding_cash_is_rejected_and_state_unchanged(self):
        portfolio = Portfolio(initial_cash=500)
        with pytest.raises(PortfolioError):
            portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=1.0))  # costs 1001

        assert portfolio.cash == 500
        assert portfolio.positions == {}
        assert portfolio.trade_log == []

    def test_buy_exactly_at_available_cash_succeeds(self):
        portfolio = Portfolio(initial_cash=1_001)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=1.0))
        assert portfolio.cash == pytest.approx(0.0)


class TestSellFills:
    def test_sell_updates_cash_and_realized_pnl(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=1.0))
        cash_after_buy = portfolio.cash

        portfolio.on_fill(fill(T1, "SELL", quantity=10, price=120.0, commission=1.0))

        proceeds = 10 * 120.0 - 1.0
        assert portfolio.cash == pytest.approx(cash_after_buy + proceeds)
        expected_realized = (120.0 - 100.0) * 10 - 1.0
        assert portfolio.positions["AAPL"].realized_pnl == pytest.approx(expected_realized)
        assert portfolio.positions["AAPL"].quantity == 0

    def test_sell_more_than_held_is_rejected_long_only(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=5, price=100.0))

        with pytest.raises(PortfolioError):
            portfolio.on_fill(fill(T1, "SELL", quantity=10, price=100.0))

        # Rejected fill must not have mutated state.
        assert portfolio.positions["AAPL"].quantity == 5
        assert len(portfolio.trade_log) == 1

    def test_sell_with_no_position_at_all_is_rejected(self):
        portfolio = Portfolio(initial_cash=100_000)
        with pytest.raises(PortfolioError):
            portfolio.on_fill(fill(T0, "SELL", quantity=1, price=100.0))

    def test_partial_sell_reduces_quantity_but_keeps_avg_cost(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0))
        portfolio.on_fill(fill(T1, "SELL", quantity=4, price=110.0))

        position = portfolio.positions["AAPL"]
        assert position.quantity == 6
        assert position.avg_cost == pytest.approx(100.0)


class TestMarkToMarket:
    def test_unrealized_pnl_reflects_price_movement(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0))
        portfolio.update_market_price(bar(T1, close=115.0))

        assert portfolio.unrealized_pnl("AAPL") == pytest.approx(150.0)  # (115-100)*10
        assert portfolio.market_value("AAPL") == pytest.approx(1_150.0)

    def test_market_value_without_price_update_raises(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0))
        with pytest.raises(PortfolioError):
            portfolio.market_value("AAPL")

    def test_flat_symbol_reports_zero_value_without_needing_a_price(self):
        portfolio = Portfolio(initial_cash=100_000)
        assert portfolio.market_value("AAPL") == 0.0
        assert portfolio.unrealized_pnl("AAPL") == 0.0


class TestSnapshotAndEquityCurve:
    def test_snapshot_total_value_is_cash_plus_positions(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=1.0))
        portfolio.update_market_price(bar(T0, close=100.0))

        snap = portfolio.snapshot(T0)

        assert snap.cash == pytest.approx(100_000 - 1_000 - 1.0)
        assert snap.positions_value == pytest.approx(1_000.0)
        assert snap.total_value == pytest.approx(100_000 - 1.0)

    def test_snapshot_does_not_mutate_equity_curve(self):
        portfolio = Portfolio(initial_cash=100_000)
        portfolio.snapshot(T0)
        assert portfolio.equity_curve == []

    def test_equity_curve_accumulates_over_a_multi_day_sequence(self):
        portfolio = Portfolio(initial_cash=100_000)

        # Day 0: buy 10 shares at 100.
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=1.0))
        portfolio.update_market_price(bar(T0, close=100.0))
        portfolio.record_snapshot(T0)

        # Day 1: price rises to 110, no trading.
        portfolio.update_market_price(bar(T1, close=110.0))
        portfolio.record_snapshot(T1)

        # Day 2: sell everything at 120.
        portfolio.on_fill(fill(T2, "SELL", quantity=10, price=120.0, commission=1.0))
        portfolio.update_market_price(bar(T2, close=120.0))
        portfolio.record_snapshot(T2)

        assert len(portfolio.equity_curve) == 3
        day0, day1, day2 = portfolio.equity_curve

        assert day0.timestamp == T0
        assert day0.total_value == pytest.approx(100_000 - 1.0)  # cash out, position in at cost

        assert day1.total_value == pytest.approx(day0.total_value + 100.0)  # unrealized gain of (110-100)*10

        # After the sell, position value is 0 and all value is cash. The
        # sell realizes an extra (120-110)*10 of gain over day1's 110 mark,
        # minus the $1 exit commission.
        assert day2.positions_value == 0.0
        assert day2.total_value == pytest.approx(day1.total_value + 100.0 - 1.0)

    def test_total_realized_and_unrealized_pnl_aggregate_across_symbols(self):
        portfolio = Portfolio(initial_cash=1_000_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=10, price=100.0, commission=0.0, symbol="AAPL"))
        portfolio.on_fill(fill(T0, "BUY", quantity=5, price=50.0, commission=0.0, symbol="MSFT"))
        portfolio.update_market_price(bar(T0, close=110.0, symbol="AAPL"))
        portfolio.update_market_price(bar(T0, close=40.0, symbol="MSFT"))

        assert portfolio.total_unrealized_pnl() == pytest.approx((110 - 100) * 10 + (40 - 50) * 5)

        portfolio.on_fill(fill(T1, "SELL", quantity=10, price=110.0, commission=0.0, symbol="AAPL"))
        assert portfolio.total_realized_pnl() == pytest.approx((110 - 100) * 10)


class TestPositionSizePctValidation:
    def test_default_is_point_nine_five(self):
        assert Portfolio(initial_cash=10_000).position_size_pct == pytest.approx(0.95)

    def test_boundary_of_one_is_accepted(self):
        assert Portfolio(initial_cash=10_000, position_size_pct=1.0).position_size_pct == 1.0

    def test_zero_is_rejected(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=10_000, position_size_pct=0.0)

    def test_above_one_is_rejected(self):
        with pytest.raises(ValueError):
            Portfolio(initial_cash=10_000, position_size_pct=1.5)


class TestGenerateOrder:
    def test_long_sizes_using_cash_position_size_pct_and_price(self):
        portfolio = Portfolio(initial_cash=10_000)  # position_size_pct=0.95
        portfolio.update_market_price(bar(T0, close=100.0))

        order = portfolio.generate_order(signal(T0, "LONG"))

        assert order is not None
        assert order.direction == "BUY"
        assert order.order_type == "MARKET"
        assert order.symbol == "AAPL"
        assert order.timestamp == T0
        assert order.quantity == 95  # int(10_000 * 0.95 / 100)

    def test_long_sizing_scales_with_signal_strength(self):
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.update_market_price(bar(T0, close=100.0))

        order = portfolio.generate_order(signal(T0, "LONG", strength=0.5))

        assert order.quantity == 47  # int(10_000 * 0.95 * 0.5 / 100)

    def test_long_while_already_holding_is_a_noop(self):
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=5, price=100.0))
        portfolio.update_market_price(bar(T0, close=100.0))

        assert portfolio.generate_order(signal(T0, "LONG")) is None

    def test_long_after_a_full_round_trip_is_allowed_again(self):
        # Position dict entry survives a full sell at quantity=0 -- the
        # "already holding" check must use quantity, not dict membership.
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=5, price=100.0))
        portfolio.on_fill(fill(T1, "SELL", quantity=5, price=100.0))
        assert "AAPL" in portfolio.positions  # entry still present, quantity 0
        portfolio.update_market_price(bar(T1, close=100.0))

        order = portfolio.generate_order(signal(T1, "LONG"))
        assert order is not None
        assert order.direction == "BUY"

    def test_long_with_insufficient_cash_for_even_one_share_returns_none(self):
        portfolio = Portfolio(initial_cash=1.0)
        portfolio.update_market_price(bar(T0, close=100.0))

        assert portfolio.generate_order(signal(T0, "LONG")) is None

    def test_long_with_no_known_price_raises(self):
        portfolio = Portfolio(initial_cash=10_000)
        with pytest.raises(PortfolioError):
            portfolio.generate_order(signal(T0, "LONG"))

    def test_exit_with_no_position_is_a_noop(self):
        portfolio = Portfolio(initial_cash=10_000)
        assert portfolio.generate_order(signal(T0, "EXIT")) is None

    def test_exit_after_a_full_round_trip_is_a_noop(self):
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=5, price=100.0))
        portfolio.on_fill(fill(T1, "SELL", quantity=5, price=100.0))

        assert portfolio.generate_order(signal(T1, "EXIT")) is None

    def test_exit_while_holding_sells_entire_position(self):
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.on_fill(fill(T0, "BUY", quantity=7, price=100.0))

        order = portfolio.generate_order(signal(T1, "EXIT"))

        assert order is not None
        assert order.direction == "SELL"
        assert order.quantity == 7
        assert order.order_type == "MARKET"

    def test_short_signal_raises_portfolio_error(self):
        portfolio = Portfolio(initial_cash=10_000)
        portfolio.update_market_price(bar(T0, close=100.0))

        with pytest.raises(PortfolioError):
            portfolio.generate_order(signal(T0, "SHORT"))
