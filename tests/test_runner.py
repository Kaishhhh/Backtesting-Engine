"""Integration test for engine/runner.py: a short, fully hand-calculated
synthetic run through the whole pipeline -- EventQueue -> Strategy ->
Portfolio -> ExecutionHandler -> Portfolio -- exercising exactly one
LONG -> BUY -> fill and one EXIT -> SELL -> fill.

Bars use closes [10, 10, 10, 20, 20, 20, 5] with SMA(fast=2, slow=3):

  bar0 (10): undersized, no signal.
  bar1 (10): undersized, no signal.
  bar2 (10): window full, fast=slow=10 -> tie/baseline, no signal.
  bar3 (20): fast=avg(10,20)=15, slow=avg(10,10,20)=13.33 -> above ->
      LONG signal. Portfolio sizes a BUY of int(10_000 * 0.95 / 20) = 475
      shares (price = this bar's close, 20) and submits it -- pending,
      since ExecutionHandler never fills same-bar.
  bar4 (open=20, close=20): the pending BUY fills here (strictly later than
      bar3), at open=20 with zero slippage, $1 commission. Cost =
      475*20 + 1 = 9501, leaving cash = 10_000 - 9501 = 499.
      fast=avg(20,20)=20, slow=avg(10,20,20)=16.67 -> still above, no new
      signal.
  bar5 (20): fast=avg(20,20)=20, slow=avg(20,20,20)=20 -> tie counts as
      "not above" -> crossed down from the prior above state -> EXIT
      signal. Portfolio sizes a SELL of the full 475-share position.
  bar6 (open=5, close=5): the pending SELL fills here, at open=5, zero
      slippage, $1 commission. Proceeds = 475*5 - 1 = 2374, leaving
      cash = 499 + 2374 = 2873. fast=avg(20,5)=12.5, slow=avg(20,20,5)=15
      -> below, unchanged from bar5 -> no new signal.

Final expected state: cash=2873.0, flat position, 2 fills (one BUY, one
SELL), 7 equity-curve points (one per bar).
"""

from datetime import datetime, timedelta

import pytest

from engine.events import MarketEvent
from engine.execution import ExecutionHandler, FixedBpsSlippage
from engine.portfolio import Portfolio
from engine.runner import run_backtest
from strategies.sma_crossover import SMACrossoverStrategy

T0 = datetime(2024, 1, 2, 9, 30)


def ts(n: int) -> datetime:
    return T0 + timedelta(days=n)


def bar(n: int, close: float, open_: float | None = None, symbol: str = "AAPL") -> MarketEvent:
    o = open_ if open_ is not None else close
    return MarketEvent(
        timestamp=ts(n), symbol=symbol, open=o, high=max(o, close) + 1,
        low=min(o, close) - 1, close=close, volume=1_000,
    )


def build_bars() -> list[MarketEvent]:
    return [
        bar(0, close=10),
        bar(1, close=10),
        bar(2, close=10),
        bar(3, close=20),
        bar(4, close=20, open_=20),
        bar(5, close=20),
        bar(6, close=5, open_=5),
    ]


class TestEndToEndRun:
    def test_one_round_trip_produces_expected_final_state(self):
        bars = build_bars()
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=3)
        portfolio = Portfolio(initial_cash=10_000)
        execution = ExecutionHandler(slippage_model=FixedBpsSlippage(bps=0))

        run_backtest(bars, strategy, portfolio, execution)

        assert portfolio.cash == pytest.approx(2_873.0)
        assert portfolio.positions["AAPL"].quantity == 0

        assert len(portfolio.trade_log) == 2
        buy, sell = portfolio.trade_log

        assert buy.direction == "BUY"
        assert buy.quantity == 475
        assert buy.fill_price == pytest.approx(20.0)
        assert buy.commission == pytest.approx(1.0)
        assert buy.timestamp == ts(4)

        assert sell.direction == "SELL"
        assert sell.quantity == 475
        assert sell.fill_price == pytest.approx(5.0)
        assert sell.commission == pytest.approx(1.0)
        assert sell.timestamp == ts(6)

    def test_equity_curve_has_one_point_per_bar_and_matches_hand_calc(self):
        bars = build_bars()
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=3)
        portfolio = Portfolio(initial_cash=10_000)
        execution = ExecutionHandler(slippage_model=FixedBpsSlippage(bps=0))

        run_backtest(bars, strategy, portfolio, execution)

        assert len(portfolio.equity_curve) == len(bars)

        # bar3: LONG signal just submitted, not yet filled -- still all cash.
        assert portfolio.equity_curve[3].total_value == pytest.approx(10_000.0)

        # bar4: BUY filled at open=20; mark-to-market at this bar's close
        # (also 20) values the position at exactly what was paid for it.
        assert portfolio.equity_curve[4].cash == pytest.approx(499.0)
        assert portfolio.equity_curve[4].total_value == pytest.approx(9_999.0)

        # bar6: SELL filled at open=5, flat again -- total value is cash only.
        assert portfolio.equity_curve[6].positions_value == 0.0
        assert portfolio.equity_curve[6].total_value == pytest.approx(2_873.0)

    def test_no_trading_produces_a_flat_equity_curve_at_initial_cash(self):
        bars = [bar(n, close=10) for n in range(5)]  # never a full slow window's worth of movement
        strategy = SMACrossoverStrategy(fast_window=2, slow_window=3)
        portfolio = Portfolio(initial_cash=10_000)
        execution = ExecutionHandler()

        run_backtest(bars, strategy, portfolio, execution)

        assert portfolio.trade_log == []
        assert all(snap.total_value == pytest.approx(10_000.0) for snap in portfolio.equity_curve)
