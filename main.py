"""End-to-end pipeline smoke test: real data through the full event chain.

Symbol/range/parameters (chosen, not tuned): AAPL, 2015-01-01 through
2024-12-31 -- a full decade spanning multiple regimes (the 2018 selloff,
2020's crash and recovery, 2022's rate-hike drawdown), long enough that the
strategy sees several real crossovers rather than being cherry-picked around
one flattering run. SMA(20, 50) is the classic "golden/death cross" pairing.
initial_cash=$100,000 is an arbitrary round number.

This script's job is to prove the pipeline works, not to demonstrate alpha
-- see CLAUDE.md.
"""

from __future__ import annotations

from analytics.performance import format_tearsheet, generate_tearsheet
from data.loader import load_market_events
from engine.execution import ExecutionHandler
from engine.portfolio import Portfolio, PortfolioError
from engine.runner import run_backtest
from strategies.sma_crossover import SMACrossoverStrategy

SYMBOL = "AAPL"
START = "2015-01-01"
END = "2024-12-31"
FAST_WINDOW = 20
SLOW_WINDOW = 50
INITIAL_CASH = 100_000.0


def main() -> None:
    bars = load_market_events(SYMBOL, START, END)
    strategy = SMACrossoverStrategy(fast_window=FAST_WINDOW, slow_window=SLOW_WINDOW)
    portfolio = Portfolio(initial_cash=INITIAL_CASH)
    execution = ExecutionHandler()

    try:
        run_backtest(bars, strategy, portfolio, execution)
    except PortfolioError as exc:
        print(f"Backtest aborted by a Portfolio invariant violation: {exc}")
        raise

    print(f"Symbol:            {SYMBOL}")
    print(f"Date range:        {START} to {END} ({len(bars)} bars)")
    print(f"Strategy:          SMA({FAST_WINDOW}, {SLOW_WINDOW}) crossover")
    print()
    print(format_tearsheet(generate_tearsheet(portfolio.equity_curve, portfolio.trade_log)))


if __name__ == "__main__":
    main()
