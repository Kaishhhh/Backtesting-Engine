"""End-to-end pipeline run for RSIMeanReversionStrategy, directly comparable
to main.py's SMA crossover baseline: same symbol, date range, and starting
cash (AAPL, 2015-01-01 through 2024-12-31, $100,000), only the strategy
differs. RSI(14) is the classic default lookback; 30/70 are the classic
oversold/overbought thresholds -- not tuned for this symbol/range.

Like main.py, this script's job is to prove the pipeline works for a second,
conceptually different strategy (mean-reversion vs. main.py's trend-
following), not to demonstrate alpha -- see CLAUDE.md.
"""

from __future__ import annotations

from analytics.performance import format_tearsheet, generate_tearsheet
from data.loader import load_market_events
from engine.execution import ExecutionHandler
from engine.portfolio import Portfolio, PortfolioError
from engine.runner import run_backtest
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

SYMBOL = "AAPL"
START = "2015-01-01"
END = "2024-12-31"
PERIOD = 14
OVERSOLD = 30.0
OVERBOUGHT = 70.0
INITIAL_CASH = 100_000.0


def main() -> None:
    bars = load_market_events(SYMBOL, START, END)
    strategy = RSIMeanReversionStrategy(period=PERIOD, oversold=OVERSOLD, overbought=OVERBOUGHT)
    portfolio = Portfolio(initial_cash=INITIAL_CASH)
    execution = ExecutionHandler()

    try:
        run_backtest(bars, strategy, portfolio, execution)
    except PortfolioError as exc:
        print(f"Backtest aborted by a Portfolio invariant violation: {exc}")
        raise

    print(f"Symbol:            {SYMBOL}")
    print(f"Date range:        {START} to {END} ({len(bars)} bars)")
    print(f"Strategy:          RSI({PERIOD}) mean-reversion ({OVERSOLD:.0f}/{OVERBOUGHT:.0f})")
    print()
    print(format_tearsheet(generate_tearsheet(portfolio.equity_curve, portfolio.trade_log)))


if __name__ == "__main__":
    main()
