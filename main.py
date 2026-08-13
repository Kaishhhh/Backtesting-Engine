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

from datetime import datetime, time

from data.loader import load_ohlcv
from engine.events import MarketEvent
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

# Daily bars carry no intraday timestamp; 9:30 stands in for "market open",
# matching the convention already used throughout tests/ (e.g. T0 = datetime
# (2024, 1, 2, 9, 30) in tests/test_portfolio.py).
BAR_TIME = time(9, 30)


def _bars_for(symbol: str, start: str, end: str) -> list[MarketEvent]:
    df = load_ohlcv(symbol, start, end)
    return [
        MarketEvent(
            timestamp=datetime.combine(row.date, BAR_TIME),
            symbol=row.symbol,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume),
        )
        for row in df.itertuples(index=False)
    ]


def main() -> None:
    bars = _bars_for(SYMBOL, START, END)
    strategy = SMACrossoverStrategy(fast_window=FAST_WINDOW, slow_window=SLOW_WINDOW)
    portfolio = Portfolio(initial_cash=INITIAL_CASH)
    execution = ExecutionHandler()

    try:
        run_backtest(bars, strategy, portfolio, execution)
    except PortfolioError as exc:
        print(f"Backtest aborted by a Portfolio invariant violation: {exc}")
        raise

    final = portfolio.equity_curve[-1]
    num_trades = len(portfolio.trade_log)

    print(f"Symbol:            {SYMBOL}")
    print(f"Date range:        {START} to {END} ({len(bars)} bars)")
    print(f"Strategy:          SMA({FAST_WINDOW}, {SLOW_WINDOW}) crossover")
    print(f"Starting cash:     ${INITIAL_CASH:,.2f}")
    print(f"Ending cash:       ${portfolio.cash:,.2f}")
    print(f"Ending positions:  ${final.positions_value:,.2f}")
    print(f"Final total value: ${final.total_value:,.2f}")
    print(f"Total return:      {(final.total_value / INITIAL_CASH - 1) * 100:.2f}%")
    print(f"Number of fills:   {num_trades}")
    print(f"Round trips:       {num_trades // 2}")


if __name__ == "__main__":
    main()
