"""Survivorship-bias comparison: the actual payoff of data/universe.py's
survivorship_biased toggle. main.py's single-symbol AAPL backtest never
touches the universe module at all -- AAPL survived, so toggling the flag
on it alone would trivially show 0% difference. This script instead runs
the SAME strategy/date-range/starting-cash across a small curated pool of
real S&P 500 constituents, once restricted to the point-in-time-accurate
membership and once restricted to today's (survivorship-biased) roster, and
compares the resulting tearsheets.

Why a curated pool, not the full ~500-name index: yfinance (this project's
only data source -- see data/loader.py) reliably has NO price history at
all for most acquired/delisted tickers. Verified directly: CELG, CERN, BCR,
AGN, DNB, RTN, and MON (all real 2015 S&P 500 constituents later removed
by M&A) each returned "possibly delisted; no data found." This is a
separate, real limitation from the universe-membership one, and is
documented here rather than silently worked around by only ever picking
names that happen to still have data.

The 20-ticker pool below was checked ticker-by-ticker for full data
availability over 2015-01-01..2024-12-31 before being chosen:
  - 12 names that were S&P 500 constituents in 2015 AND still are today
    (the overlap -- present in both universe views).
  - 8 names that were 2015 constituents but are NOT in today's roster:
    BBBY, CAG, GT, MAT, UNM, XRAY have full 2015-2024 data (still-trading
    companies that simply fell out of the index -- BBBY's history includes
    its real 2023 bankruptcy, a genuine survivorship-bias example). TWX is
    kept in deliberately even though its data stops mid-backtest (acquired
    by AT&T, closed 2018-06-14) specifically to exercise -- and document --
    correct handling of a constituent whose data runs out before the
    backtest's end date (see _aggregate_curves). M (Macy's, still trading)
    fills the 8th slot as of the fix below.

  POST-STEP-6 FIX: this pool originally used COL (Rockwell Collins) in
  TWX's slot. It was replaced with M after discovering yfinance's `COL`
  ticker does not actually return Rockwell Collins data -- its price
  series ($0.05-$0.75, 2015-2020) doesn't match the real company (which
  traded $92-$141 before its Nov 2018 UTC acquisition) and never stops
  trading at the real delisting date. Confirmed as a genuine data bug, not
  a false alarm, by checking it against TWX: TWX's yfinance metadata looks
  equally suspicious (quoteType MUTUALFUND) but its actual price series
  and delisting date both check out as real Time Warner data, so the two
  cases needed to be verified independently rather than assumed identical.
  Full investigation and before/after tearsheet numbers in CLAUDE.md.

Not a claim that this pool is representative of the whole index's
survivorship effect -- it's sized for a fast, reproducible demo with real,
verifiable data, per the same "reproducibility over completeness" tradeoff
documented for data/universe.py itself in CLAUDE.md.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from analytics.performance import format_tearsheet, generate_tearsheet
from data.loader import load_market_events
from data.universe import get_universe
from engine.events import FillEvent
from engine.execution import ExecutionHandler
from engine.portfolio import Portfolio, PortfolioError, PortfolioSnapshot
from engine.runner import run_backtest
from strategies.base import Strategy
from strategies.sma_crossover import SMACrossoverStrategy

AS_OF = "2015-01-01"
END = "2024-12-31"
FAST_WINDOW = 20
SLOW_WINDOW = 50
INITIAL_CASH = 100_000.0

# Default strategy_factory/label for this module's own SMA comparison.
# strategies/rsi_mean_reversion.py reuses main() below (via
# compare_survivorship_rsi.py) by passing its own factory/label instead --
# see that script and CLAUDE.md's Step 6 design decisions for why this was
# parameterized rather than duplicated.
_DEFAULT_STRATEGY_FACTORY: Callable[[], Strategy] = lambda: SMACrossoverStrategy(
    fast_window=FAST_WINDOW, slow_window=SLOW_WINDOW
)
_DEFAULT_LABEL = f"SMA({FAST_WINDOW}, {SLOW_WINDOW}) crossover"

CANDIDATE_POOL = frozenset(
    {
        # Survived to today (in both the 2015 point-in-time and current universe).
        "AAPL", "ABT", "ACN", "ADBE", "ADP", "AMGN", "AMZN", "AON", "APD", "ADM", "AMAT", "AIG",
        # 2015 constituents dropped from today's roster.
        "BBBY", "CAG", "GT", "MAT", "UNM", "XRAY", "TWX", "M",
    }
)


def _run_one(
    symbol: str, cash: float, strategy_factory: Callable[[], Strategy]
) -> tuple[list[PortfolioSnapshot], list[FillEvent]]:
    bars = load_market_events(symbol, AS_OF, END)
    strategy = strategy_factory()
    portfolio = Portfolio(initial_cash=cash)
    execution = ExecutionHandler()
    try:
        run_backtest(bars, strategy, portfolio, execution)
    except PortfolioError as exc:
        # Portfolio.generate_order's documented residual risk: its
        # fixed-fraction sizer estimates order size off the signal bar's
        # close, but the fill happens at a later, unknown price -- a large
        # enough overnight gap between the two can exceed that headroom.
        # Splitting $100k across up to 20 names (vs. main.py's single
        # full-size position) shrinks the per-symbol dollar buffer, so
        # this is more likely to actually bite here than in main.py. Given
        # this is one name out of a curated pool built for a demo, one
        # name's gap risk shouldn't abort the whole comparison -- freeze
        # its contribution at its last successful snapshot instead, the
        # same forward-fill treatment TWX/COL's real M&A-truncated data
        # already gets in _aggregate_curves.
        print(f"  [{symbol}] stopped early ({exc}); freezing its contribution from here on.")
    return portfolio.equity_curve, portfolio.trade_log


def _aggregate_curves(curves: dict[str, list[PortfolioSnapshot]]) -> list[PortfolioSnapshot]:
    """Sum per-symbol equity curves into one aggregate curve.

    Not a naive timestamp-matched sum: TWX and COL's curves end mid-backtest
    (real M&A delistings truncate their available data). Summing only
    matching timestamps would silently drop their entire value from the
    aggregate for every date after their last bar, understating the
    "accurate" universe's result relative to what an investor actually
    held. Instead, each symbol's (cash, positions_value) series is
    reindexed to the full union of trading dates across the pool and
    forward-filled, so a delisted symbol's last known value persists flat
    for the remainder of the backtest before the per-date sum. This avoids
    the silent drop, but does NOT model the real cash/stock payout an
    actual M&A would have delivered -- a documented simplification, not an
    oversight.
    """
    frames: dict[str, pd.DataFrame] = {}
    for symbol, curve in curves.items():
        frames[symbol] = pd.DataFrame(
            {
                "cash": [s.cash for s in curve],
                "positions_value": [s.positions_value for s in curve],
            },
            index=pd.Index([s.timestamp for s in curve], name="timestamp"),
        )

    # Plain python datetimes, not pd.Timestamp, so the aggregate
    # PortfolioSnapshots below match the type every other snapshot in the
    # codebase carries.
    all_timestamps = sorted(
        {ts.to_pydatetime() for frame in frames.values() for ts in frame.index}
    )
    total_cash = pd.Series(0.0, index=all_timestamps)
    total_positions_value = pd.Series(0.0, index=all_timestamps)
    for frame in frames.values():
        filled = frame.reindex(all_timestamps).ffill()
        total_cash = total_cash.add(filled["cash"], fill_value=0.0)
        total_positions_value = total_positions_value.add(filled["positions_value"], fill_value=0.0)

    return [
        PortfolioSnapshot(
            timestamp=ts,
            cash=float(total_cash.loc[ts]),
            positions_value=float(total_positions_value.loc[ts]),
        )
        for ts in all_timestamps
    ]


def run_universe_backtest(
    symbols: list[str], strategy_factory: Callable[[], Strategy]
) -> tuple[list[PortfolioSnapshot], list[FillEvent]]:
    per_symbol_cash = INITIAL_CASH / len(symbols)
    curves: dict[str, list[PortfolioSnapshot]] = {}
    all_trades: list[FillEvent] = []
    for symbol in symbols:
        curve, trades = _run_one(symbol, per_symbol_cash, strategy_factory)
        curves[symbol] = curve
        all_trades.extend(trades)
    all_trades.sort(key=lambda fill: fill.timestamp)
    return _aggregate_curves(curves), all_trades


def main(
    strategy_factory: Callable[[], Strategy] = _DEFAULT_STRATEGY_FACTORY,
    label: str = _DEFAULT_LABEL,
) -> None:
    accurate_members = get_universe(AS_OF, survivorship_biased=False)
    biased_members = get_universe(AS_OF, survivorship_biased=True)
    accurate_symbols = sorted(CANDIDATE_POOL & accurate_members)
    biased_symbols = sorted(CANDIDATE_POOL & biased_members)
    dropped = sorted(set(accurate_symbols) - set(biased_symbols))

    print("Survivorship bias comparison")
    print("=============================")
    print(f"Strategy:                     {label}")
    print(f"As of:                        {AS_OF}")
    print(f"Candidate pool:               {len(CANDIDATE_POOL)} tickers (see module docstring)")
    print(f"Accurate (point-in-time) universe: {len(accurate_symbols)} names -> {accurate_symbols}")
    print(f"Biased (today's roster) universe:  {len(biased_symbols)} names -> {biased_symbols}")
    print(f"Silently dropped by the biased view: {dropped}")
    print()

    accurate_curve, accurate_trades = run_universe_backtest(accurate_symbols, strategy_factory)
    biased_curve, biased_trades = run_universe_backtest(biased_symbols, strategy_factory)

    accurate_sheet = generate_tearsheet(accurate_curve, accurate_trades)
    biased_sheet = generate_tearsheet(biased_curve, biased_trades)

    print(format_tearsheet(accurate_sheet, title="Accurate universe (point-in-time)"))
    print()
    print(format_tearsheet(biased_sheet, title="Biased universe (today's roster)"))
    print()
    print("Difference (accurate - biased):")
    print(f"  Total return: {(accurate_sheet['total_return'] - biased_sheet['total_return']) * 100:+.2f} pp")
    print(f"  CAGR:         {(accurate_sheet['cagr'] - biased_sheet['cagr']) * 100:+.2f} pp")
    print(f"  Sharpe:       {accurate_sheet['sharpe_ratio'] - biased_sheet['sharpe_ratio']:+.2f}")
    print(
        f"  Max drawdown: {(accurate_sheet['max_drawdown_pct'] - biased_sheet['max_drawdown_pct']) * 100:+.2f} pp"
    )


if __name__ == "__main__":
    main()
