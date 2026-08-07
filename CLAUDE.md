# Quant Backtesting Engine — Project Context

## What this is
An event-driven backtesting engine built from scratch (no backtrader/zipline/vectorbt).
The point of this project is to demonstrate correct handling of look-ahead bias,
survivorship bias, and realistic execution simulation — not just to produce a
profitable-looking equity curve. Correctness and honesty of the backtest matter
more than strategy performance.

## Core design principle: no look-ahead
This is the single most important invariant in the codebase. A strategy must never
have access to data timestamped after the current event being processed. Every
design decision should be checked against this rule. If you (Claude) are ever
unsure whether a piece of code violates this, flag it explicitly rather than
assuming it's fine.

## Architecture: event-driven, not vectorized
Everything flows through a central event queue, processed strictly in time order:

```
MarketEvent  → Strategy generates → SignalEvent
SignalEvent  → Portfolio sizes into → OrderEvent
OrderEvent   → Execution simulates → FillEvent
FillEvent    → Portfolio updates cash/positions
```

### Directory structure
```
├── data/
│   ├── loader.py       # fetch & cache historical data (parquet on disk)
│   └── universe.py      # point-in-time universe membership (survivorship bias handling)
├── engine/
│   ├── events.py         # MarketEvent, SignalEvent, OrderEvent, FillEvent (dataclasses)
│   ├── event_queue.py    # main event loop
│   ├── portfolio.py      # position tracking, cash accounting, PnL
│   └── execution.py      # fill simulation: slippage, commissions, partial fills
├── strategies/
│   ├── base.py           # abstract Strategy class (on_market_event -> SignalEvent | None)
│   └── ...               # one file per strategy
├── analytics/
│   └── performance.py    # Sharpe, Sortino, max drawdown, win rate, tearsheet
├── validation/
│   └── walk_forward.py   # walk-forward train/test harness
└── tests/
    └── ...                # unit tests, especially for portfolio accounting & event queue
```

## Build order (do NOT build everything in one pass)
1. Events + event queue skeleton, tested with dummy/synthetic data
2. Data loader with point-in-time universe support (toggle: survivorship-biased
   universe vs. accurate historical universe) — this is a key differentiator, don't skip it
3. Portfolio + execution simulation (cash accounting, slippage, commissions)
4. One simple strategy (SMA crossover) to validate the full pipeline end-to-end
5. Performance analytics module
6. A second, more interesting strategy (momentum or pairs/stat-arb)
7. Walk-forward validation harness

Work through these in order. Don't jump ahead to strategies before the engine
plumbing is tested and correct — bugs in fill simulation or cash accounting are
the easiest way to produce a backtest that looks profitable but is silently wrong.

## Testing philosophy
Write tests for `engine/` components before or alongside implementation, especially:
- Event queue processes events in strict time order
- Portfolio cash/position accounting is correct after sequences of fills
- Execution simulation applies slippage/commissions as configured
- No component can access data from a future timestamp

## Data conventions
- Store historical data as parquet, partitioned by date or symbol
- Universe membership must be point-in-time (e.g., don't use today's S&P 500
  list for a 2015 backtest — pull historical constituents)
- Explicitly track and document data source, date range, and any survivorship
  bias tradeoffs made due to data availability

## Style / conventions
- Python 3.11+, type hints throughout
- Events are immutable dataclasses
- Prefer explicit over clever — this code will be read by recruiters
- Every non-obvious design decision (bias tradeoffs, assumptions) gets a comment
  or a note in README.md, not just buried in code

## Current status
_(update this section as the project progresses)_
- [x] Event queue skeleton
- [ ] Data loader + point-in-time universe
- [ ] Portfolio + execution
- [ ] First strategy (SMA crossover) validating full pipeline
- [ ] Performance analytics
- [ ] Second strategy
- [ ] Walk-forward validation