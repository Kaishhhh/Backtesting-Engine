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

## Design Decisions
_(recruiter-facing rationale for non-obvious choices — see also
data/reference/SOURCES.md for the full point-in-time data writeup)_

### Step 1 — Events + event queue
- **FIFO queue, not a timestamp-sorted heap.** MarketEvent → Signal → Order →
  Fill are causally dependent and often share a timestamp. The main loop
  design (pull one MarketEvent, fully drain everything it cascades into
  before pulling the next) means insertion order already equals both
  chronological *and* causal order. A heap would need an arbitrary tiebreaker
  for same-timestamp events, risking scrambling that causal chain — FIFO is
  simpler and can't make that mistake. As a backstop against a caller
  breaking that discipline, `EventQueue.put()` rejects any event timestamped
  earlier than the most recently dequeued event.
- **`__post_init__` validation on events** (OHLC consistency, positive
  quantities, LIMIT orders requiring a price). Not strictly required for a
  "skeleton," but cheap, and it catches exactly the kind of silent
  data-corruption bug that produces a backtest that *looks* profitable but
  is wrong — which is this project's stated failure mode to guard against.

### Step 2 — Data loader + point-in-time universe
- **Point-in-time S&P 500 membership: vendored free dataset, not a live
  scrape, not a paid vendor.** Evaluated three options: (a) scrape
  Wikipedia's "Selected changes" table live at runtime — free but explicitly
  labeled non-exhaustive by Wikipedia itself, fragile to scrape, and
  non-reproducible (a backtest's universe could change if the page is edited
  between runs); (b) vendor a static CSV
  ([fja05680/sp500](https://github.com/fja05680/sp500), MIT licensed) —
  free, covers 1996–2026, verified against an independent fact (TSLA added
  2020-12-21, dataset gets it exactly right), with candidly documented gaps
  (pre-2001 likely undercounts membership; post-2019 still inherits
  Wikipedia's incompleteness); (c) a paid vendor (Norgate/Sharadar/CRSP) —
  institutional-grade but out of scope without sign-off. Chose (b): baked a
  static parquet snapshot into the repo (reproducibility matters more than
  freshness for a backtester) with the limitations documented prominently in
  `data/universe.py`'s module docstring and `data/reference/SOURCES.md`, per
  the "don't silently pretend it's point-in-time accurate" requirement.
- **`survivorship_biased` toggle returns *today's* roster regardless of
  `as_of` when True.** This deliberately reproduces the classic mistake
  (backtesting 2015 with today's S&P 500 list) so `analytics/` can show the
  biased vs. accurate equity curves side-by-side later, rather than the
  toggle being a no-op difference.
- **`auto_adjust=True` on the yfinance fetch.** Unadjusted closes show a fake
  ~-75% "crash" at every 4:1 stock split (e.g. AAPL, Aug 2020) — adjusted
  prices are the only correct default for anything touching price levels.
- **Cache is coarse-grained (whole-range re-fetch on a miss), not
  gap-filling.** A request only partially covered by the cache re-fetches
  its *entire* range rather than just the missing delta. Simpler and can't
  get multi-gap reconciliation wrong; costs some redundant network calls,
  acceptable at this project's data volume.

## Current status
_(update this section as the project progresses)_
- [x] Event queue skeleton
- [x] Data loader + point-in-time universe
- [ ] Portfolio + execution
- [ ] First strategy (SMA crossover) validating full pipeline
- [ ] Performance analytics
- [ ] Second strategy
- [ ] Walk-forward validation