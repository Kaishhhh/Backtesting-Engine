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

### Step 3 — Portfolio + execution
- **Fills happen on the next bar's open, never the bar that produced the
  order.** A SignalEvent derived from bar T's data cascades into an
  OrderEvent timestamped T in the same event-queue pass; filling that order
  against bar T's own close (or open) would mean trading on a price already
  known when the decision was made — a same-bar fill is look-ahead bias in
  disguise. `ExecutionHandler` holds every order pending and only fills it
  against a `MarketEvent` strictly later than the order's own timestamp; the
  check (`order.timestamp >= market_event.timestamp` → stays pending) is a
  runtime guard, not just a documented convention, matching the event
  queue's own approach to invariant enforcement.
- **Slippage only applies to MARKET fills, never LIMIT fills.** A limit
  order's entire purpose is a price guarantee; applying slippage to it would
  contradict that guarantee. Slippage model is fixed-bps-of-price
  (`FixedBpsSlippage`), pushing MARKET fills against the trader (BUYs fill
  higher, SELLs fill lower) — the simplest model that's still directionally
  honest. Commission (`PerTradeCommission`: fixed + optional per-share)
  applies to every fill regardless of order type.
- **LIMIT fill logic checks the bar's [low, high] range, not just its
  open.** A buy limit fills if the bar's low reached the limit price, at
  `min(open, limit)` — so a gap-down open more favorable than the limit
  still gives the trader that better price. Orders that never cross stay
  pending indefinitely rather than being forced to fill.
- **Long-only, cash account, no margin/leverage — enforced, not just
  assumed.** `Portfolio` rejects (raises `PortfolioError`) any BUY fill that
  would take cash negative and any SELL fill that would take a position
  negative (i.e. shorting). Per project scope, this project does not model
  margin or short positions unless that's a deliberate future decision, not
  an accident of missing a check.
- **Commission is not folded into a position's cost basis.** `avg_cost` is
  the volume-weighted average *fill price* only; commission reduces cash
  immediately on every fill and is subtracted again from realized PnL when a
  position is sold. Keeps "what did I pay for the shares" separate from
  "what did trading cost me," while total portfolio value still nets out
  correctly either way.
- **Equity-curve cadence is the caller's decision, not Portfolio's.**
  `Portfolio.record_snapshot(timestamp)` must be called explicitly; the
  class doesn't assume how many symbols or bars constitute "one point in
  time" for the eventual multi-symbol backtest loop.
- **Partial fills are explicitly deferred, not silently dropped.** The
  directory-structure sketch above mentions partial fills as a future
  `execution.py` capability; this step's orders always fill in full or not
  at all (LIMIT) / in full (MARKET). Modeling fills limited by a bar's
  volume is a reasonable later addition, not an oversight.

### Step 4 — SMA crossover strategy + end-to-end pipeline
- **Strategy state is a fixed-size rolling window, never a stored
  history.** `SMACrossoverStrategy` keeps a per-symbol
  `collections.deque(maxlen=slow_window)` of closes and recomputes both
  SMAs from it on every bar; there is no DataFrame of "everything seen so
  far" anywhere. This isn't just an implementation convenience — it's the
  mechanism that makes `Strategy.on_market_event(event: MarketEvent) ->
  SignalEvent | None` a real no-look-ahead boundary rather than a
  convention: the interface physically cannot hand a strategy more than one
  bar at a time, so any "memory" it has must be built the same way a live,
  streaming strategy would have to build it. No signal is emitted until the
  deque is full (no garbage from an undersized window), and — more subtly —
  no signal is emitted on the very bar the window *first* becomes full
  either: that bar only establishes a baseline fast-vs-slow relationship,
  not an observed crossover, and trading on it would mean acting on trend
  history from before the strategy started watching.
- **Signal-to-order sizing landed on `Portfolio.generate_order()`, closing
  a real gap in Steps 1-3.** The architecture diagram always said
  "SignalEvent → Portfolio sizes into → OrderEvent," but no such method
  existed until this step — sizing needs cash/position state, which only
  `Portfolio` owns, so it couldn't reasonably live on the stateless
  `Strategy` side. New `Portfolio(initial_cash, position_size_pct=0.95)`
  param sizes LONG orders as `cash * position_size_pct *
  signal.strength / latest_close`. Using the current bar's close to *size*
  an order is not a look-ahead violation (it's already-observed data at
  decision time) — the actual fill still happens at a later, unknown price
  via `ExecutionHandler`'s existing next-bar-open mechanism. The 0.95
  default leaves headroom for slippage/commission/ordinary price movement
  between the signal bar's close and the fill bar's open, but that headroom
  isn't a guarantee: a large enough gap can still make `on_fill` raise
  `PortfolioError` for insufficient cash, and that's allowed to propagate
  out of the backtest loop uncaught rather than being silently clamped —
  consistent with this project's fail-loud philosophy elsewhere.
- **No pyramiding, no double-exit, SHORT signals raise.** A LONG signal
  while already holding a position (`quantity > 0`, not mere dict
  membership — a fully-sold `Position` stays in `Portfolio.positions` with
  `quantity == 0`) is a no-op; likewise EXIT while flat. A `SHORT` signal
  reaching `generate_order` raises `PortfolioError` rather than being
  silently ignored, matching `_apply_buy`/`_apply_sell`'s existing
  philosophy that long-only invariant violations should fail loudly, since
  a SHORT here means either a misconfigured strategy or a real bug.
- **The runner's fill-before-strategy ordering within a bar is load-bearing,
  not incidental.** `engine/runner.py`'s `run_backtest` processes a bar's
  `MARKET` event by resolving pending fills first (`queue.put()` any
  resulting `FillEvent`s), *then* updating the mark price, *then* calling
  the strategy. Because `EventQueue` is strict FIFO, this guarantees a
  same-bar fill (from an order placed on the prior bar) is applied to
  `Portfolio` before that same bar's new signal reaches
  `generate_order()`'s "already holding?" check — reversing the order would
  let sizing act on stale, pre-fill state.
- **Validation run**: AAPL, 2015-01-01 to 2024-12-31, SMA(20, 50),
  $100,000 initial cash (see `main.py`'s docstring for the "why this
  symbol/range" rationale — a decade spanning multiple regimes, not
  cherry-picked). Ran clean end-to-end: 2,516 bars, 55 fills (27 round
  trips), ending value $341,072.22 (+241.07%). This is a pipeline
  correctness check, not a claim of strategy alpha.

## Current status
_(update this section as the project progresses)_
- [x] Event queue skeleton
- [x] Data loader + point-in-time universe
- [x] Portfolio + execution
- [x] First strategy (SMA crossover) validating full pipeline
- [ ] Performance analytics
- [ ] Second strategy
- [ ] Walk-forward validation