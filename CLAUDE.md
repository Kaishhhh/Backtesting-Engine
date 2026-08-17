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

### Step 5 — Performance analytics + survivorship bias comparison
- **Annualization/risk-free-rate assumptions are named constants, not magic
  numbers.** `analytics/performance.py` defines `TRADING_DAYS_PER_YEAR = 252`,
  `CALENDAR_DAYS_PER_YEAR = 365.25` (used for CAGR's elapsed-time calc, via
  `total_seconds()` rather than `.days` so a multi-year span isn't biased by
  truncating the sub-day remainder), and `DEFAULT_RISK_FREE_RATE = 0.0`.
- **Sharpe/Sortino's zero-denominator case returns a named constant
  (`UNDEFINED_RATIO_FALLBACK = 0.0`), not `inf`/`NaN`/an exception.** This
  covers three situations identically: too few return periods to measure
  volatility at all, a genuinely flat equity curve (zero volatility), and —
  Sortino only — zero downside deviation (no losing periods). All three mean
  "this data can't support a risk-adjusted answer," and `0.0` is the honest
  way to say that rather than claiming a number the data doesn't justify.
  Tests cover this as a *distinct* code path from "the real calculation
  happens to equal zero" (e.g. returns `[+0.01, -0.01]` has nonzero
  volatility and a genuine 0 mean, exercising actual division; a flat curve
  short-circuits on zero volatility instead).
- **Round-trip win/loss logic didn't exist before this step — it was built
  from scratch, not reused.** Step 4's "27 round trips" print was a naive
  `len(trade_log) // 2` fill count, not a real BUY/SELL matcher (confirmed
  by grep before writing anything). `extract_round_trips()` does genuine
  FIFO lot-matching per symbol (a `deque` of open BUY lots; a SELL consumes
  the oldest lot(s) first, splitting into multiple `RoundTrip`s on a partial
  match), general enough for a future strategy that buys in multiple
  tranches. Each `RoundTrip.pnl` nets out *both* entry and exit commission —
  deliberately more conservative than `Position.realized_pnl`'s running
  total, which only nets the sell-side commission (buy-side commission
  already reduced cash at entry without being logged against "realized"
  PnL there). Both are correct; they answer different questions. `main.py`
  now uses this real matcher instead of the naive estimate — for the same
  AAPL/2015-2024 run, it happens to also land on 27, confirming the
  heuristic wasn't actually wrong for this specific always-single-position
  strategy, just uninformative for anything that partially fills.
- **Survivorship-bias gap, flagged to the user before writing any
  comparison code, per their explicit request.** The single-symbol AAPL
  backtest never touches `data/universe.py` at all (AAPL survived, so
  toggling the flag on it alone shows a trivial 0% difference) — this was
  surfaced explicitly rather than silently building a comparison that
  couldn't demonstrate anything. Presented three options (full concurrent
  multi-symbol engine rework; membership-list-only diff with no P&L; or
  independent single-symbol backtests summed into an aggregate curve); the
  user picked the third. `compare_survivorship.py` runs a curated 20-ticker
  pool (12 survivors + 8 real 2015 constituents later dropped from the
  index, including `BBBY`'s real 2023 bankruptcy) through the *unmodified*
  `run_backtest()` independently per symbol — no changes needed to
  `engine/runner.py` or `Portfolio`'s sizing — then sums the resulting
  equity curves. **Result on the real data**: the biased (today's-roster)
  universe shows +88.89% total return vs. the accurate (point-in-time)
  universe's +50.13% — a textbook demonstration that excluding
  since-dropped constituents inflates apparent performance, not just a
  membership-count difference.
- **yfinance has no usable data at all for most acquired/delisted
  tickers — a second, separate limitation from the universe-membership
  one, verified directly rather than assumed.** `CELG`, `CERN`, `BCR`,
  `AGN`, `DNB`, `RTN`, `MON` (all real 2015 S&P 500 constituents later
  removed by M&A) each returned "possibly delisted; no data found." The
  20-ticker comparison pool was chosen only from names checked to actually
  have data, documented in `compare_survivorship.py`'s module docstring as
  a scope/data-availability tradeoff, not a claim of index-wide
  representativeness.
- **Forward-fill when aggregating per-symbol equity curves of different
  lengths.** Two pool members (`TWX`, `COL`) have real M&A-truncated data
  (acquired mid-backtest, no bars afterward). Naively summing per-symbol
  curves by matching timestamp would silently drop their entire value from
  the aggregate for every date after their last bar, understating the
  accurate universe's result. `_aggregate_curves()` reindexes every
  symbol's (cash, positions_value) series to the full union of trading
  dates and forward-fills before summing, so a delisted symbol's last known
  value persists flat instead of vanishing — documented as a
  simplification (it does not model the real cash/stock payout an actual
  M&A would deliver), not silently assumed away.
- **A per-symbol `PortfolioError` in the multi-name comparison is caught
  and the symbol's contribution is frozen, rather than aborting the whole
  run.** Splitting $100k across up to 20 names (vs. `main.py`'s single
  full-size position) shrinks each symbol's absolute cash buffer under
  `generate_order`'s 0.95 headroom, making the Step 4-documented "large gap
  exceeds headroom" residual risk considerably more likely to actually
  fire — and it did, for real, on `BBBY`, `COL`, and `UNM`. One name's gap
  risk failing shouldn't invalidate a 20-name comparison, so `_run_one()`
  catches `PortfolioError` and returns whatever equity curve/trade log had
  already accumulated, letting the same forward-fill aggregation treat it
  identically to `TWX`/`COL`'s data-truncation case. `engine/runner.py`
  itself is untouched — this is a caller-level decision specific to running
  many independent backtests, not a change to the engine's fail-loud
  philosophy (a single-symbol run via `main.py` still lets the error
  propagate uncaught).
- **Found and fixed: `auto_adjust` occasionally produces an open/close a
  hair outside `[low, high]`.** Discovered for real (not hypothetically)
  while building this step: e.g. `ADP` on 2018-10-30 came back with
  `close=115.84330749511719` against `high=115.84330749511717` (a ~2e-14
  gap), and other symbol/dates were off by as much as ~7.6e-6 — floating-
  point noise in the adjustment arithmetic, not a real OHLC inconsistency,
  and confirmed to vary slightly between separate fetches of the identical
  `[symbol, date]`. `data/loader.py` now clamps `open`/`close` back onto
  the `[low, high]` boundary when the gap is within a `$0.01` tolerance
  (~1000x the largest gap observed) — anything larger is left untouched
  and still raises via `MarketEvent.__post_init__`, since that's a real
  data problem worth seeing, not noise to hide.
- **Known, unfixed limitation surfaced during the above investigation:
  `load_ohlcv`'s cache-hit check effectively never hits for a `start` date
  that falls on a non-trading day (e.g. any "YYYY-01-01" — New Year's Day
  is always a market holiday), since the cache's recorded start is the
  first actual *trading* day, which is always after the requested calendar
  start.** This means both `main.py` and `compare_survivorship.py`
  silently do a full live re-fetch on every single run rather than reading
  the cache, which is what surfaced the adjustment-noise variability above
  in the first place. This is a latent gap in Step 2's (already-committed,
  already-documented-as-a-tradeoff) coarse-grained caching design, not
  something introduced by this step. Deliberately **not** fixed here —
  it's a Step 2 concern, not a Step 5 one, and fixing it well means
  distinguishing "genuinely uncached" from "cached, just starts on a
  holiday" without a trading-calendar dependency. Flagged here so it isn't
  silently lost; worth a dedicated pass later.
- **Validation runs**: `main.py` (AAPL, unchanged parameters) still ends
  around $341k/+241% (exact cents vary run-to-run now that the caching gap
  above means every run re-fetches — see that note). `compare_survivorship.py`
  (AS_OF=2015-01-01, same SMA(20,50)/$100k as `main.py`): accurate universe
  +50.13% (CAGR 4.15%, Sharpe 0.44, max drawdown -18.20%) vs. biased
  universe +88.89% (CAGR 6.57%, Sharpe 0.60, max drawdown -21.08%) — see
  chat history for full side-by-side tearsheets.

## Current status
_(update this section as the project progresses)_
- [x] Event queue skeleton
- [x] Data loader + point-in-time universe
- [x] Portfolio + execution
- [x] First strategy (SMA crossover) validating full pipeline
- [x] Performance analytics
- [ ] Second strategy
- [ ] Walk-forward validation