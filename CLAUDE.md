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
  chat history for full side-by-side tearsheets. **These `compare_survivorship.py`
  figures were later found to rest on bad data for one pool member (`COL`)
  and were corrected post-Step-6 — see "COL data-quality bug, found and
  fixed" below for the root cause and the corrected numbers. Left
  unedited here rather than silently rewritten, so the record shows what
  was originally reported and why it changed.**

### Step 6 — Second strategy (RSI mean-reversion)
- **Strategy choice: RSI mean-reversion, proposed and confirmed with the
  user before writing any code, per their explicit request.** Presented
  three options — RSI mean-reversion, N-day momentum/ROC, and Donchian
  channel breakout — and picked mean-reversion because it is a genuinely
  different *bet* from SMA crossover (reversion vs. trend continuation),
  not just different parameters on the same trend-following idea; the
  other two candidates were both still trend-following in spirit. User
  confirmed this choice via explicit sign-off before implementation began.
- **Cutler's (simple-average) RSI, not Wilder's smoothed RSI.**
  `RSIMeanReversionStrategy` computes average gain/loss as a plain mean
  over the trailing `period` bar-over-bar changes, recomputed fresh from a
  `deque(maxlen=period + 1)` every bar — deliberately not Wilder's
  exponential smoothing, which would make the indicator depend on the
  strategy's entire history since inception rather than just its current
  window. Cutler's variant is a standard, documented RSI alternative, not
  an invented simplification, and it keeps the same "fixed rolling window,
  no stored history" discipline `SMACrossoverStrategy` established in Step
  4 — the actual point of this step, proving `strategies/base.py`'s
  one-bar-at-a-time interface generalizes to a second, differently-shaped
  indicator.
- **Zone-based crossing detection (`"low"`/`"mid"`/`"high"`), not two
  independent boolean flags.** RSI < 30 → LONG (entering oversold), RSI >
  70 → EXIT (entering overbought); 30–70 inclusive is neutral. A signal
  fires only on a *transition into* an extreme zone (mirroring SMA
  crossover's "only on an observed crossing" rule) — leaving one extreme
  and landing in the neutral zone without reaching the opposite extreme
  emits nothing, since the reversion (or its exit) hasn't actually
  happened yet. Boundary values (RSI exactly 30 or 70) count as neutral,
  the same tie-goes-to-"not triggered" convention SMA crossover uses. No
  signal on the bar the window first becomes full, for the same reason as
  Step 4: that bar only establishes a baseline zone, not an observed
  crossing.
- **`compare_survivorship.py` parameterized (`strategy_factory` + `label`
  arguments on `main()`/`_run_one()`/`run_universe_backtest()`), not
  duplicated — flagged to the user before making the change, per their
  explicit request (same as the Step 5 survivorship-gap flag).** Presented
  two options: parameterize the existing script with a thin new
  `compare_survivorship_rsi.py` entry point reusing its aggregation/
  forward-fill/error-handling logic, or fork the whole ~150-line script.
  User picked parameterizing. `python compare_survivorship.py` with no
  arguments is unchanged (defaults to the original SMA crossover
  factory/label); `compare_survivorship_rsi.py` imports `main` and passes
  an `RSIMeanReversionStrategy` factory instead. `engine/` and
  `data/universe.py` were untouched.
- **Validation runs**: `main_rsi.py` (AAPL, 2015-2024, RSI(14) 30/70,
  $100,000, same range/cash as `main.py`) ends at $158,718.09 (+58.72%,
  CAGR 4.73%, Sharpe 0.33, max drawdown -29.94%, 29 round trips, 72.4% win
  rate) — a real, un-cherry-picked run, not tuned for this result.
  `compare_survivorship_rsi.py` (same 20-ticker pool as the SMA version):
  accurate universe +79.08% (CAGR 6.00%, Sharpe 0.56) vs. biased universe
  +124.04% (CAGR 8.40%, Sharpe 0.67) — the survivorship-bias effect
  reproduces with a second, differently-behaved strategy (519 round trips
  vs. the SMA run's much lower turnover), not just SMA crossover
  specifically. **`main_rsi.py`'s single-symbol AAPL numbers are unaffected
  and stand as final — it never touches `CANDIDATE_POOL`. The
  `compare_survivorship_rsi.py` pool figures above rested on the same bad
  `COL` data described below and were corrected afterward; see "COL
  data-quality bug, found and fixed."**
- **Found in this step, fixed in a dedicated follow-up pass immediately
  after: `COL`'s data in the `CANDIDATE_POOL` was not Rockwell Collins.**
  Discovered because this step's RSI run hit a `PortfolioError` on `COL`
  (a BUY sized at 93,540 shares against a $0.055 price) that looked too
  extreme to be the documented "overnight gap exceeds sizing headroom"
  case (`MAT`'s same-run error is that case — its price series is a
  plausible real Mattel history). Originally flagged here as found-but-
  not-fixed, since the `CANDIDATE_POOL` and its data were Step 5's scope
  and swapping/removing `COL` would change a result already documented as
  final. The user asked for a dedicated investigation and fix before
  moving on to Step 7; that follow-up is recorded below rather than
  silently folded into this bullet, so the record shows a bug existed and
  was then fixed, not that the numbers were quietly correct all along.

#### Post-Step-6 fix: COL data-quality bug, found and fixed
- **Root cause, confirmed against an external reference.** Real Rockwell
  Collins traded $92.76 (start of 2017) rising to $141.63 (Nov 2018, on
  China's regulatory approval of the United Technologies acquisition),
  delisted 2018-11-26 (deal terms: $93.33 cash + $46.67 in UTX stock per
  share, ~$30B total). `yfinance`'s `COL` series instead ranges
  $0.05–$0.75 for the *entire* 2015–2020 window and keeps trading years
  past the real delisting date; `yf.Ticker('COL').info` returns
  `quoteType=MUTUALFUND`, `shortName='789776'`, `exchange='YHD'` —
  garbage/placeholder metadata consistent with Yahoo having reassigned
  the `COL` symbol to an unrelated, effectively-defunct instrument after
  the real Rockwell Collins delisted.
- **Ruled out a false alarm by checking `TWX` the same way, since it
  shares the same suspicious metadata.** `TWX` also reports
  `quoteType=MUTUALFUND` with an equally garbled `shortName`/`exchange` —
  if metadata alone were the signal, `TWX` would look just as broken as
  `COL`. But `TWX`'s actual price *series* is legitimate real Time Warner
  data: ~$80 in Jan 2015 rising to ~$99, and trading stops exactly at
  2018-06-14/15, matching the real AT&T acquisition close date. Confirms
  `yfinance`'s per-symbol `.info` metadata is not trustworthy for this
  kind of check (it's stale/wrong for both tickers) but the actual OHLCV
  *history* can still be correct even when the metadata is garbage —
  each symbol had to be verified independently by its price series and
  known real-world facts, not by a shared metadata flag.
- **What `MarketEvent.__post_init__` catches vs. does not, and why this
  slipped through it.** Its four checks (`high >= low`; `open`/`close`
  each inside `[low, high]`; `volume >= 0`) validate *intra-bar*
  structural consistency only — is this one bar internally coherent. They
  have no concept of cross-bar continuity, price-magnitude plausibility,
  or data provenance (is this even the right instrument). A bar for the
  wrong security entirely is still a perfectly self-consistent OHLC bar,
  so it passes by construction — the validator's job was never "is this
  data correct," only "is this data internally sane," and those are
  different questions. Notably, a **day-over-day discontinuity check**
  specifically would **not** have caught this: `COL`'s bad series declines
  *smoothly* from $0.70 to $0.05 over four years, with no single sharp
  jump to flag.
- **Fix chosen and applied, after presenting options for sign-off:**
  1. `data/loader.py`: `fetch_ohlcv` now calls `_warn_if_implausibly_low`,
     which emits a `warnings.warn` (not a hard `LoaderError`) if a
     symbol's fetched closes never exceed `_MIN_PLAUSIBLE_PRICE = $1.00`
     over the requested range. Warn, not raise, matching
     `data/universe.py`'s existing precedent of warning rather than
     hard-failing on a data-*plausibility* concern (as opposed to a
     provable structural invariant, which does hard-fail elsewhere in
     this codebase). Deliberately narrow and cheap, tuned to this
     project's actual use case (S&P 500 large/mid-cap constituents) —
     not a general data-quality engine, and it would not catch a
     wrong-ticker bug that happens to map to another liquid,
     normally-priced stock. Verified: fires for `COL`, silent for `AAPL`
     and `M`.
  2. `compare_survivorship.py`: `CANDIDATE_POOL` swaps `COL` for `M`
     (Macy's) — verified `quoteType=EQUITY`, real price range
     $3.65–$45.36 across the full 2015–2024 range, a genuine 2015 S&P 500
     constituent (per `data/universe.py`'s vendored snapshot) later
     dropped from today's roster, not already used elsewhere in the pool
     or on the documented no-data list (`CELG`/`CERN`/`BCR`/`AGN`/`DNB`/`RTN`/`MON`).
     Pool stays at 20 names / 8 dropped-constituents, same structure as
     before. The module docstring's own description of `COL` (previously
     stated as "acquired by UTC in 2020") was also corrected to the real
     date (Nov 2018) while touching this text.
- **Corrected numbers, both scripts re-run end-to-end against the fixed
  pool:**

  | Run | Universe | Old (COL, bugged) | New (M, corrected) |
  |---|---|---|---|
  | SMA (`compare_survivorship.py`) | Accurate | +50.13% (CAGR 4.15%, Sharpe 0.44, DD -18.20%) | +52.71% (CAGR 4.33%, Sharpe 0.46, DD -18.98%) |
  | SMA (`compare_survivorship.py`) | Biased | +88.89% (CAGR 6.57%, Sharpe 0.60, DD -21.08%) | +88.86% (CAGR 6.57%, Sharpe 0.60, DD -21.09%) |
  | RSI (`compare_survivorship_rsi.py`) | Accurate | +79.08% (CAGR 6.00%, Sharpe 0.56) | +77.14% (CAGR 5.89%, Sharpe 0.54, DD -27.30%) |
  | RSI (`compare_survivorship_rsi.py`) | Biased | +124.04% (CAGR 8.40%, Sharpe 0.67) | +124.03% (CAGR 8.40%, Sharpe 0.67) |

  Biased-universe figures barely move (`COL`/`M` were never in that
  universe — it's today's roster, and neither ticker is a current S&P 500
  member — the small residual delta is the pre-existing run-to-run
  caching/re-fetch noise already documented in Step 5). Accurate-universe
  figures move more, since that's the universe `COL`/`M` actually
  populate: SMA's accurate return rises (+50.13% → +52.71%, `COL`'s
  near-worthless contribution before its early `PortfolioError` freeze
  understated the aggregate); RSI's accurate return *falls* slightly
  (+79.08% → +77.14%, since RSI traded `COL`'s degenerate series
  differently than SMA did before erroring out — this direction change
  is expected and not a sign of a new bug, just a different strategy
  reacting differently to a full vs. partial replacement of one pool
  member). The core finding survives the correction either way: the
  biased universe still meaningfully outperforms the accurate one for
  both strategies.
- **`main.py` and `main_rsi.py` (single-symbol AAPL) are unaffected and
  remain final as originally reported** — neither touches
  `CANDIDATE_POOL` or `data/universe.py`.
- **Not done as part of this fix**: a full re-verification of every other
  `CANDIDATE_POOL` ticker's price history against an external reference.
  `TWX` was checked (see above) and is fine; the other 12 "survived"
  names and `BBBY`/`CAG`/`GT`/`MAT`/`UNM`/`XRAY` were not re-checked
  beyond their original Step 5 "has full data" pass and the new loader
  warning (which fired for none of them on this run). The loader's new
  warning is a partial safety net going forward, not a substitute for
  that full re-verification, which remains a worthwhile dedicated pass
  later if the pool composition matters for something higher-stakes than
  this demo.

## Current status
_(update this section as the project progresses)_
- [x] Event queue skeleton
- [x] Data loader + point-in-time universe
- [x] Portfolio + execution
- [x] First strategy (SMA crossover) validating full pipeline
- [x] Performance analytics
- [x] Second strategy
- [ ] Walk-forward validation