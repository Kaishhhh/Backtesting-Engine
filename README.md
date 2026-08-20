# Backtesting Engine

An event-driven quant backtesting engine built from scratch with no
backtrader/zipline/vectorbt. This README covers setup and the data-source
caveats a reader should know about before trusting any numbers this engine
produces.

## Architecture

The engine is event-driven, not vectorized: everything flows through a
central queue and is processed strictly in chronological order. This is
the design choice that makes the core invariant of the project enforceable
rather than just assumed — **a strategy can never see data timestamped
after the event currently being processed.**

```
MarketEvent  → Strategy generates → SignalEvent
SignalEvent  → Portfolio sizes into → OrderEvent
OrderEvent   → Execution simulates → FillEvent
FillEvent    → Portfolio updates cash/positions
```

```
├── data/
│   ├── loader.py       # cache-first OHLCV fetching
│   └── universe.py     # point-in-time S&P 500 membership
├── engine/
│   ├── events.py        # MarketEvent, SignalEvent, OrderEvent, FillEvent
│   ├── event_queue.py   # FIFO queue enforcing chronological order
│   ├── portfolio.py     # cash accounting, positions, PnL, equity curve, sizing
│   ├── execution.py     # fill simulation: slippage, commissions
│   └── runner.py        # per-bar event-drain loop wiring it all together
├── strategies/
│   ├── base.py                  # abstract Strategy interface
│   ├── sma_crossover.py         # dual SMA crossover (Step 4, trend-following)
│   └── rsi_mean_reversion.py    # RSI mean-reversion (Step 6, mean-reversion)
├── analytics/
│   └── performance.py    # Sharpe, Sortino, max drawdown, tearsheet
├── validation/
│   └── walk_forward.py   # walk-forward train/test harness
├── main.py                       # end-to-end run: SMA crossover on AAPL
├── main_rsi.py                   # end-to-end run: RSI mean-reversion on AAPL
├── compare_survivorship.py       # survivorship-bias comparison (SMA crossover)
├── compare_survivorship_rsi.py   # same comparison, RSI mean-reversion
└── tests/                        # unit tests for every module above
```

## Setup

python -m venv .venv
.venv/Scripts/activate # Windows
pip install -e ".[dev]"
pytest

python main.py                      # SMA crossover strategy on AAPL, 2015-2024
python main_rsi.py                  # RSI mean-reversion strategy on AAPL, 2015-2024
python compare_survivorship.py      # survivorship-bias comparison, SMA crossover
python compare_survivorship_rsi.py  # survivorship-bias comparison, RSI mean-reversion


## Build order / status

Built incrementally, one stage at a time, with tests passing before moving
on — each stage below was its own commit.

- [x] **1. Event queue skeleton** — event types + FIFO queue, chronological
      order enforced at runtime (`EventQueueError` on out-of-order `put()`)
- [x] **2. Data loader + point-in-time universe** — cache-first market data,
      point-in-time index membership with a survivorship-bias toggle
- [x] **3. Portfolio + execution** — cash accounting, position tracking,
      slippage/commission simulation, next-bar-open fills
- [x] **4. First strategy (SMA crossover)** — validates the full pipeline
      end-to-end on real data
- [x] **5. Performance analytics** — Sharpe, Sortino, drawdown, tearsheet;
      also where the survivorship-bias toggle gets demonstrated side-by-side
      (`compare_survivorship.py`)
- [x] **6. Second strategy (RSI mean-reversion)** — conceptually opposite
      bet from SMA crossover (reversion vs. trend-following), proves the
      `Strategy` interface and `compare_survivorship.py`'s comparison
      generalize beyond one strategy's assumptions
- [ ] **7. Walk-forward validation** — rolling train/test splits instead of
      a single static backtest

## Design rationale

A few decisions worth calling out explicitly, since they're the parts of a
backtester that are easy to get subtly wrong:

- **FIFO queue, not a timestamp-sorted heap.** The standard event-driven
  pattern pulls one `MarketEvent`, then fully drains everything it
  causally spawns (Signal → Order → Fill, same timestamp) before pulling
  the next bar. Insertion order already equals chronological+causal order,
  so a heap would need an arbitrary tiebreaker for same-timestamp events
  that could scramble that causal chain instead of preserving it.
- **No-look-ahead is a runtime guarantee, not a convention.** `put()` on
  the event queue rejects any event timestamped earlier than the most
  recently dequeued event. This turns the core invariant of the project
  into something that fails loudly if violated, rather than something a
  reader has to trust the code respects.
- **Point-in-time universe over current-membership shortcuts.** Most DIY
  backtesters apply today's index membership retroactively, which silently
  bakes in survivorship bias. This engine vendors a real historical
  constituent dataset and exposes `survivorship_biased=True/False` as an
  explicit toggle, so the bias can be measured rather than hidden.
- **Vendored MIT-licensed data over live scraping or a paid vendor.** Live
  Wikipedia scraping isn't reproducible run-to-run; a paid vendor
  (Norgate/Sharadar/CRSP) was out of scope for a portfolio project. The
  vendored dataset was spot-checked against a known fact (TSLA joined the
  S&P 500 on 2020-12-21) before being trusted. Full tradeoff analysis in
  `data/reference/SOURCES.md`.
- **Fills happen on the next bar's open, never the bar that produced the
  order.** A same-bar fill would mean trading on a price already known when
  the decision was made — look-ahead bias in disguise. `ExecutionHandler`
  enforces this at runtime (an order timestamped the same as or after the
  incoming bar stays pending) rather than trusting callers to sequence
  things correctly.
- **Long-only, cash account, no margin/leverage — enforced, not assumed.**
  `Portfolio` raises rather than silently allowing a BUY that would take
  cash negative or a SELL that would take a position negative (shorting).
  Modeling margin/shorting is a deliberate future scope decision, not a
  missing guardrail.
- **Strategies see one bar at a time, by construction.**
  `Strategy.on_market_event(event: MarketEvent) -> SignalEvent | None` has
  no back-channel to history or future bars — `SMACrossoverStrategy` keeps
  its own fixed-size rolling window per symbol instead of a stored
  DataFrame, the same constraint a live/streaming strategy would face. It
  also withholds a signal on the very bar its window first becomes full,
  since that bar only establishes a baseline state, not an observed
  crossover.
- **Survivorship bias, demonstrated with a real result, not just a
  toggle.** `compare_survivorship.py` runs the same strategy across a
  curated pool of real 2015 S&P 500 constituents, once restricted to the
  accurate point-in-time membership and once to today's roster. On the
  real data: the biased (today's-roster) universe shows +88.86% total
  return vs. the accurate universe's +52.71% — excluding the names that
  later left the index measurably inflates the biased backtest's apparent
  performance.
- **A second, conceptually different strategy, not just different
  parameters.** `RSIMeanReversionStrategy` bets the opposite direction from
  SMA crossover — reversion instead of trend continuation — while reusing
  the identical `Strategy` interface and rolling-window-state discipline.
  `compare_survivorship.py`'s comparison logic was parameterized (a
  `strategy_factory` callable) rather than duplicated so both strategies
  share one aggregation/forward-fill implementation; `compare_survivorship_rsi.py`
  just supplies a different factory. On the same 20-name pool: the biased
  universe still outperforms the accurate one (+124.03% vs. +77.14% total
  return), confirming the survivorship-bias effect isn't an artifact of one
  particular strategy.
- **A data bug found via this comparison, investigated, and fixed — not
  silently corrected.** One pool member (`COL`) turned out to not
  actually be Rockwell Collins in `yfinance`'s data (a $0.05–$0.75
  penny-stock series, versus the real company's $80–140 range before its
  2018 acquisition) — likely a reused/reassigned ticker symbol. Replaced
  with `M` (Macy's, verified against real price history) and added a
  cheap sanity check to `data/loader.py` that warns when a fetched
  symbol's price never exceeds $1 across the requested range. The figures
  above are the corrected numbers; full root-cause writeup and the
  before/after comparison are in `CLAUDE.md`.
- **Round-trip win/loss is real FIFO lot-matching, not a fill-count
  heuristic.** `analytics/performance.py`'s `extract_round_trips()` matches
  each SELL against the oldest open BUY lot(s) per symbol, splitting across
  partial fills — replacing an earlier naive `len(trade_log) // 2` estimate
  that didn't actually look at trade contents.

## ⚠️ Data sources and known limitations

**Market data** (`data/loader.py`): fetched via `yfinance` (free, delayed,
best-effort — not a substitute for a licensed market data feed), cached
locally as parquet per symbol under `data/cache/` (gitignored — regenerated
on demand, not vendored). Prices are split/dividend-adjusted
(`auto_adjust=True`).