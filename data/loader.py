"""OHLCV data loader: fetch via yfinance, cache-first as parquet per symbol.

Design notes (see CLAUDE.md's Design Decisions section for the short version):

- ``auto_adjust=True`` is passed to yfinance, so Open/High/Low/Close are
  split- and dividend-adjusted. This is deliberate, not a default left
  untouched: an *unadjusted* close series shows a fake ~-75% "crash" at every
  4:1 stock split (e.g. AAPL, Aug 2020), which would corrupt any indicator or
  P&L computation that touches price levels. Adjusted prices are the correct
  choice for a backtester.

- The cache is coarse-grained, not gap-filling. Each symbol's cache file
  tracks whatever date range has already been fetched. A request whose
  [start, end] falls entirely inside that cached range is served from disk
  with zero network calls. A request that extends outside the cached range
  triggers a full re-fetch of the *entire* requested range (not just the
  missing delta), which is then merged into the cache. This is simpler and
  more robust than incremental gap-filling -- which has to reason about
  multiple disjoint gaps, weekends, holidays, and partial-day data -- at the
  cost of occasionally re-fetching data we already had. Acceptable tradeoff
  at this project's data volumes.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

from engine.events import MarketEvent

DEFAULT_CACHE_DIR = Path(__file__).parent / "cache"

# Daily bars carry no intraday timestamp; 9:30 stands in for "market open",
# matching the convention used throughout tests/ and the runner scripts.
DEFAULT_BAR_TIME = time(9, 30)

SCHEMA_COLUMNS = ["date", "open", "high", "low", "close", "volume", "symbol"]

DateLike = date | datetime | str


class LoaderError(RuntimeError):
    """Raised when historical OHLCV data cannot be loaded or is malformed."""


def _to_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _cache_path(symbol: str, cache_dir: Path) -> Path:
    return cache_dir / f"{symbol}.parquet"


def _read_cache(symbol: str, cache_dir: Path) -> pd.DataFrame | None:
    path = _cache_path(symbol, cache_dir)
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def _write_cache(df: pd.DataFrame, symbol: str, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    df.sort_values("date").reset_index(drop=True).to_parquet(
        _cache_path(symbol, cache_dir), index=False
    )


def _validate_schema(df: pd.DataFrame) -> None:
    if list(df.columns) != SCHEMA_COLUMNS:
        raise LoaderError(
            f"OHLCV frame has unexpected schema {list(df.columns)}, "
            f"expected {SCHEMA_COLUMNS}"
        )


# yfinance's auto_adjust split/dividend adjustment occasionally leaves open
# or close a hair outside [low, high] -- observed directly while building
# Step 5's multi-symbol backtest: e.g. ADP on 2018-10-30 came back with
# close=115.84330749511719 against high=115.84330749511717 (a ~2e-14 ULP
# gap), and other symbols/dates were off by as much as ~7.6e-6 (still a
# fraction of a cent). This is floating-point noise in the adjustment
# arithmetic, not a real OHLC inconsistency, and it was even observed to
# vary between separate fetches of the identical [symbol, date] pair.
# MarketEvent.__post_init__ is deliberately strict and rejects it outright
# (see CLAUDE.md's no-look-ahead/fail-loud philosophy) -- correctly, since
# it can't tell noise from a real data problem on its own. This clamps
# values back onto the boundary only when the gap is well within this
# tolerance; anything larger is left untouched and will still raise, since
# that's a real problem worth seeing, not noise to hide.
_OHLC_NOISE_TOLERANCE = 0.01  # dollars; ~1000x the largest gap seen so far


def _clamp_adjustment_noise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    tol = _OHLC_NOISE_TOLERANCE
    for edge_col, bound_col, above in (
        ("close", "high", True), ("close", "low", False),
        ("open", "high", True), ("open", "low", False),
    ):
        if above:
            mask = (df[edge_col] > df[bound_col]) & (df[edge_col] - df[bound_col] <= tol)
        else:
            mask = (df[edge_col] < df[bound_col]) & (df[bound_col] - df[edge_col] <= tol)
        df.loc[mask, edge_col] = df.loc[mask, bound_col]
    return df


def fetch_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    """Fetch raw OHLCV from yfinance and normalize it to the engine's schema.

    Kept as its own function (rather than inlined into load_ohlcv) so tests
    can monkeypatch it and exercise the caching logic without any real
    network call.
    """
    raw = yf.download(
        symbol,
        start=start,
        end=end + timedelta(days=1),  # yfinance's end is exclusive
        auto_adjust=True,
        progress=False,
    )
    if raw.empty:
        raise LoaderError(f"yfinance returned no data for {symbol} [{start}, {end}]")

    if isinstance(raw.columns, pd.MultiIndex):
        # Even single-symbol requests can come back as a (Price, Ticker)
        # MultiIndex; the ticker level is constant here, so drop it.
        raw.columns = raw.columns.get_level_values(0)

    df = raw.reset_index()
    df.columns = [str(c).lower() for c in df.columns]
    df = df.rename(columns={"datetime": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = symbol
    df = df[SCHEMA_COLUMNS]
    df = _clamp_adjustment_noise(df)
    return df.sort_values("date").reset_index(drop=True)


def load_ohlcv(
    symbol: str,
    start: DateLike,
    end: DateLike,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Load OHLCV for `symbol` over [start, end] (inclusive), cache-first.

    Returns a DataFrame with columns [date, open, high, low, close, volume,
    symbol]. If the on-disk cache for `symbol` already covers [start, end],
    this makes no network call. Otherwise the full [start, end] range is
    fetched, merged into the cache (deduped by date, existing rows for a
    date are overwritten by the fresh fetch), and persisted before returning.
    """
    symbol = symbol.upper()
    start = _to_date(start)
    end = _to_date(end)
    if start > end:
        raise LoaderError(f"start ({start}) is after end ({end})")

    cached = None if force_refresh else _read_cache(symbol, cache_dir)

    if cached is not None and not cached.empty:
        cached_start, cached_end = cached["date"].iloc[0], cached["date"].iloc[-1]
        if start >= cached_start and end <= cached_end:
            mask = (cached["date"] >= start) & (cached["date"] <= end)
            result = cached.loc[mask].reset_index(drop=True)
            _validate_schema(result)
            return result

    fresh = fetch_ohlcv(symbol, start, end)

    if cached is not None and not cached.empty:
        merged = pd.concat([cached, fresh], ignore_index=True)
        merged = merged.drop_duplicates(subset="date", keep="last")
        merged = merged.sort_values("date").reset_index(drop=True)
    else:
        merged = fresh

    _write_cache(merged, symbol, cache_dir)

    mask = (merged["date"] >= start) & (merged["date"] <= end)
    result = merged.loc[mask].reset_index(drop=True)
    _validate_schema(result)
    return result


def load_market_events(
    symbol: str,
    start: DateLike,
    end: DateLike,
    bar_time: time = DEFAULT_BAR_TIME,
    cache_dir: Path = DEFAULT_CACHE_DIR,
) -> list[MarketEvent]:
    """load_ohlcv, converted to MarketEvents ready for the event queue.

    Kept alongside load_ohlcv (rather than in a runner script) so every
    caller that needs an OHLCV DataFrame turned into a MarketEvent stream
    -- main.py, compare_survivorship.py, or any future one -- shares this
    one conversion instead of reimplementing the row-by-row loop.
    """
    df = load_ohlcv(symbol, start, end, cache_dir=cache_dir)
    return [
        MarketEvent(
            timestamp=datetime.combine(row.date, bar_time),
            symbol=row.symbol,
            open=float(row.open),
            high=float(row.high),
            low=float(row.low),
            close=float(row.close),
            volume=int(row.volume),
        )
        for row in df.itertuples(index=False)
    ]
