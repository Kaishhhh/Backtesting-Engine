"""Tests for data/loader.py.

fetch_ohlcv (the yfinance-calling boundary) is monkeypatched throughout so
these tests never touch the network -- loader.py factors that call out into
its own function specifically so it can be swapped out like this.
"""

from datetime import date, datetime, time

import pandas as pd
import pytest

from data import loader
from data.loader import SCHEMA_COLUMNS, LoaderError, load_market_events, load_ohlcv


def make_ohlcv(symbol: str, start: date, end: date) -> pd.DataFrame:
    dates = list(pd.date_range(start, end, freq="D").date)
    n = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + i for i in range(n)],
            "high": [101.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [100.5 + i for i in range(n)],
            "volume": [1_000 + i for i in range(n)],
            "symbol": symbol,
        }
    )


class FetchSpy:
    """Stands in for fetch_ohlcv: records every call, serves synthetic data."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, date, date]] = []

    def __call__(self, symbol: str, start: date, end: date) -> pd.DataFrame:
        self.calls.append((symbol, start, end))
        return make_ohlcv(symbol, start, end)


@pytest.fixture
def spy(monkeypatch):
    fetch_spy = FetchSpy()
    monkeypatch.setattr(loader, "fetch_ohlcv", fetch_spy)
    return fetch_spy


class TestCacheBehavior:
    def test_cache_miss_fetches_and_writes_parquet(self, tmp_path, spy):
        result = load_ohlcv("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        assert len(spy.calls) == 1
        assert (tmp_path / "AAPL.parquet").exists()
        assert len(result) == 5

    def test_cache_hit_does_not_refetch(self, tmp_path, spy):
        load_ohlcv("AAPL", "2023-01-01", "2023-01-10", cache_dir=tmp_path)
        assert len(spy.calls) == 1

        result = load_ohlcv("AAPL", "2023-01-03", "2023-01-06", cache_dir=tmp_path)

        assert len(spy.calls) == 1  # served entirely from cache
        assert len(result) == 4
        assert result["date"].min() == date(2023, 1, 3)
        assert result["date"].max() == date(2023, 1, 6)

    def test_request_outside_cached_range_triggers_refetch(self, tmp_path, spy):
        load_ohlcv("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        assert len(spy.calls) == 1

        result = load_ohlcv("AAPL", "2023-01-01", "2023-01-10", cache_dir=tmp_path)

        assert len(spy.calls) == 2
        assert len(result) == 10

    def test_force_refresh_bypasses_cache(self, tmp_path, spy):
        load_ohlcv("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        assert len(spy.calls) == 1

        load_ohlcv(
            "AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path, force_refresh=True
        )
        assert len(spy.calls) == 2

    def test_different_symbols_get_separate_cache_files(self, tmp_path, spy):
        load_ohlcv("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        load_ohlcv("MSFT", "2023-01-01", "2023-01-05", cache_dir=tmp_path)

        assert (tmp_path / "AAPL.parquet").exists()
        assert (tmp_path / "MSFT.parquet").exists()
        assert len(spy.calls) == 2

    def test_symbol_is_uppercased_for_cache_key(self, tmp_path, spy):
        load_ohlcv("aapl", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        assert (tmp_path / "AAPL.parquet").exists()


class TestSchema:
    def test_returned_columns_match_schema(self, tmp_path, spy):
        result = load_ohlcv("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        assert list(result.columns) == SCHEMA_COLUMNS

    def test_symbol_column_populated(self, tmp_path, spy):
        result = load_ohlcv("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        assert (result["symbol"] == "AAPL").all()

    def test_start_after_end_raises_before_any_fetch(self, tmp_path, spy):
        with pytest.raises(LoaderError):
            load_ohlcv("AAPL", "2023-01-10", "2023-01-01", cache_dir=tmp_path)
        assert len(spy.calls) == 0


class TestFetchOhlcvNormalization:
    """Exercises the real fetch_ohlcv, with yf.download itself monkeypatched
    to return shapes matching real yfinance output (MultiIndex columns)."""

    def test_flattens_multiindex_and_normalizes_schema(self, monkeypatch):
        idx = pd.date_range("2023-01-03", "2023-01-05", freq="D")
        idx.name = "Date"
        raw = pd.DataFrame(
            {
                ("Open", "AAPL"): [1.0, 2.0, 3.0],
                ("High", "AAPL"): [1.5, 2.5, 3.5],
                ("Low", "AAPL"): [0.5, 1.5, 2.5],
                ("Close", "AAPL"): [1.2, 2.2, 3.2],
                ("Volume", "AAPL"): [100, 200, 300],
            },
            index=idx,
        )
        raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])

        monkeypatch.setattr(loader.yf, "download", lambda *a, **k: raw)

        result = loader.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 5))

        assert list(result.columns) == SCHEMA_COLUMNS
        assert (result["symbol"] == "AAPL").all()
        assert result["date"].tolist() == [
            date(2023, 1, 3),
            date(2023, 1, 4),
            date(2023, 1, 5),
        ]
        assert result["close"].tolist() == [1.2, 2.2, 3.2]

    def test_empty_response_raises(self, monkeypatch):
        monkeypatch.setattr(loader.yf, "download", lambda *a, **k: pd.DataFrame())
        with pytest.raises(LoaderError):
            loader.fetch_ohlcv("NOPE", date(2023, 1, 3), date(2023, 1, 5))


class TestAdjustmentNoiseClamp:
    """auto_adjust occasionally leaves close/open a hair outside [low, high]
    -- observed for real on ADP, CAG, UNM, ABT, and AMGN while building the
    Step 5 multi-symbol backtest (see data/loader.py's docstring note above
    _OHLC_NOISE_TOLERANCE). This exercises fetch_ohlcv's real clamp against
    a contrived MultiIndex response shaped like actual yfinance output."""

    @staticmethod
    def _download(close_second_day: float):
        idx = pd.date_range("2023-01-03", "2023-01-04", freq="D")
        idx.name = "Date"
        raw = pd.DataFrame(
            {
                ("Open", "AAPL"): [1.0, 2.0],
                ("High", "AAPL"): [1.5, 2.5],
                ("Low", "AAPL"): [0.5, 1.5],
                ("Close", "AAPL"): [1.2, close_second_day],
                ("Volume", "AAPL"): [100, 200],
            },
            index=idx,
        )
        raw.columns = pd.MultiIndex.from_tuples(raw.columns, names=["Price", "Ticker"])
        return lambda *a, **k: raw

    def test_close_a_hair_above_high_is_clamped_to_high(self, monkeypatch):
        monkeypatch.setattr(loader.yf, "download", self._download(close_second_day=2.5000000001))
        result = loader.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 4))
        assert result["close"].iloc[1] == 2.5

    def test_close_a_hair_below_low_is_clamped_to_low(self, monkeypatch):
        monkeypatch.setattr(loader.yf, "download", self._download(close_second_day=1.4999999999))
        result = loader.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 4))
        assert result["close"].iloc[1] == 1.5

    def test_violation_beyond_tolerance_is_left_untouched(self, monkeypatch):
        # A whole dollar outside [low, high] is a real data problem, not
        # adjustment noise -- must NOT be silently clamped away.
        monkeypatch.setattr(loader.yf, "download", self._download(close_second_day=10.0))
        result = loader.fetch_ohlcv("AAPL", date(2023, 1, 3), date(2023, 1, 4))
        assert result["close"].iloc[1] == 10.0


class TestLoadMarketEvents:
    def test_converts_every_row_to_a_market_event(self, tmp_path, spy):
        events = load_market_events("AAPL", "2023-01-01", "2023-01-03", cache_dir=tmp_path)

        assert len(events) == 3
        first = events[0]
        assert first.symbol == "AAPL"
        assert first.timestamp == datetime(2023, 1, 1, 9, 30)  # default bar_time
        assert first.open == pytest.approx(100.0)
        assert first.high == pytest.approx(101.0)
        assert first.low == pytest.approx(99.0)
        assert first.close == pytest.approx(100.5)
        assert first.volume == 1_000
        assert isinstance(first.volume, int)

    def test_events_are_in_chronological_order(self, tmp_path, spy):
        events = load_market_events("AAPL", "2023-01-01", "2023-01-05", cache_dir=tmp_path)
        timestamps = [e.timestamp for e in events]
        assert timestamps == sorted(timestamps)

    def test_custom_bar_time_is_honored(self, tmp_path, spy):
        events = load_market_events(
            "AAPL", "2023-01-01", "2023-01-01", bar_time=time(16, 0), cache_dir=tmp_path
        )
        assert events[0].timestamp == datetime(2023, 1, 1, 16, 0)
