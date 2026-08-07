"""Tests for data/universe.py.

Two layers:
1. A small synthetic membership table -- deterministic, exercises every
   branch of the as-of lookup logic without depending on the real vendored
   dataset's contents.
2. The real vendored data/reference/sp500_membership.parquet, checked
   against one independently-verifiable fact (Tesla joined the S&P 500 on
   2020-12-21) to prove the point-in-time/survivorship-bias distinction
   isn't just a synthetic-data artifact.
"""

from datetime import date

import pandas as pd
import pytest

from data.universe import (
    UniverseError,
    current_members,
    get_universe,
    load_membership_table,
    point_in_time_members,
)


@pytest.fixture
def membership() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date(2020, 1, 1), date(2020, 6, 1), date(2021, 1, 1)],
            "tickers": [
                ["AAA", "BBB", "CCC"],
                ["AAA", "BBB", "DDD"],  # CCC removed, DDD added
                ["AAA", "DDD", "EEE"],  # BBB removed, EEE added
            ],
        }
    )


class TestPointInTimeMembers:
    def test_exact_snapshot_date(self, membership):
        assert point_in_time_members(membership, date(2020, 1, 1)) == frozenset(
            {"AAA", "BBB", "CCC"}
        )

    def test_between_snapshots_uses_latest_prior_snapshot(self, membership):
        # 2020-07-01 falls between the 2020-06-01 and 2021-01-01 snapshots.
        assert point_in_time_members(membership, date(2020, 7, 1)) == frozenset(
            {"AAA", "BBB", "DDD"}
        )

    def test_after_last_snapshot_falls_back_with_warning(self, membership):
        with pytest.warns(UserWarning, match="after the last available snapshot"):
            result = point_in_time_members(membership, date(2022, 1, 1))
        assert result == frozenset({"AAA", "DDD", "EEE"})

    def test_before_coverage_raises(self, membership):
        with pytest.raises(UniverseError, match="before the dataset's coverage"):
            point_in_time_members(membership, date(2019, 1, 1))

    def test_accepts_string_dates(self, membership):
        assert point_in_time_members(membership, "2020-01-01") == frozenset(
            {"AAA", "BBB", "CCC"}
        )


class TestCurrentMembers:
    def test_returns_latest_snapshot(self, membership):
        assert current_members(membership) == frozenset({"AAA", "DDD", "EEE"})


class TestGetUniverseToggle:
    def test_accurate_mode_is_point_in_time(self, membership):
        result = get_universe(date(2020, 1, 1), survivorship_biased=False, membership=membership)
        assert result == frozenset({"AAA", "BBB", "CCC"})

    def test_biased_mode_ignores_as_of_and_returns_current(self, membership):
        result = get_universe(date(2020, 1, 1), survivorship_biased=True, membership=membership)
        assert result == frozenset({"AAA", "DDD", "EEE"})

    def test_toggle_changes_result_for_the_same_date(self, membership):
        accurate = get_universe(date(2020, 1, 1), survivorship_biased=False, membership=membership)
        biased = get_universe(date(2020, 1, 1), survivorship_biased=True, membership=membership)

        assert accurate != biased
        # CCC genuinely existed on 2020-01-01 but was later removed from the
        # index -- the biased (today's-list) query silently drops it, which
        # is exactly the bias this toggle exists to demonstrate.
        assert "CCC" in accurate
        assert "CCC" not in biased
        # EEE wasn't added until 2021 -- the biased query incorrectly
        # includes it for a 2020-01-01 backtest.
        assert "EEE" not in accurate
        assert "EEE" in biased


@pytest.fixture(scope="module")
def real_membership() -> pd.DataFrame:
    return load_membership_table()


class TestRealVendoredDataset:
    """Sanity-checks the real data/reference/sp500_membership.parquet
    against an independently verifiable fact: Tesla was added to the S&P 500
    effective 2020-12-21."""

    def test_schema(self, real_membership):
        assert list(real_membership.columns) == ["date", "tickers"]
        assert len(real_membership) > 0

    def test_tsla_absent_before_it_was_added(self, real_membership):
        universe = point_in_time_members(real_membership, date(2019, 1, 1))
        assert "TSLA" not in universe

    def test_tsla_present_after_it_was_added(self, real_membership):
        universe = point_in_time_members(real_membership, date(2021, 6, 1))
        assert "TSLA" in universe

    def test_current_vs_historical_differ_for_tsla(self, real_membership):
        historical = get_universe(
            date(2019, 1, 1), survivorship_biased=False, membership=real_membership
        )
        biased = get_universe(
            date(2019, 1, 1), survivorship_biased=True, membership=real_membership
        )
        assert "TSLA" not in historical
        assert "TSLA" in biased
