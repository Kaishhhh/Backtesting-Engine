"""Point-in-time S&P 500 universe membership.

============================================================================
KNOWN LIMITATION -- READ BEFORE TRUSTING THIS BEYOND A PORTFOLIO BACKTEST
============================================================================
Point-in-time index membership is vendored from a free, community-maintained
dataset (github.com/fja05680/sp500, MIT licensed), not a paid/institutional
source (Norgate, Sharadar, CRSP). It is good enough to demonstrate correct
handling of survivorship bias, but it is NOT institutionally audited, its
first ~5 years (1996-2001) likely undercount true membership, and its
post-2019 portion inherits Wikipedia's explicitly-labeled-incomplete
"Selected changes" table. Full writeup: data/reference/SOURCES.md and
CLAUDE.md's Design Decisions section.
============================================================================

Two query modes are exposed, matching CLAUDE.md's requirement that the
project be able to demonstrate survivorship bias side-by-side:

- ``survivorship_biased=False`` (accurate): returns the S&P 500 membership
  as it actually stood on the requested date -- e.g. a symbol added to the
  index in 2020 will not appear in a 2015 query.
- ``survivorship_biased=True`` (biased, the common mistake): always returns
  today's constituent list regardless of the requested date, reproducing the
  classic error of backtesting 2015 with 2024's S&P 500 roster.
"""

from __future__ import annotations

import warnings
from datetime import date, datetime
from pathlib import Path

import pandas as pd

DEFAULT_MEMBERSHIP_PATH = Path(__file__).parent / "reference" / "sp500_membership.parquet"

DateLike = date | datetime | str


class UniverseError(RuntimeError):
    """Raised when a universe membership query cannot be answered."""


def _to_date(value: DateLike) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def load_membership_table(path: Path = DEFAULT_MEMBERSHIP_PATH) -> pd.DataFrame:
    """Load the vendored snapshot table: columns `date` (date) and `tickers`
    (list[str]), sorted ascending by date."""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.date
    return df.sort_values("date").reset_index(drop=True)


def point_in_time_members(membership: pd.DataFrame, as_of: DateLike) -> frozenset[str]:
    """Return the S&P 500 membership as it actually stood on `as_of`.

    The vendored table is one row per snapshot date (not one row per
    calendar day), so this is an "as-of" lookup: the most recent snapshot
    dated on or before `as_of`. Dates before the dataset's coverage raise;
    dates after the last snapshot fall back to the latest known snapshot
    with a warning, since the vendored data lags real time.
    """
    if membership.empty:
        raise UniverseError("membership table is empty")

    as_of = _to_date(as_of)
    dates = membership["date"]
    earliest, latest = dates.iloc[0], dates.iloc[-1]

    if as_of < earliest:
        raise UniverseError(
            f"as_of={as_of} is before the dataset's coverage starts "
            f"({earliest}); no point-in-time answer is available for this "
            f"date. See data/reference/SOURCES.md."
        )
    if as_of > latest:
        warnings.warn(
            f"as_of={as_of} is after the last available snapshot ({latest}); "
            f"falling back to the latest known membership as an "
            f"approximation. The vendored S&P 500 membership dataset needs "
            f"a refresh -- see data/reference/build_sp500_membership.py.",
            UserWarning,
            stacklevel=2,
        )
        return frozenset(membership.iloc[-1]["tickers"])

    idx = dates.searchsorted(as_of, side="right") - 1
    return frozenset(membership.iloc[idx]["tickers"])


def current_members(membership: pd.DataFrame) -> frozenset[str]:
    """The latest known snapshot -- used as 'today's' universe for the
    deliberately survivorship-biased query path."""
    if membership.empty:
        raise UniverseError("membership table is empty")
    return frozenset(membership.iloc[-1]["tickers"])


def get_universe(
    as_of: DateLike,
    survivorship_biased: bool = False,
    membership: pd.DataFrame | None = None,
) -> frozenset[str]:
    """Return the set of symbols in the S&P 500 universe for `as_of`.

    survivorship_biased=False (default): point-in-time accurate -- the
    membership as it actually stood on `as_of`.
    survivorship_biased=True: ignores `as_of` and always returns today's
    constituent list, deliberately reproducing survivorship bias so the
    analytics module can show both results side-by-side.
    """
    if membership is None:
        membership = load_membership_table()

    if survivorship_biased:
        return current_members(membership)
    return point_in_time_members(membership, as_of)
