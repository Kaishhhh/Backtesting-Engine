# data/reference/ — vendored reference datasets

## sp500_membership.parquet

**What it is:** A point-in-time snapshot table of S&P 500 index membership,
one row per date the source maintainer captured or updated the list, columns
`date` (the snapshot date) and `tickers` (the full list of member tickers as
of that date).

**Source:** [github.com/fja05680/sp500](https://github.com/fja05680/sp500),
file `S&P 500 Historical Components & Changes (Updated).csv`. MIT License
(© 2019-2020 Farrell J. Aultman). Fetched and converted to parquet on
2026-08-07 by `data/reference/build_sp500_membership.py` — re-run that script
to refresh.

**Coverage:** 1996-01-02 through 2026-06-30 (~2,718 snapshot rows).

**Provenance per the source repo's own README:**
- 1996–2019: originates from a Norgate Data extract shipped with Andreas
  Clenow's book *Trading Evolved*.
- 2019–present: hand-reconciled every couple of months by the maintainer
  against Wikipedia's ["List of S&P 500
  companies"](https://en.wikipedia.org/wiki/List_of_S%26P_500_companies)
  changes table.

## ⚠️ Known limitations — read before trusting this for anything beyond a portfolio backtest

1. **Not institutionally audited.** This is a free, community-maintained,
   best-effort dataset — not CRSP, not S&P's own index files, not a paid
   vendor (Norgate, Sharadar). We chose it over those because it's free and
   good enough for a research/portfolio backtest; see `data/universe.py` for
   the fuller tradeoff writeup and `CLAUDE.md`'s Design Decisions section.
2. **First ~5 years (1996–2001) likely undercount membership.** The source
   maintainer notes the earliest rows have as few as 487 tickers (S&P 500
   membership is normally ~500-505); they recommend treating pre-2001 data
   with caution or excluding it.
3. **Wikipedia's underlying changes table is explicitly "Selected changes"
   — not exhaustive.** The 2019+ portion of this dataset inherits that gap;
   the maintainer supplements it with manual research but does not claim
   completeness.
4. **Snapshot dates are not daily.** A query for a date between two snapshot
   rows returns the most recent prior snapshot's membership (an "as-of"
   lookup) — see `data/universe.py::point_in_time_members`.
5. **The dataset lags real time.** As of this fetch, the last snapshot is
   2026-06-30. Queries for dates after the last snapshot fall back to the
   latest known snapshot and emit a `UserWarning`.

If accuracy at this level is insufficient for a real (non-portfolio) use
case, the next step up is a paid vendor (Norgate Data, Sharadar via Nasdaq
Data Link, or CRSP for academic use) — deliberately not adopted here without
sign-off, since it changes the cost profile of the whole project.
