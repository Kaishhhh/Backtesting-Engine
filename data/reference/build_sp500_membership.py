"""One-time build script: vendor a point-in-time S&P 500 membership dataset.

This is NOT run automatically by the engine or the test suite -- it documents,
reproducibly, how data/reference/sp500_membership.parquet was produced, and
can be re-run by hand to refresh the vendored snapshot.

Source: https://github.com/fja05680/sp500 (MIT License), file
"S&P 500 Historical Components & Changes (Updated).csv". Per that repo's
README: the 1996-2019 portion originates from a Norgate Data extract shipped
with Andreas Clenow's book "Trading Evolved"; the post-2019 portion is
maintained by hand-reconciling against Wikipedia's "List of S&P 500
companies" changes table every couple of months. See
data/reference/SOURCES.md and the module docstring of data/universe.py for
the full accuracy discussion -- this is a free, actively-maintained, but
NOT institutionally-audited source. Do not treat it as ground truth for
anything beyond a portfolio/research backtest.

Each row of the source CSV is a snapshot date and the full comma-separated
list of tickers that were S&P 500 members as of that date. We store that
shape as-is (one row per snapshot date, tickers as a list column) rather than
exploding it into one row per (date, symbol): the source is not one row per
calendar day, it's one row per point where the maintainer captured or updated
the list, so point-in-time lookups are an "as-of" search (latest snapshot
date <= query date) rather than an exact-date join. Storing it exploded would
inflate ~2.7k rows into >1M rows of pure redundancy for no benefit.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path
from urllib.request import urlopen

import pandas as pd

SOURCE_URL = (
    "https://raw.githubusercontent.com/fja05680/sp500/master/"
    "S%26P%20500%20Historical%20Components%20%26%20Changes%20(Updated).csv"
)
OUTPUT_PATH = Path(__file__).parent / "sp500_membership.parquet"


def fetch_raw_csv(dest: Path) -> None:
    with urlopen(SOURCE_URL, timeout=30) as response:  # noqa: S310 -- fixed, known URL
        dest.write_bytes(response.read())


def parse_membership_table(csv_path: Path) -> pd.DataFrame:
    records: list[tuple[date, list[str]]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ["date", "tickers"], f"unexpected header: {header}"
        for row_date, tickers_field in reader:
            tickers = sorted(t for t in tickers_field.split(",") if t)
            records.append((date.fromisoformat(row_date), tickers))

    df = pd.DataFrame(records, columns=["date", "tickers"])
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.sort_values("date").reset_index(drop=True)
    return df


def main() -> None:
    raw_csv_path = Path(__file__).parent / "_sp500_raw_source.csv"
    print(f"Fetching {SOURCE_URL} ...")
    fetch_raw_csv(raw_csv_path)

    df = parse_membership_table(raw_csv_path)
    print(f"Parsed {len(df)} snapshot rows: {df['date'].min()} .. {df['date'].max()}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")

    raw_csv_path.unlink()  # don't keep the multi-MB raw copy around


if __name__ == "__main__":
    main()
