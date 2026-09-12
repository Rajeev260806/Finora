"""
Step 1 — Inspect Person A's cleaned dataset.

Run this FIRST, before touching PortfolioEnv. It tells you exactly
what you have to work with: index structure, date range, tickers,
available columns, and missing data — so the observation-space design
isn't guesswork.

Usage:
    python inspect_dataset.py data/clean/full_dataset.parquet
"""

import sys
import pandas as pd


def inspect(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    print(f"Loaded: {path}")
    print(f"Shape: {df.shape}")
    print(f"Index levels: {df.index.names}")

    if isinstance(df.index, pd.MultiIndex):
        date_level = df.index.names[0]
        ticker_level = df.index.names[-1]
        dates = df.index.get_level_values(date_level)
        tickers = df.index.get_level_values(ticker_level).unique()
        print(f"Date range: {dates.min()} to {dates.max()}  ({dates.nunique()} unique dates)")
        print(f"Tickers ({len(tickers)}): {sorted(tickers)}")
    else:
        print("Index is not a MultiIndex — check whether (Date, Ticker) was expected.")

    print(f"\nColumns: {df.columns.tolist()}")
    print("\nMissing values per column (fraction):")
    print(df.isnull().mean().sort_values(ascending=False))

    print("\nSample rows:")
    print(df.head(10))

    return df


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/clean/full_dataset.parquet"
    inspect(path)