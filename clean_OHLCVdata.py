import numpy as np
import pandas as pd
import os

raw_dir = "data/raw"
clean_dir = "data/clean"

os.makedirs(clean_dir, exist_ok=True)

stock_files = os.listdir(raw_dir)

stock_dataframes = {}

for ticker in stock_files:
    filepath = "data/raw/"+ticker
    key = ticker[:-4]
    df = pd.read_csv(filepath, parse_dates=["Date"])
    df = df.set_index("Date")
    duplicate_count = df.index.duplicated().sum()
    if duplicate_count > 0:
        print(ticker, "has", duplicate_count, "duplicate dates")
        df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    stock_dataframes[key] = df

union_dates = stock_dataframes["AAPL"].index

for dfs in stock_dataframes:
    if dfs!="AAPL":
        union_dates = union_dates.union(stock_dataframes[dfs].index)

for dfs in stock_dataframes:
    stock_dataframes[dfs] = stock_dataframes[dfs].reindex(union_dates)

remove_stocks = []
for ticker in stock_dataframes:
    df = stock_dataframes[ticker]
    missing = df.isna().sum().sum()
    rows,cols = df.shape
    total = rows*cols
    miss_perc = missing/total * 100
    if miss_perc>5:
        remove_stocks.append(ticker)

for ticker in remove_stocks:
    print("Removing", ticker, "because missing data > 5%")
    del stock_dataframes[ticker]

#check for negative price
for ticker in stock_dataframes:
    df = stock_dataframes[ticker]
    neg_price = (df[["Open", "High", "Low", "Close"]]<0).sum().sum()
    print(ticker,"negative prices:",neg_price)

#check for negative volume
for ticker in stock_dataframes:
    df = stock_dataframes[ticker]
    neg_volume = (df["Volume"]<0).sum()
    print(ticker,"negative volume:",neg_volume)

#================================================
#Calculate Returns and Annual Volatility(after cleaning is done)
#================================================
for ticker in stock_dataframes:
    df = stock_dataframes[ticker]
    df["Returns Daily"] = df["Close"].pct_change()
    df["Volatility 20D"] = df["Returns Daily"].rolling(20).std()
    df["Volume Normalized"] = df["Volume"]/(df["Volume"].rolling(20).mean())
    df["Volatility Annual"] = df["Volatility 20D"]*(252**0.5)
    df["Log Return"] = np.log(df["Close"] / df["Close"].shift(1))

stock_dataframes = pd.concat(stock_dataframes,keys=stock_dataframes.keys())
stock_dataframes.index.names = ["Ticker","Date"]
stock_dataframes = stock_dataframes.swaplevel("Ticker","Date")
stock_dataframes = stock_dataframes.sort_index()
print(stock_dataframes)

# 1. Number of unique tickers
print("Unique tickers:", stock_dataframes.index.get_level_values("Ticker").nunique())

# 2. Number of rows for each ticker
print("\nRows per ticker:")
print(stock_dataframes.index.get_level_values("Ticker").value_counts().sort_index())

# 3. Number of duplicate (Date, Ticker) combinations
print("\nDuplicate index entries:",
      stock_dataframes.index.duplicated().sum())

# 4. Date range
dates = stock_dataframes.index.get_level_values("Date")

print("\nEarliest date:", dates.min())
print("Latest date:", dates.max())

# 5. Number of unique dates
print("Unique dates:", dates.nunique())


clean_file = os.path.join(clean_dir,"full_dataset.parquet")
stock_dataframes.to_parquet(clean_file)
print("\nClean dataset saved to:", clean_file)

loaded_df = pd.read_parquet(clean_file)

print("\nReloaded dataset shape:", loaded_df.shape)

print(
    "Reloaded index levels:",
    loaded_df.index.names
)

print(
    "Reloaded unique tickers:",
    loaded_df.index
    .get_level_values("Ticker")
    .nunique()
)

print(
    "Reloaded duplicate index entries:",
    loaded_df.index.duplicated().sum()
)