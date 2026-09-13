import pandas as pd
data = pd.read_parquet("data/clean/full_dataset.parquet")
returns = data["Returns Daily"].unstack(level="Ticker")
returns = returns.fillna(0.0)
print(f"Returns shape: {returns.shape}")
print(f"Date range: {returns.index.min()} to {returns.index.max()}")
