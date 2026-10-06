import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clean",
    "full_dataset.parquet"
)

data = pd.read_parquet(DATA_PATH)
returns = data["Returns Daily"].unstack(level="Ticker")
returns = returns.fillna(0.0)
print(f"Returns shape: {returns.shape}")
print(f"Date range: {returns.index.min()} to {returns.index.max()}")