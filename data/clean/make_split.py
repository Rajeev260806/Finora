import os
import pandas as pd

# Path relative to project root or local directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
full_path = os.path.join(BASE_DIR, "full_dataset.parquet")

df = pd.read_parquet(full_path)
df = df.sort_index()
dates = df.index.get_level_values("Date")

# Strict splits as per PROJECT_ACTION_PLAN.md:
# Train: 2015-01-02 to 2021-12-31
# Val:   2022-01-01 to 2022-12-28 (tuning only)
# Test:  2022-12-29 to 2024-12-31 (final evaluation only)
train_mask = (dates >= "2015-01-02") & (dates <= "2021-12-31")
val_mask = (dates >= "2022-01-01") & (dates <= "2022-12-28")
test_mask = (dates >= "2022-12-29") & (dates <= "2024-12-31")

train_df = df.loc[train_mask]
val_df = df.loc[val_mask]
test_df = df.loc[test_mask]

train_path = os.path.join(BASE_DIR, "train.parquet")
val_path = os.path.join(BASE_DIR, "val.parquet")
test_path = os.path.join(BASE_DIR, "test.parquet")

train_df.to_parquet(train_path, index=True)
val_df.to_parquet(val_path, index=True)
test_df.to_parquet(test_path, index=True)

train_dates = train_df.index.get_level_values("Date").unique().sort_values()
val_dates = val_df.index.get_level_values("Date").unique().sort_values()
test_dates = test_df.index.get_level_values("Date").unique().sort_values()

print("Strict Split Summary:")
print("=" * 60)
print(f"Train: {train_dates.min().strftime('%Y-%m-%d')} to {train_dates.max().strftime('%Y-%m-%d')} | {len(train_dates)} days | {len(train_df)} rows")
print(f"Val:   {val_dates.min().strftime('%Y-%m-%d')} to {val_dates.max().strftime('%Y-%m-%d')} | {len(val_dates)} days | {len(val_df)} rows")
print(f"Test:  {test_dates.min().strftime('%Y-%m-%d')} to {test_dates.max().strftime('%Y-%m-%d')} | {len(test_dates)} days | {len(test_df)} rows")
print("=" * 60)
print("Files generated successfully:")
print(" - train.parquet")
print(" - val.parquet")
print(" - test.parquet")