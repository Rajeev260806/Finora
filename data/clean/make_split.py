import pandas as pd

df = pd.read_parquet("data\\clean\\full_dataset.parquet")
df = df.sort_index()
dates = df.index.get_level_values("Date").unique().sort_values()
split_index = int(len(dates) * 0.80)

train_dates = dates[:split_index]
test_dates = dates[split_index:]

train_df = df.loc[df.index.get_level_values("Date").isin(train_dates)]
test_df = df.loc[df.index.get_level_values("Date").isin(test_dates)]

train_df.to_parquet("data\\clean\\train.parquet", index=True)
test_df.to_parquet("data\\clean\\test.parquet", index=True)

print("Train:")
print("Rows:", len(train_df))
print("Date range:", train_dates.min(), "to", train_dates.max())
print()

print("Test:")
print("Rows:", len(test_df))
print("Date range:", test_dates.min(), "to", test_dates.max())
print()

print("Train index:", train_df.index.names)
print("Test index:", test_df.index.names)