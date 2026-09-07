import pandas as pd
import quantstats as qs

# Load dataset
data = pd.read_parquet("data/clean/full_dataset.parquet")

# Run Buy & Hold
from buy_and_hold_baseline import run_buy_and_hold

history = run_buy_and_hold(data)

# Get portfolio returns
returns = history["portfolio_return"]

# QuantStats analysis
print("Cumulative Return:")
print(qs.stats.comp(returns))

print("\nAnnualized Return:")
print(qs.stats.cagr(returns))

print("\nSharpe Ratio:")
print(qs.stats.sharpe(returns))

print("\nMaximum Drawdown:")
print(qs.stats.max_drawdown(returns))

print("\nVolatility:")
print(qs.stats.volatility(returns))