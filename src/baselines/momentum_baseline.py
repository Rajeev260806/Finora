import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from src.environment.portfolioenv import PortfolioEnv


# -----------------------------
# Settings
# -----------------------------
LOOKBACK_DAYS = 252       # approximately 12 months
SKIP_DAYS = 21            # skip most recent month
TOP_N = 5                 # invest in top 5 stocks
INVESTED_WEIGHT = 0.80    # 80% invested
CASH_WEIGHT = 0.20        # 20% cash

TRANSACTION_COST_BPS = 10.0
INITIAL_VALUE = 1.0
SEED = 0


# -----------------------------
# Load data
# -----------------------------
if len(sys.argv) < 2:
    raise SystemExit("Usage: python momentum_baseline.py <path to full_dataset.parquet>")
DATA_PATH = sys.argv[1]
data = pd.read_parquet(DATA_PATH)

print("Dataset loaded.")
print("Number of rows:", len(data))


# -----------------------------
# Create environment
# -----------------------------
env = PortfolioEnv(
    data=data,
    window=LOOKBACK_DAYS + SKIP_DAYS,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    initial_cash=INITIAL_VALUE,
    seed=SEED
)

obs, info = env.reset(seed=SEED)

print("Number of stocks:", env.n_assets)
print("Stocks:", env.tickers)


# -----------------------------
# Calculate momentum score
# -----------------------------
def calculate_momentum(returns, current_idx):
    """
    Calculate 12-1 month momentum.

    Uses approximately the previous 12 months of returns,
    while skipping the most recent month.
    """

    start = current_idx - LOOKBACK_DAYS
    end = current_idx - SKIP_DAYS

    historical_returns = returns[start:end]

    momentum = np.prod(1.0 + historical_returns, axis=0) - 1.0

    return momentum


# -----------------------------
# Convert target weights to
# environment action
# -----------------------------
def weights_to_action(weights):
    """
    Convert portfolio weights into logits.

    PortfolioEnv converts logits into weights using softmax.
    """

    small_value = 1e-6

    safe_weights = np.maximum(weights, small_value)

    return np.log(safe_weights)


# -----------------------------
# Run momentum strategy
# -----------------------------
portfolio_values = []
dates = []
turnovers = []

previous_month = None

terminated = False

while not terminated:

    current_idx = env.current_idx
    current_date = env.dates[current_idx]

    current_month = current_date.to_period("M")

    # Keep previous portfolio unless it is
    # time for monthly rebalancing.
    if previous_month != current_month:

        momentum_scores = calculate_momentum(
            env.returns,
            current_idx
        )

        ranking = np.argsort(momentum_scores)[::-1]

        selected = ranking[:TOP_N]

        target_weights = np.zeros(env.n_assets + 1)

        # Equal weight among selected stocks
        stock_weight = INVESTED_WEIGHT / TOP_N

        target_weights[selected] = stock_weight

        # Last position is cash
        target_weights[-1] = CASH_WEIGHT

        action = weights_to_action(target_weights)

        previous_month = current_month

        print(
            f"{current_date.date()} | "
            f"Top stocks: "
            f"{[env.tickers[i] for i in selected]}"
        )

    else:
        # Keep the same portfolio between rebalancing dates
        action = weights_to_action(target_weights)

    obs, reward, terminated, truncated, info = env.step(action)

    portfolio_values.append(info["portfolio_value"])
    dates.append(info["date"])
    turnovers.append(info["turnover"])


# -----------------------------
# Create results DataFrame
# -----------------------------
results = pd.DataFrame({
    "portfolio_value": portfolio_values,
    "turnover": turnovers
}, index=pd.to_datetime(dates))


# -----------------------------
# Calculate returns
# -----------------------------
results["daily_return"] = results["portfolio_value"].pct_change()

results = results.dropna()


# -----------------------------
# Print results
# -----------------------------
final_value = results["portfolio_value"].iloc[-1]

mean_return = results["daily_return"].mean()

daily_volatility = results["daily_return"].std()

annualized_return = (1 + mean_return) ** 252 - 1

sharpe_ratio = (mean_return / daily_volatility) * np.sqrt(252)

running_max = results["portfolio_value"].cummax()
drawdown = results["portfolio_value"] / running_max - 1
maximum_drawdown = drawdown.min()

print("\nMomentum Baseline Results")
print("-------------------------")
print(f"Final portfolio value: {final_value:.4f}")
print(f"Annualized return: {annualized_return:.4f}")
print(f"Sharpe ratio: {sharpe_ratio:.4f}")
print(f"Maximum drawdown: {maximum_drawdown:.4f}")
print(f"Mean daily return: {mean_return:.6f}")
print(f"Daily volatility: {daily_volatility:.6f}")
print(f"Average turnover: {results['turnover'].mean():.6f}")

# -----------------------------
# Plot portfolio value
# -----------------------------
plt.figure(figsize=(10, 5))

plt.plot(
    results.index,
    results["portfolio_value"],
    label="Momentum"
)

plt.axhline(
    y=1.0,
    linestyle="--",
    label="Starting Value"
)

plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.title("Momentum Baseline Portfolio Performance")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig("momentum_baseline.png")

print("\nPlot saved to momentum_baseline.png")

plt.show()