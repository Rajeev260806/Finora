import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from src.environment.portfolioenv import PortfolioEnv


# -----------------------------
# Settings
# -----------------------------
LOOKBACK_DAYS = 252
REBALANCE_DAYS = 21

RISK_AVERSION = 1.0

TRANSACTION_COST_BPS = 10.0
INITIAL_VALUE = 1.0
SEED = 0


# -----------------------------
# Load data
# -----------------------------
if len(sys.argv) < 2:
    raise SystemExit("Usage: python mean_variance_baseline.py <path to full_dataset.parquet>")
DATA_PATH = sys.argv[1]
data = pd.read_parquet(DATA_PATH)

print("Dataset loaded.")
print("Number of rows:", len(data))


# -----------------------------
# Create environment
# -----------------------------
env = PortfolioEnv(
    data=data,
    window=LOOKBACK_DAYS,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    initial_cash=INITIAL_VALUE,
    seed=SEED
)

obs, info = env.reset(seed=SEED)

print("Number of stocks:", env.n_assets)
print("Stocks:", env.tickers)
# -----------------------------
# Mean-Variance optimization
# -----------------------------
def calculate_mean_variance_weights(returns):
    """
    Calculate portfolio weights using Mean-Variance optimization.

    Objective:
        maximize expected return - risk_aversion * variance

    Constraints:
        - Long-only
        - Weights sum to 1
    """

    mean_returns = returns.mean(axis=0)
    covariance = np.cov(returns, rowvar=False)

    n_assets = returns.shape[1]

    # Start with equal weights
    weights = np.ones(n_assets) / n_assets

    # Simple iterative optimization
    learning_rate = 0.01

    for _ in range(500):

        portfolio_return = np.dot(weights, mean_returns)

        portfolio_variance = np.dot(
            weights,
            np.dot(covariance, weights)
        )

        # Gradient of:
        # expected return - risk_aversion * variance
        gradient = (
            mean_returns
            - 2 * RISK_AVERSION * np.dot(covariance, weights)
        )

        weights = weights + learning_rate * gradient

        # Long-only constraint
        weights = np.maximum(weights, 0)

        # Normalize so weights sum to 1
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_assets) / n_assets

    return weights
# -----------------------------
# Run Mean-Variance strategy
# -----------------------------

def weights_to_action(weights):
    """Convert portfolio weights into logits for PortfolioEnv."""
    return np.log(np.maximum(weights, 1e-8))


obs, info = env.reset(seed=SEED)

returns = env.returns

current_weights = np.zeros(env.n_assets + 1)
current_weights[-1] = 1.0  # Start with 100% cash

last_rebalance_idx = -REBALANCE_DAYS

portfolio_values = []
dates = []
turnovers = []

done = False

while not done:

    current_idx = env.current_idx

    # Rebalance every 21 trading days
    if current_idx - last_rebalance_idx >= REBALANCE_DAYS:

        historical_returns = returns[
            current_idx - LOOKBACK_DAYS : current_idx
        ]

        stock_weights = calculate_mean_variance_weights(
            historical_returns
        )

        # Add cash weight
        current_weights = np.append(stock_weights, 0.0)

        last_rebalance_idx = current_idx

    # Convert weights to environment action
    action = weights_to_action(current_weights)

    obs, reward, terminated, truncated, info = env.step(action)

    portfolio_values.append(info["portfolio_value"])
    dates.append(info["date"])
    turnovers.append(info["turnover"])

    done = terminated or truncated


print("Mean-Variance simulation completed.")
print("Number of trading days:", len(portfolio_values))
# -----------------------------
# Performance metrics
# -----------------------------

portfolio_series = pd.Series(
    portfolio_values,
    index=pd.to_datetime(dates)
)

daily_returns = portfolio_series.pct_change().dropna()

final_value = portfolio_series.iloc[-1]

annualized_return = (
    (final_value / INITIAL_VALUE)
    ** (252 / len(daily_returns))
) - 1

sharpe_ratio = (
    daily_returns.mean() / daily_returns.std()
) * np.sqrt(252)

running_max = portfolio_series.cummax()
drawdown = (portfolio_series / running_max) - 1
max_drawdown = drawdown.min()

average_turnover = np.mean(turnovers)


print("\nMean-Variance Baseline Results")
print("------------------------------")
print(f"Final portfolio value: {final_value:.4f}")
print(f"Annualized return: {annualized_return:.4f}")
print(f"Sharpe ratio: {sharpe_ratio:.4f}")
print(f"Maximum drawdown: {max_drawdown:.4f}")
print(f"Mean daily return: {daily_returns.mean():.6f}")
print(f"Daily volatility: {daily_returns.std():.6f}")
print(f"Average turnover: {average_turnover:.6f}")


# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    portfolio_series.index,
    portfolio_series.values
)

plt.title("Mean-Variance Baseline")
plt.xlabel("Date")
plt.ylabel("Portfolio Value")
plt.grid(True)

plt.tight_layout()

plt.savefig("mean_variance_baseline.png")

print("Plot saved to mean_variance_baseline.png")