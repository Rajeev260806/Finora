import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.environment.portfolioenv import PortfolioEnv


# ==================================================
# SETTINGS
# ==================================================

LOOKBACK_DAYS = 60
REBALANCE_DAYS = 21

RISK_AVERSION = 1.0

TRANSACTION_COST_BPS = 10.0
INITIAL_VALUE = 1.0
SEED = 0


# ==================================================
# PATHS
# ==================================================

RESULTS_DIR = PROJECT_ROOT / "results" / "baselines" / "mean_variance_test"
PLOTS_DIR = PROJECT_ROOT / "results" / "plots"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD DATA
# ==================================================

if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python mean_variance_baseline.py "
        "<path_to_full_dataset.parquet>"
    )

DATA_PATH = Path(sys.argv[1])

data = pd.read_parquet(DATA_PATH)

print("Dataset loaded.")
print("Number of rows:", len(data))


# ==================================================
# CREATE ENVIRONMENT
# ==================================================

env = PortfolioEnv(
    data=data,
    window=LOOKBACK_DAYS,
    transaction_cost_bps=TRANSACTION_COST_BPS,
    initial_cash=INITIAL_VALUE,
    seed=SEED
)

# The final 20% of dates is the held-out test period.
test_data = pd.read_parquet(PROJECT_ROOT / "data" / "clean" / "test.parquet")
test_start_date = test_data.index.get_level_values("Date").min()
print("Held-out test starts:", test_start_date)

obs, info = env.reset(seed=SEED)

print("Number of stocks:", env.n_assets)
print("Stocks:", env.tickers)


# ==================================================
# MEAN-VARIANCE OPTIMIZATION
# ==================================================

def calculate_mean_variance_weights(returns):
    """
    Calculate long-only Mean-Variance portfolio weights.

    Objective:

        maximize:
            w^T mu - lambda * w^T Sigma w

    Constraints:

        w >= 0
        sum(w) = 1
    """

    mean_returns = returns.mean(axis=0)

    covariance = np.cov(
        returns,
        rowvar=False
    )

    n_assets = returns.shape[1]

    # Start with equal weights
    weights = np.ones(n_assets) / n_assets

    learning_rate = 0.01

    for _ in range(500):

        gradient = (
            mean_returns
            - 2
            * RISK_AVERSION
            * np.dot(covariance, weights)
        )

        weights = weights + learning_rate * gradient

        # Long-only constraint
        weights = np.maximum(
            weights,
            0
        )

        # Budget constraint
        if weights.sum() > 0:
            weights = weights / weights.sum()
        else:
            weights = np.ones(n_assets) / n_assets

    return weights


# ==================================================
# CONVERT WEIGHTS TO ENVIRONMENT ACTION
# ==================================================

def weights_to_action(weights):
    """
    PortfolioEnv expects action logits.
    Convert portfolio weights into logits.
    """

    return np.log(
        np.maximum(
            weights,
            1e-8
        )
    )


# ==================================================
# RUN STRATEGY
# ==================================================

obs, info = env.reset(seed=SEED)

returns = env.returns

current_weights = np.zeros(
    env.n_assets + 1
)

# Start with 100% cash
current_weights[-1] = 1.0

last_rebalance_idx = -REBALANCE_DAYS

portfolio_values = []
dates = []
daily_returns = []
turnovers = []
transaction_costs = []
weights_history = []

done = False


while not done:

    current_idx = env.current_idx
    current_date = env.dates[current_idx]

    # Skip training-period portfolio simulation. Use its returns only
    # as historical lookback for the first test allocation.
    if pd.Timestamp(current_date) < pd.Timestamp(test_start_date):
        env.step(weights_to_action(current_weights))
        continue

    # Start the test portfolio from cash with value 1.0.
    if not dates:
        env.portfolio_value = INITIAL_VALUE
        env.prev_weights = np.zeros(env.n_assets + 1, dtype=np.float32)
        env.prev_weights[-1] = 1.0
        current_weights = env.prev_weights.copy()
        last_rebalance_idx = current_idx - REBALANCE_DAYS

    # ----------------------------------------------
    # Rebalance
    # ----------------------------------------------

    if (
        current_idx - last_rebalance_idx
        >= REBALANCE_DAYS
    ):

        historical_returns = returns[
            current_idx - LOOKBACK_DAYS:
            current_idx
        ]

        stock_weights = calculate_mean_variance_weights(
            historical_returns
        )

        # Add cash weight
        current_weights = np.append(
            stock_weights,
            0.0
        )

        last_rebalance_idx = current_idx

    # ----------------------------------------------
    # Save weights
    # ----------------------------------------------

    weights_history.append(
        current_weights.copy()
    )

    # ----------------------------------------------
    # Convert weights to action
    # ----------------------------------------------

    action = weights_to_action(
        current_weights
    )

    # ----------------------------------------------
    # Environment step
    # ----------------------------------------------

    obs, reward, terminated, truncated, info = env.step(
        action
    )

    # ----------------------------------------------
    # Save outputs
    # ----------------------------------------------

    portfolio_values.append(
        info["portfolio_value"]
    )

    dates.append(
        info["date"]
    )

    turnovers.append(
        info["turnover"]
    )

    transaction_costs.append(
        info["transaction_cost"]
    )

    daily_returns.append(
        info["raw_return"]
    )

    done = terminated or truncated


print("\nMean-Variance simulation completed.")
print(
    "Number of trading days:",
    len(portfolio_values)
)


# ==================================================
# CREATE DATAFRAMES
# ==================================================

dates = pd.to_datetime(dates)

portfolio_series = pd.Series(
    portfolio_values,
    index=dates,
    name="portfolio_value"
)

returns_series = pd.Series(
    daily_returns,
    index=dates,
    name="daily_return"
)

turnover_series = pd.Series(
    turnovers,
    index=dates,
    name="turnover"
)

transaction_cost_series = pd.Series(
    transaction_costs,
    index=dates,
    name="transaction_cost"
)


# ==================================================
# WEIGHTS DATAFRAME
# ==================================================

weight_columns = env.tickers + ["CASH"]

weights_df = pd.DataFrame(
    weights_history,
    index=dates,
    columns=weight_columns
)


# ==================================================
# PERFORMANCE METRICS
# ==================================================

daily_returns_clean = (
    portfolio_series
    .pct_change()
    .dropna()
)

final_value = portfolio_series.iloc[-1]

num_days = len(daily_returns_clean)

if num_days > 0:
    annualized_return = (
        (final_value / INITIAL_VALUE)
        ** (252 / num_days)
    ) - 1
else:
    annualized_return = 0.0


if daily_returns_clean.std() > 0:
    sharpe_ratio = (
        daily_returns_clean.mean()
        / daily_returns_clean.std()
    ) * np.sqrt(252)
else:
    sharpe_ratio = 0.0


running_max = portfolio_series.cummax()

drawdown = (
    portfolio_series / running_max
) - 1

max_drawdown = drawdown.min()

average_turnover = turnover_series.mean()

total_transaction_cost = (
    transaction_cost_series.sum()
)


# ==================================================
# PRINT RESULTS
# ==================================================

print("\n========================================")
print("Mean-Variance Baseline Results")
print("========================================")

print(
    f"Final portfolio value: "
    f"{final_value:.4f}"
)

print(
    f"Annualized return: "
    f"{annualized_return:.4f}"
)

print(
    f"Sharpe ratio: "
    f"{sharpe_ratio:.4f}"
)

print(
    f"Maximum drawdown: "
    f"{max_drawdown:.4f}"
)

print(
    f"Mean daily return: "
    f"{daily_returns_clean.mean():.6f}"
)

print(
    f"Daily volatility: "
    f"{daily_returns_clean.std():.6f}"
)

print(
    f"Average turnover: "
    f"{average_turnover:.6f}"
)

print(
    f"Total transaction cost: "
    f"{total_transaction_cost:.6f}"
)


# ==================================================
# SAVE PORTFOLIO VALUE
# ==================================================

portfolio_output = pd.DataFrame({
    "date": portfolio_series.index,
    "portfolio_value": portfolio_series.values,
    "daily_return": returns_series.values,
    "turnover": turnover_series.values,
    "transaction_cost": transaction_cost_series.values
})

portfolio_output.to_csv(
    RESULTS_DIR / "portfolio_value.csv",
    index=False
)


# ==================================================
# SAVE WEIGHTS
# ==================================================

weights_output = weights_df.copy()

weights_output.insert(
    0,
    "date",
    weights_output.index
)

weights_output.to_csv(
    RESULTS_DIR / "weights.csv",
    index=False
)


# ==================================================
# SAVE METRICS
# ==================================================

metrics = {
    "final_portfolio_value": float(final_value),
    "annualized_return": float(annualized_return),
    "sharpe_ratio": float(sharpe_ratio),
    "maximum_drawdown": float(max_drawdown),
    "mean_daily_return": float(
        daily_returns_clean.mean()
    ),
    "daily_volatility": float(
        daily_returns_clean.std()
    ),
    "average_turnover": float(
        average_turnover
    ),
    "total_transaction_cost": float(
        total_transaction_cost
    )
}

with open(
    RESULTS_DIR / "metrics.json",
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ==================================================
# SAVE PARAMETERS
# ==================================================

parameters = {
    "lookback_days": LOOKBACK_DAYS,
    "rebalance_days": REBALANCE_DAYS,
    "risk_aversion": RISK_AVERSION,
    "transaction_cost_bps": TRANSACTION_COST_BPS,
    "initial_value": INITIAL_VALUE,
    "seed": SEED,
    "optimization": "long-only mean-variance",
    "transaction_cost_model": "linear turnover bps"
}

with open(
    RESULTS_DIR / "params.json",
    "w"
) as f:

    json.dump(
        parameters,
        f,
        indent=4
    )


# ==================================================
# EQUITY CURVE
# ==================================================

plt.figure(
    figsize=(10, 5)
)

plt.plot(
    portfolio_series.index,
    portfolio_series.values
)

plt.title(
    "Mean-Variance Baseline"
)

plt.xlabel(
    "Date"
)

plt.ylabel(
    "Portfolio Value"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    PLOTS_DIR / "mean_variance_test_equity.png",
    dpi=300
)

plt.close()


# ==================================================
# FINAL OUTPUT
# ==================================================

print("\n========================================")
print("Files saved successfully")
print("========================================")

print(
    RESULTS_DIR / "weights.csv"
)

print(
    RESULTS_DIR / "portfolio_value.csv"
)

print(
    RESULTS_DIR / "metrics.json"
)

print(
    RESULTS_DIR / "params.json"
)

print(
    PLOTS_DIR / "mean_variance_test_equity.png"
)