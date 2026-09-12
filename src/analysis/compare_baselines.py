import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.portfolioenv import PortfolioEnv


# -----------------------------
# Buy & Hold
# -----------------------------
def run_buy_and_hold(data):
    returns = data["Returns Daily"].unstack(level="Ticker")
    returns = returns.fillna(0.0)
    returns = returns.sort_index(axis=1)

    n_assets = len(returns.columns)
    initial_weights = np.ones(n_assets) / n_assets

    portfolio_value = 1.0
    portfolio_values = []
    portfolio_returns = []

    asset_values = initial_weights.copy()

    for date, daily_returns in returns.iterrows():

        asset_values = asset_values * (1 + daily_returns.values)

        new_portfolio_value = asset_values.sum()

        daily_return = (new_portfolio_value / portfolio_value) - 1

        portfolio_value = new_portfolio_value

        portfolio_values.append(portfolio_value)
        portfolio_returns.append(daily_return)

    return pd.DataFrame(
        {
            "portfolio_value": portfolio_values,
            "portfolio_return": portfolio_returns,
        },
        index=returns.index,
    )


# -----------------------------
# Equal Weight
# -----------------------------
def run_equal_weight(data):
    env = PortfolioEnv(
        data,
        window=20,
        transaction_cost_bps=10.0
    )

    obs, info = env.reset(seed=0)

    action = np.zeros(
        env.action_space.shape,
        dtype=np.float32
    )

    action[-1] = -10.0

    done = False

    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    return env.get_history_df()


# -----------------------------
# Momentum
# -----------------------------
def run_momentum(data):

    LOOKBACK_DAYS = 252
    SKIP_DAYS = 21
    TOP_N = 5

    INVESTED_WEIGHT = 0.80
    CASH_WEIGHT = 0.20

    TRANSACTION_COST_BPS = 10.0
    INITIAL_VALUE = 1.0
    SEED = 0

    env = PortfolioEnv(
        data=data,
        window=LOOKBACK_DAYS + SKIP_DAYS,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        initial_cash=INITIAL_VALUE,
        seed=SEED
    )

    obs, info = env.reset(seed=SEED)

    def calculate_momentum(returns, current_idx):

        start = current_idx - LOOKBACK_DAYS
        end = current_idx - SKIP_DAYS

        historical_returns = returns[start:end]

        momentum = (
            np.prod(1.0 + historical_returns, axis=0) - 1.0
        )

        return momentum

    def weights_to_action(weights):

        small_value = 1e-6

        safe_weights = np.maximum(
            weights,
            small_value
        )

        return np.log(safe_weights)

    portfolio_values = []
    dates = []

    previous_month = None
    target_weights = None

    terminated = False

    while not terminated:

        current_idx = env.current_idx
        current_date = env.dates[current_idx]

        current_month = current_date.to_period("M")

        if previous_month != current_month:

            momentum_scores = calculate_momentum(
                env.returns,
                current_idx
            )

            ranking = np.argsort(
                momentum_scores
            )[::-1]

            selected = ranking[:TOP_N]

            target_weights = np.zeros(
                env.n_assets + 1
            )

            stock_weight = INVESTED_WEIGHT / TOP_N

            target_weights[selected] = stock_weight

            target_weights[-1] = CASH_WEIGHT

            action = weights_to_action(
                target_weights
            )

            previous_month = current_month

            print(
                f"{current_date.date()} | "
                f"Top stocks: "
                f"{[env.tickers[i] for i in selected]}"
            )

        else:

            action = weights_to_action(
                target_weights
            )

        obs, reward, terminated, truncated, info = env.step(
            action
        )

        portfolio_values.append(
            info["portfolio_value"]
        )

        dates.append(
            info["date"]
        )

        results = env.get_history_df()

    results["portfolio_return"] = (
        results["portfolio_value"].pct_change()
    )

    return results.dropna()


# -----------------------------
# Calculate Metrics
# -----------------------------
def calculate_metrics(history, return_column):

    returns = history[return_column].dropna()

    cumulative_return = (
        (1 + returns).prod() - 1
    )

    annualized_return = (
        (1 + cumulative_return)
        ** (252 / len(returns))
    ) - 1

    volatility = (
        returns.std() * np.sqrt(252)
    )

    sharpe = (
        returns.mean()
        / returns.std()
        * np.sqrt(252)
        if returns.std() != 0
        else np.nan
    )

    portfolio = history["portfolio_value"]

    running_max = portfolio.cummax()

    drawdown = (
        portfolio / running_max - 1
    )

    max_drawdown = drawdown.min()

    return {
    "Final Portfolio Value": history["portfolio_value"].iloc[-1],
    "Cumulative Return": cumulative_return,
    "Annualized Return": annualized_return,
    "Sharpe Ratio": sharpe,
    "Volatility": volatility,
    "Max Drawdown": max_drawdown,
}


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":

    if len(sys.argv) < 2:

        raise SystemExit(
            "Usage: python compare_baselines.py "
            "<path to full_dataset.parquet>"
        )

    data = pd.read_parquet(
        sys.argv[1]
    )

    print("Running Buy & Hold...")
    buy_hold = run_buy_and_hold(data)

    print("Running Equal-Weight...")
    equal_weight = run_equal_weight(data)

    print("Running Momentum...")
    momentum = run_momentum(data)

    # -------------------------
    # Metrics
    # -------------------------

    bh_metrics = calculate_metrics(
        buy_hold,
        "portfolio_return"
    )

    ew_metrics = calculate_metrics(
        equal_weight,
        "raw_return"
    )

    momentum_metrics = calculate_metrics(
        momentum,
        "portfolio_return"
    )

    # -------------------------
    # Comparison
    # -------------------------

    comparison = pd.DataFrame(
        {
            "Buy & Hold": bh_metrics,
            "Equal-Weight": ew_metrics,
            "Momentum": momentum_metrics,
        }
    )

    print("\nBaseline Performance Comparison")
    print("--------------------------------")

    print(
        comparison.to_string()
    )

    # -------------------------
    # Save CSV
    # -------------------------

    comparison.to_csv(
        "baseline_comparison.csv"
    )

    # -------------------------
    # Plot
    # -------------------------

    plt.figure(figsize=(10, 5))

    plt.plot(
        buy_hold.index,
        buy_hold["portfolio_value"],
        label="Buy & Hold"
    )

    plt.plot(
        equal_weight.index,
        equal_weight["portfolio_value"],
        label="Equal-Weight"
    )

    plt.plot(
        momentum.index,
        momentum["portfolio_value"],
        label="Momentum"
    )

    plt.xlabel("Time")
    plt.ylabel("Portfolio value")

    plt.title(
        "Finora — Baseline Strategy Comparison"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        "baseline_comparison.png",
        dpi=150
    )

    print("\nSaved:")
    print("baseline_comparison.csv")
    print("baseline_comparison.png")