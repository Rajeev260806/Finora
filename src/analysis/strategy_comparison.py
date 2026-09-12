import numpy as np
import pandas as pd
import sys

from src.environment.portfolioenv import PortfolioEnv

from src.analysis.compare_baselines import (
    run_buy_and_hold,
    run_equal_weight,
    run_momentum
)

from src.visualization.visualization import (
    plot_portfolio_value,
    plot_drawdown,
    plot_turnover,
    plot_return_distribution,
    plot_final_values
)


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
# Mean-Variance Optimizer
# -----------------------------

def optimize_mean_variance(
    returns,
    risk_aversion=1.0,
    iterations=500,
    learning_rate=0.01
):
    mean_returns = returns.mean().values
    covariance = returns.cov().values

    n_assets = len(mean_returns)

    weights = np.ones(n_assets) / n_assets

    for _ in range(iterations):

        gradient = (
            mean_returns
            - 2 * risk_aversion * covariance @ weights
        )

        weights = weights + learning_rate * gradient

        weights = np.clip(weights, 0, None)

        total = weights.sum()

        if total > 0:
            weights = weights / total
        else:
            weights = np.ones(n_assets) / n_assets

    return weights


# -----------------------------
# Convert Weights to Action
# -----------------------------

def weights_to_action(weights):
    return np.log(
        np.maximum(weights, 1e-8)
    )


# -----------------------------
# Mean-Variance Baseline
# -----------------------------

def run_mean_variance(data):

    prices = data.reset_index().pivot(
        index="Date",
        columns="Ticker",
        values="Close"
        ).sort_index()

    returns = prices.pct_change().dropna()

    env = PortfolioEnv(
        data=data,
        window=LOOKBACK_DAYS,
        transaction_cost_bps=TRANSACTION_COST_BPS,
        seed=SEED
    )

    observation, info = env.reset(seed=SEED)

    n_assets = len(env.tickers)

    done = False

    current_idx = LOOKBACK_DAYS

    while not done:

        if (
            current_idx >= LOOKBACK_DAYS
            and (
                current_idx == LOOKBACK_DAYS
                or current_idx % REBALANCE_DAYS == 0
            )
        ):

            historical_returns = returns.iloc[
                max(
                    0,
                    current_idx - LOOKBACK_DAYS
                ):current_idx
            ]

            historical_returns = (
                historical_returns
                .replace([np.inf, -np.inf], np.nan)
                .dropna(axis=1, how="any")
            )

            available_tickers = [
                ticker
                for ticker in env.tickers
                if ticker in historical_returns.columns
            ]

            if len(available_tickers) > 0:

                clean_returns = historical_returns[
                    available_tickers
                ]

                stock_weights = optimize_mean_variance(
                    clean_returns,
                    risk_aversion=RISK_AVERSION
                )

                weights = np.zeros(n_assets + 1)

                for ticker, weight in zip(
                    available_tickers,
                    stock_weights
                ):
                    ticker_index = env.tickers.index(ticker)
                    weights[ticker_index] = weight

                total_stock_weight = weights[:-1].sum()

                if total_stock_weight > 0:
                    weights[:-1] = (
                        weights[:-1]
                        / total_stock_weight
                    )

                # No cash allocation for MV baseline
                weights[-1] = 0.0

                action = weights_to_action(weights)

            else:

                weights = np.zeros(n_assets + 1)
                weights[-1] = 1.0

                action = weights_to_action(weights)

        else:

            action = np.zeros(n_assets + 1)

            # Keep current allocation if not rebalancing
            if hasattr(env, "weights"):
                current_weights = np.asarray(
                    env.weights,
                    dtype=float
                )

                if len(current_weights) == n_assets + 1:
                    action = weights_to_action(
                        current_weights
                    )
                else:
                    action[-1] = 1.0

            else:
                action[-1] = 1.0

        observation, reward, terminated, truncated, info = env.step(
            action
        )

        done = terminated or truncated

        current_idx += 1

    history = env.get_history_df()

    history["portfolio_return"] = (
        history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )

    return history


# -----------------------------
# Calculate Metrics
# -----------------------------

def calculate_metrics(history):

    portfolio_values = history["portfolio_value"]

    daily_returns = (
        history["portfolio_return"]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    final_value = portfolio_values.iloc[-1]

    cumulative_return = final_value - 1

    if len(daily_returns) > 1:

        annualized_return = (
            (1 + cumulative_return)
            ** (252 / len(daily_returns))
            - 1
        )

        volatility = (
            daily_returns.std()
            * np.sqrt(252)
        )

        if volatility > 0:

            sharpe_ratio = (
                daily_returns.mean()
                / daily_returns.std()
                * np.sqrt(252)
            )

        else:
            sharpe_ratio = 0.0

    else:

        annualized_return = 0.0
        volatility = 0.0
        sharpe_ratio = 0.0

    running_max = portfolio_values.cummax()

    drawdown = (
        portfolio_values
        / running_max
        - 1
    )

    max_drawdown = drawdown.min()

    if "turnover" in history.columns:
        average_turnover = history["turnover"].mean()
    else:
        average_turnover = 0.0

    return {
        "Final Portfolio Value": final_value,
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Volatility": volatility,
        "Max Drawdown": max_drawdown,
        "Average Turnover": average_turnover
    }


# -----------------------------
# Main
# -----------------------------

if __name__ == "__main__":

    print("Loading dataset...")

    if len(sys.argv) < 2:
        raise SystemExit("Usage: python strategy_comparison.py <path to full_dataset.parquet>")
    data_path = sys.argv[1]
    data = pd.read_parquet(data_path)

    print("Dataset loaded.")
    print(f"Number of rows: {len(data)}")
    print()

    # -------------------------
    # Run Buy & Hold
    # -------------------------

    print("Running Buy & Hold...")

    buy_hold_history = run_buy_and_hold(data)

    buy_hold_history["portfolio_return"] = (
        buy_hold_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )

    # -------------------------
    # Run Equal Weight
    # -------------------------

    print("Running Equal Weight...")

    equal_weight_history = run_equal_weight(data)

    equal_weight_history["portfolio_return"] = (
        equal_weight_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )

    # -------------------------
    # Run Momentum
    # -------------------------

    print("Running Momentum...")

    momentum_history = run_momentum(data)

    momentum_history["portfolio_return"] = (
        momentum_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )

    # -------------------------
    # Run Mean-Variance
    # -------------------------

    print("Running Mean-Variance...")

    mean_variance_history = run_mean_variance(data)

    print()
    print("All strategies completed.")
    print()

    # -------------------------
    # Store Histories
    # -------------------------

    histories = {
        "Buy & Hold": buy_hold_history,
        "Equal Weight": equal_weight_history,
        "Momentum": momentum_history,
        "Mean-Variance": mean_variance_history
    }

    # -------------------------
    # Calculate Metrics
    # -------------------------

    results = {}

    for name, history in histories.items():

        results[name] = calculate_metrics(history)

    comparison_df = pd.DataFrame(results).T

    # -------------------------
    # Print Comparison
    # -------------------------

    print("=" * 80)
    print("FINAL BASELINE STRATEGY COMPARISON")
    print("=" * 80)

    print(
        comparison_df.to_string(
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print()

    # -------------------------
    # Save Comparison
    # -------------------------

    comparison_df.to_csv(
        "strategy_comparison.csv"
    )

    print("Saved: strategy_comparison.csv")

    # -------------------------
    # Generate Visualizations
    # -------------------------

    plot_portfolio_value(
        histories,
        "portfolio_value_comparison.png"
    )

    plot_drawdown(
        histories,
        "drawdown_comparison.png"
    )

    plot_turnover(
        histories,
        "turnover_comparison.png"
    )

    returns = {
        name: history["portfolio_return"]
        for name, history in histories.items()
    }

    plot_return_distribution(
        returns,
        "return_distribution.png"
    )

    plot_final_values(
        comparison_df,
        "strategy_comparison.png"
    )

    print("All visualization plots generated successfully.")