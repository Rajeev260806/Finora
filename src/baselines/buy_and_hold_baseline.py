import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def run_buy_and_hold(data):
    # Convert Returns Daily into a Date x Ticker table
    returns = data["Returns Daily"].unstack(level="Ticker")
    returns = returns.fillna(0.0)
    # Make sure the stocks are in a consistent order
    returns = returns.sort_index(axis=1)

    # Start with equal investment in every stock
    n_assets = len(returns.columns)
    initial_weights = np.ones(n_assets) / n_assets

    # Portfolio starts at 1.0
    portfolio_value = 1.0

    portfolio_values = []
    portfolio_returns = []

    # Current value invested in each stock
    asset_values = initial_weights.copy()

    for date, daily_returns in returns.iterrows():

        # Each stock's value changes according to its daily return
        asset_values = asset_values * (1 + daily_returns.values)

        # Total portfolio value
        new_portfolio_value = asset_values.sum()

        # Portfolio return for this day
        daily_portfolio_return = (
            new_portfolio_value / portfolio_value
        ) - 1

        portfolio_value = new_portfolio_value

        portfolio_values.append(portfolio_value)
        portfolio_returns.append(daily_portfolio_return)

    history = pd.DataFrame(
        {
            "portfolio_value": portfolio_values,
            "portfolio_return": portfolio_returns,
        },
        index=returns.index,
    )

    return history


if __name__ == "__main__":

    # Load the cleaned dataset
    data = pd.read_parquet("data/clean/full_dataset.parquet")

    # Run Buy & Hold
    history = run_buy_and_hold(data)

    # Display results
    print(
        f"Final portfolio value (buy & hold): "
        f"{history['portfolio_value'].iloc[-1]:.4f}"
    )

    print(
        f"Mean daily return: "
        f"{history['portfolio_return'].mean():.6f}"
    )

    print(
        f"Daily volatility: "
        f"{history['portfolio_return'].std():.6f}"
    )

    # Create portfolio value plot
    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        history.index,
        history["portfolio_value"]
    )

    ax.axhline(
        1.0,
        color="black",
        linestyle="--",
        linewidth=0.8
    )

    ax.set_title("Buy & Hold Baseline — Portfolio Value")
    ax.set_ylabel("Portfolio value (start = 1.0)")

    fig.tight_layout()

    fig.savefig(
        "buy_and_hold_baseline.png",
        dpi=150
    )

    print("Plot saved to buy_and_hold_baseline.png")