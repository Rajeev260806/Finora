import numpy as np
import pandas as pd
import sys
import os
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.analysis.run_test_comparison import (
    run_buy_and_hold,
    run_equal_weight,
    run_momentum,
    run_mean_variance,
    run_gnn_ppo_model,
    calculate_metrics
)


def plot_individual_baseline(history, name, output_file):
    """Plot individual baseline strategy"""
    plt.figure(figsize=(12, 6))

    plt.plot(
        history.index,
        history["portfolio_value"],
        label=name
    )

    plt.axhline(
        y=1.0,
        linestyle="--",
        label="Starting Value"
    )

    plt.title(f"{name} Baseline")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(output_file)
    plt.close()


def generate_baseline_plots(data):
    """Generate individual baseline plots"""
    print("Generating individual baseline plots...")

    # Buy & Hold
    print("  Generating Buy & Hold baseline plot...")
    buy_hold_history = run_buy_and_hold(data)
    buy_hold_history["portfolio_return"] = (
        buy_hold_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )
    plot_individual_baseline(
        buy_hold_history,
        "Buy & Hold",
        "results/plots/buy_and_hold_baseline_test.png"
    )

    # Equal Weight
    print("  Generating Equal Weight baseline plot...")
    equal_weight_history = run_equal_weight(data)
    equal_weight_history["portfolio_return"] = (
        equal_weight_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )
    plot_individual_baseline(
        equal_weight_history,
        "Equal Weight",
        "results/plots/equal_weight_baseline_test.png"
    )

    # Momentum
    print("  Generating Momentum baseline plot...")
    momentum_history = run_momentum(data)
    momentum_history["portfolio_return"] = (
        momentum_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )
    plot_individual_baseline(
        momentum_history,
        "Momentum",
        "results/plots/momentum_baseline_test.png"
    )

    # Mean-Variance
    print("  Generating Mean-Variance baseline plot...")
    mean_variance_history = run_mean_variance(data)
    mean_variance_history["portfolio_return"] = (
        mean_variance_history["portfolio_value"]
        .pct_change()
        .fillna(0)
    )
    plot_individual_baseline(
        mean_variance_history,
        "Mean-Variance",
        "results/plots/mean_variance_baseline_test.png"
    )

    # Try to generate GNN+PPO model plot if possible
    print("  Generating GNN+PPO Model plot...")
    try:
        gnn_ppo_history = run_gnn_ppo_model(data)
        gnn_ppo_history["portfolio_return"] = (
            gnn_ppo_history["portfolio_value"]
            .pct_change()
            .fillna(0)
        )
        plot_individual_baseline(
            gnn_ppo_history,
            "GNN+PPO Model",
            "results/plots/gnn_ppo_baseline_test.png"
        )
    except Exception as e:
        print(f"  Could not generate GNN+PPO model plot: {e}")


def generate_correlation_analysis(data):
    """Generate correlation matrix and network analysis"""
    print("Generating correlation analysis...")

    # Pivot data to get returns matrix
    prices = data.reset_index().pivot(
        index="Date",
        columns="Ticker",
        values="Close"
    ).sort_index()

    returns = prices.pct_change().dropna()

    # Calculate correlation matrix
    correlation_matrix = returns.corr()

    # Save correlation matrix to CSV
    correlation_matrix.to_csv("results/csv/correlation_matrix_test.csv")
    print("  Saved correlation matrix to results/csv/correlation_matrix_test.csv")

    # Plot correlation matrix heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix of Asset Returns')
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.tight_layout()
    plt.savefig("results/plots/correlation_matrix_test.png")
    plt.close()
    print("  Saved correlation matrix plot to results/plots/correlation_matrix_test.png")


def generate_baseline_comparison_csv(histories_dict):
    """Generate baseline comparison CSV similar to the original"""
    print("Generating baseline comparison CSV...")

    # Calculate metrics for each history
    results = {}
    for name, history in histories_dict.items():
        results[name] = calculate_metrics(history)

    comparison_df = pd.DataFrame(results).T

    # Save to CSV
    comparison_df.to_csv("results/csv/baseline_comparison_test.csv")
    print("  Saved baseline comparison to results/csv/baseline_comparison_test.csv")


def generate_assumptions_file():
    """Generate assumptions documentation file"""
    print("Generating assumptions documentation...")

    assumptions_content = """# Model Assumptions

## Dataset
- Test dataset: data/clean/test.parquet
- Period: 2022-12-29 to 2024-12-31
- Frequency: Daily
- Assets: 20 stocks from various sectors

## Baseline Strategies

### Buy & Hold
- Equal weighting across all assets at start
- No rebalancing
- Transaction costs: 0 bps (no trading after initial purchase)

### Equal Weight
- Monthly rebalancing to equal weights
- Transaction costs: 10 bps per trade
- Lookback window: 20 days

### Momentum
- Top 5 stocks based on 12-month momentum (skip 1 month)
- 80% invested in stocks, 20% in cash
- Monthly rebalancing
- Transaction costs: 10 bps per trade
- Lookback window: 252 days + 21 days skip

### Mean-Variance
- Monthly rebalancing using mean-variance optimization
- Risk aversion parameter: 1.0
- Transaction costs: 10 bps per trade
- Lookback window: 60 days
- No cash allocation (fully invested)

### GNN+PPO Model
- Graph Neural Network + Proximal Policy Optimization
- State: Historical returns + GNN embeddings
- Action: Log portfolio weights
- Reward: Portfolio returns
- Transaction costs: 10 bps per trade
- Lookback window: 20 days
- Training data: Full dataset (2015-01-02 to 2024-12-31)
- Embeddings: Pre-trained GNN embeddings from src/gnn/embeddings_original/

## General Assumptions
- Initial portfolio value: $1.0
- Trading occurs at close prices
- Returns are calculated as simple returns
- Annualization assumes 252 trading days per year
- Sharpe ratio assumes risk-free rate of 0%
"""

    with open("results/ASSUMPTIONS_TEST.md", "w") as f:
        f.write(assumptions_content)
    print("  Saved assumptions to results/ASSUMPTIONS_TEST.md")


def generate_readme_file():
    """Generate README file for test results"""
    print("Generating README file...")

    readme_content = """# Test Dataset Strategy Comparison Results

This directory contains the results of running various investment strategies on the test dataset (data/clean/test.parquet).

## Files

### CSV Files
- `strategy_comparison_test.csv`: Comparison of all strategies (Buy & Hold, Equal Weight, Momentum, Mean-Variance, GNN+PPO Model)
- `baseline_comparison_test.csv`: Comparison of just the baseline strategies
- `correlation_matrix_test.csv`: Correlation matrix of asset returns

### Plots
- `portfolio_value_comparison_test.png`: Portfolio value over time for all strategies
- `drawdown_comparison_test.png`: Drawover comparison for all strategies
- `turnover_comparison_test.png`: Turnover comparison for all strategies
- `return_distribution_test.png`: Return distribution comparison
- `strategy_comparison_test.png`: Final portfolio value comparison bar chart
- `buy_and_hold_baseline_test.png`: Individual Buy & Hold strategy performance
- `equal_weight_baseline_test.png`: Individual Equal Weight strategy performance
- `momentum_baseline_test.png`: Individual Momentum strategy performance
- `mean_variance_baseline_test.png`: Individual Mean-Variance strategy performance
- `gnn_ppo_baseline_test.png`: Individual GNN+PPO Model performance (if available)
- `correlation_matrix_test.png`: Heatmap of asset return correlations

### Documentation
- `ASSUMPTIONS_TEST.md`: Detailed assumptions about the dataset and strategies
- `README_TEST.md`: This file

## Key Results

Based on the strategy comparison:
- Buy & Hold performed best in terms of final portfolio value and Sharpe ratio
- Mean-Variance optimization showed strong risk-adjusted returns
- The GNN+PPO model showed reasonable performance, though not exceeding the simple Buy & Hold strategy in this test period
- Momentum strategy showed good annualized return but higher volatility
- Equal Weight underperformed relative to other strategies

## Usage

To regenerate these results:
```bash
python src/analysis/run_test_comparison.py data/clean/test.parquet
```

To generate additional analysis:
```bash
python src/analysis/generate_additional_analysis.py
```
"""

    with open("results/README_TEST.md", "w") as f:
        f.write(readme_content)
    print("  Saved README to results/README_TEST.md")


def main():
    if len(sys.argv) < 2:
        data_path = "data/clean/test.parquet"
        print(f"No data path provided, using default: {data_path}")
    else:
        data_path = sys.argv[1]

    print("Loading dataset for additional analysis...")
    data = pd.read_parquet(data_path)
    print(f"Dataset loaded: {len(data)} rows from {data.index.get_level_values(0).min()} to {data.index.get_level_values(0).max()}")

    # Run strategies to get histories for comparison
    print("\nRunning strategies for additional analysis...")
    buy_hold_history = run_buy_and_hold(data)
    buy_hold_history["portfolio_return"] = buy_hold_history["portfolio_value"].pct_change().fillna(0)

    equal_weight_history = run_equal_weight(data)
    equal_weight_history["portfolio_return"] = equal_weight_history["portfolio_value"].pct_change().fillna(0)

    momentum_history = run_momentum(data)
    momentum_history["portfolio_return"] = momentum_history["portfolio_value"].pct_change().fillna(0)

    mean_variance_history = run_mean_variance(data)
    mean_variance_history["portfolio_return"] = mean_variance_history["portfolio_value"].pct_change().fillna(0)

    try:
        gnn_ppo_history = run_gnn_ppo_model(data)
        gnn_ppo_history["portfolio_return"] = gnn_ppo_history["portfolio_value"].pct_change().fillna(0)
        histories_dict = {
            "Buy & Hold": buy_hold_history,
            "Equal Weight": equal_weight_history,
            "Momentum": momentum_history,
            "Mean-Variance": mean_variance_history,
            "GNN+PPO Model": gnn_ppo_history
        }
    except Exception as e:
        print(f"Could not run GNN+PPO model: {e}")
        histories_dict = {
            "Buy & Hold": buy_hold_history,
            "Equal Weight": equal_weight_history,
            "Momentum": momentum_history,
            "Mean-Variance": mean_variance_history
        }

    # Generate additional analysis
    generate_baseline_plots(data)
    generate_correlation_analysis(data)
    generate_baseline_comparison_csv(histories_dict)
    generate_assumptions_file()
    generate_readme_file()

    print("\nAdditional analysis generation complete!")


if __name__ == "__main__":
    main()