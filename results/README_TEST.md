# Test Dataset Strategy Comparison Results

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
