# Summary of Test Dataset Strategy Comparison

## Overview
This summary documents the analysis performed on the test dataset (data/clean/test.parquet) to compare various investment strategies including baseline strategies and a learned GNN+PPO model.

## Dataset Information
- **File**: data/clean/test.parquet
- **Period**: 2022-12-29 to 2024-12-31
- **Frequency**: Daily
- **Assets**: 20 stocks (AAPL, ADBE, AEP, AMT, BAC, COP, CVX, DUK, GS, JPM, MSFT, NEE, NVDA, O, PLD, SLB, SO, SPG, V, XOM)
- **Observations**: 10,080 daily observations

## Strategies Evaluated

### Baseline Strategies
1. **Buy & Hold**: Equal weighting at inception, no rebalancing
2. **Equal Weight**: Monthly rebalancing to equal weights across all assets
3. **Momentum**: Top 5 stocks based on 12-month momentum (skip 1 month), 80% invested in stocks, 20% cash, monthly rebalancing
4. **Mean-Variance**: Monthly rebalancing using mean-variance optimization with risk aversion = 1.0

### Learned Strategy
5. **GNN+PPO Model**: Graph Neural Network + Proximal Policy Optimization model trained on the full dataset (2015-01-02 to 2024-12-31)

## Key Results (Test Period: 2022-12-29 to 2024-12-31)

### Performance Metrics
| Strategy | Final Portfolio Value | Cumulative Return | Annualized Return | Sharpe Ratio | Volatility | Max Drawdown | Avg Turnover |
|----------|----------------------|-------------------|-------------------|--------------|------------|--------------|--------------|
| Buy & Hold | 1.6980 | 0.6980 | 0.3031 | 1.7694 | 0.1517 | -0.1082 | 0.0000 |
| Equal Weight | 1.3246 | 0.3246 | 0.1576 | 1.3085 | 0.1244 | -0.1067 | 0.0041 |
| Momentum | 1.2154 | 0.2154 | 0.2383 | 1.2363 | 0.1647 | -0.0961 | 0.0097 |
| Mean-Variance | 1.3483 | 0.3483 | 0.1849 | 1.6731 | 0.1050 | -0.0918 | 0.0322 |
| GNN+PPO Model | 1.2838 | 0.2838 | 0.1389 | 1.2319 | 0.1175 | -0.1033 | 0.0040 |

### Performance Ranking
1. **Buy & Hold**: Highest final value (1.6980) and Sharpe ratio (1.7694)
2. **Mean-Variance**: Strong risk-adjusted returns (Sharpe = 1.6731)
3. **GNN+PPO Model**: Reasonable performance (Sharpe = 1.2319)
4. **Equal Weight**: Moderate performance
5. **Momentum**: Highest annualized return (23.83%) but highest volatility

## Generated Files

### CSV Data Files
- `results/csv/strategy_comparison_test.csv` - Full strategy comparison
- `results/csv/baseline_comparison_test.csv` - Baseline strategies only
- `results/csv/correlation_matrix_test.csv` - Asset return correlations

### Visualization Plots
- `results/plots/portfolio_value_comparison_test.png` - Portfolio value over time
- `results/plots/drawdown_comparison_test.png` - Drawdown comparison
- `results/plots/turnover_comparison_test.png` - Turnover comparison
- `results/plots/return_distribution_test.png` - Return distribution
- `results/plots/strategy_comparison_test.png` - Final value comparison (bar chart)
- `results/plots/buy_and_hold_baseline_test.png` - Individual Buy & Hold performance
- `results/plots/equal_weight_baseline_test.png` - Individual Equal Weight performance
- `results/plots/momentum_baseline_test.png` - Individual Momentum performance
- `results/plots/mean_variance_baseline_test.png` - Individual Mean-Variance performance
- `results/plots/gnn_ppo_baseline_test.png` - Individual GNN+PPO Model performance
- `results/plots/correlation_matrix_test.png` - Correlation matrix heatmap

### Documentation Files
- `results/ASSUMPTIONS_TEST.md` - Detailed methodology and assumptions
- `results/README_TEST.md` - Overview of results and file descriptions
- `results/SUMMARY_TEST_RESULTS.md` - This summary file

## Files Generated to Match Existing Categories

All generated files follow the naming convention of existing files in the results directory, with "_test" suffix to distinguish them from potential full-dataset results:

**Matching Existing Plot Categories:**
- portfolio_value_comparison_test.png ↔ portfolio_value_comparison.png
- drawdown_comparison_test.png ↔ drawdown_comparison.png
- turnover_comparison_test.png ↔ turnover_comparison.png
- return_distribution_test.png ↔ return_distribution.png
- strategy_comparison_test.png ↔ strategy_comparison.png
- buy_and_hold_baseline_test.png ↔ buy_and_hold_baseline.png
- equal_weight_baseline_test.png ↔ equal_weight_baseline.png
- mean_variance_baseline_test.png ↔ mean_variance_baseline.png
- momentum_baseline_test.png ↔ momentum_baseline.png
- correlation_matrix_test.png ↔ correlation_matrix.png

**Matching Existing CSV Categories:**
- strategy_comparison_test.csv ↔ strategy_comparison.csv
- baseline_comparison_test.csv ↔ baseline_comparison.csv
- correlation_matrix_test.csv ↔ correlation_matrix.csv

## How to Regenerate Results

To regenerate the main comparison:
```bash
python src/analysis/run_test_comparison.py data/clean/test.parquet
```

To generate additional analysis (individual baseline plots, correlation analysis, etc.):
```bash
python src/analysis/generate_additional_analysis.py data/clean/test.parquet
```

## Notes
1. The GNN+PPO model requires pre-trained embeddings located in src/gnn/embeddings_original/
2. If embeddings are not available, the script falls back to using a standard PPO model
3. All strategies use the same transaction costs (10 bps) and initial value ($1.0) for fair comparison
4. Returns are annualized assuming 252 trading days per year
5. Sharpe ratio calculations assume a risk-free rate of 0%