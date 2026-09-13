# Baseline Strategies

This directory contains implementations of various baseline portfolio strategies for comparison with the GNN+RL agent.

## Available Baselines

1. **Equal Weight** (`equal_weight_baseline.py`) - Equally allocates capital across all assets
2. **Momentum** (`momentum_baseline.py`) - Buys recent winners, sells recent losers (12-1 month lookback)
3. **Mean-Variance** (`mean_variance_baseline.py`) - Markowitz optimization maximizing return for given risk
4. **Buy and Hold** (`buy_and_hold_baseline.py`) - Equal initial allocation with no rebalancing

## How to Run

Each baseline script can be run independently:

```bash
python equal_weight_baseline.py data/clean/full_dataset.parquet
python momentum_baseline.py data/clean/full_dataset.parquet
python mean_variance_baseline.py data/clean/full_dataset.parquet
python buy_and_hold_baseline.py data/clean/full_dataset.parquet
```

## Outputs

Each script generates:
- Performance metrics printed to console (final value, Sharpe ratio, max drawdown, turnover, etc.)
- A PNG plot showing portfolio value over time
- Results are saved in the `results/plots/` directory

## How to Add a New Baseline

To add a new baseline strategy:

1. Create a new Python file in this directory (e.g., `my_strategy_baseline.py`)
2. Follow the pattern:
   - Load the cleaned dataset from parquet
   - Create a PortfolioEnv instance with appropriate window and transaction costs
   - Implement your strategy logic to generate weights at each rebalancing period
   - Step through the environment using your strategy's actions
   - Calculate and print performance metrics
   - Generate a portfolio value plot

3. Ensure your script:
   - Uses the same transaction cost model as the environment (10 bps)
   - Prevents look-ahead bias by using only historical data
   - Documents assumptions and parameters clearly
   - Saves outputs to the standard locations

## Integration with Comparison Framework

The baselines are designed to work with the comparison scripts:
- `src/analysis/compare_baselines.py` - compares 3 core strategies
- `src/analysis/strategy_comparison.py` - compares all 4 strategies plus visualizations

To include your new baseline in comparisons, add it to the appropriate comparison script.

## Assumptions and Limitations

See the main assumptions document for details on:
- Data frequency (daily)
- Estimation windows (252 days for momentum/MV)
- Rebalancing frequency (monthly/21 days)
- Transaction cost model (10 bps)
- Look-ahead bias prevention
- Strategy-specific constraints