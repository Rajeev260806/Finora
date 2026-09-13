# Baseline Strategies - Assumptions and Limitations

This document outlines the key assumptions, limitations, and design choices for each baseline strategy implemented in Phase 1. Understanding these is crucial for interpreting results and comparing against the GNN+RL agent.

## General Assumptions (Applicable to All Strategies)

1. **Data Frequency**: All strategies operate on daily frequency data
2. **Transaction Costs**: Modeled at 10 basis points (0.1%) per trade, matching Person B's PortfolioEnv implementation
3. **Look-Ahead Bias Prevention**: All strategies use only data available up to the rebalancing point
4. **Initial Capital**: All strategies start with portfolio value of 1.0 (normalized)
5. **Cash Handling**: Strategies may hold cash when not fully invested; cash earns zero return

## Strategy-Specific Assumptions

### Equal Weight Baseline
- **Lookback Window**: 20 days (used only for environment state, not strategy calculation)
- **Rebalancing**: Daily (constant weights maintained)
- **Weight Distribution**: Equal allocation across all available assets (~equal weights after softmax)
- **Cash Allocation**: Minimal (strategy pushes allocation to assets via negative cash logit)
- **Turnover**: Very low after initial step (weights remain ~constant)
- **Limitation**: Does not consider asset characteristics, returns, or risk

### Momentum Baseline (12-1 Month)
- **Lookback Period**: Approximately 12 months (252 trading days)
- **Skip Period**: Most recent month (21 trading days) to avoid microstructure effects
- **Ranking**: Assets ranked by past returns (12-1 month)
- **Portfolio Construction**: Long top N stocks (N=5), equal weighted among selected
- **Cash Allocation**: Fixed 20% cash, 80% invested in stocks
- **Rebalancing Frequency**: Monthly (every 21 trading days)
- **Weight Calculation**: Equal weight among selected stocks at each rebalancing
- **Limitations**: 
  - Assumes past winners continue to win (may not hold in all market regimes)
  - Skip period arbitrary (1 month chosen as standard practice)
  - Top N selection arbitrary (chosen as 5 for diversification)

### Mean-Variance Baseline (Markowitz Optimization)
- **Lookback Period**: 252 trading days (~1 year) for return/covariance estimation
- **Rebalancing Frequency**: Monthly (every 21 trading days)
- **Objective Function**: Maximize (expected return - risk_aversion × variance)
- **Risk Aversion Parameter**: λ = 1.0 (moderate risk aversion)
- **Expected Returns**: Simple historical average of returns over lookback period
- **Covariance Matrix**: Sample covariance of returns over lookback period
- **Constraints**: 
  - Long-only (no short selling)
  - Weights sum to 1.0 (full investment, no cash allocation in MV baseline)
  - No leverage (weights ≤ 1.0)
- **Optimization Method**: Simple gradient ascent with constraints (500 iterations, learning rate 0.01)
- **Limitations**:
  - Assumes historical averages are good estimates of future returns (estimation risk)
  - Sample covariance can be unstable with limited data
  - Assumes returns are normally distributed
  - Single-period optimization does not account for intertemporal hedging demands
  - Sensitivity to inputs (small changes in returns/covariance can lead to large weight changes)

### Buy and Hold Baseline
- **Initial Allocation**: Equal weight across all assets at t=0
- **Rebalancing**: None (buy and hold forever)
- **Weight Dynamics**: Weights drift based on individual asset performance
- **Limitations**:
  - Ignores changing market conditions and asset fundamentals
  - Can lead to highly concentrated portfolios over time
  - Does not control for risk exposure

## Data and Implementation Details

### Universe
- 20 stocks across 5 sectors (as established in Phase 1 by Person A)
- Daily OHLCV data cleaned and processed by Person A's pipeline

### Environment Integration
- All strategies integrate with Person B's PortfolioEnv
- Transaction costs automatically applied by environment at 10 bps
- Environment state includes 20-day window of returns (used for observation, not strategy logic in most cases)

### Randomness and Reproducibility
- Where applicable, strategies use seed=0 for reproducibility
- Equal weight uses fixed seed in environment reset
- Momentum and Mean-Variance use deterministic calculations

## Validation Checks Performed

Each baseline was validated to ensure:
1. No NaN or infinite portfolio values
2. Transaction costs properly accounted for in environment
3. Reasonable turnover levels (not excessively high unless intended by design)
4. Positive Sharpe ratios for sensible strategies (though may be negative in downturns)
5. Respected rebalancing frequency (for momentum and mean-variance)
6. Weights sum to 1.0 (or ≤1.0 if cash allowed)
7. No extreme leverage (unless explicitly modeled)

## Usage Notes for Comparison with RL Agent

1. **Consistent Metrics**: All baselines report the same set of metrics for fair comparison:
   - Final portfolio value
   - Cumulative and annualized return
   - Sharpe ratio (annualized)
   - Maximum drawdown
   - Volatility (annualized)
   - Average turnover

2. **Visualization Compatibility**: Outputs are formatted to work with the visualization suite in src/visualization/

3. **CSV Outputs**: Comparison scripts generate CSV files with metrics for easy importing and further analysis

4. **Time Period**: All strategies evaluated on the same historical period (defined by the cleaned dataset)

## Future Improvements/Considerations

For future work, consider:
1. Transaction cost models that vary with volume or volatility
2. Including constraints for sector/industry diversification
3. Alternative estimation methods (shrinkage estimators for covariance)
4. Different rebalancing frequencies (weekly, quarterly)
5. Including factors beyond momentum (value, quality, low volatility)
6. Robust optimization techniques for mean-variance to address estimation error