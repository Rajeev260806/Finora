# Model Assumptions

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
