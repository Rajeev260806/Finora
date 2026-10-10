# Phase 3: Person C - Assumptions and Limitations

This document outlines the key assumptions, parameters, and limitations for the Phase 3 deliverables created by Person C: mean-variance baseline performance, evolving graph visualization, and allocation-over-time chart.

## 📊 Baseline Performance Assumptions (Mean-Variance)

### General Approach
- **Strategy**: Long-only mean-variance optimization (Markowitz framework)
- **Objective**: Maximize (expected return - risk_aversion × variance)
- **Optimization Method**: Gradient ascent with constraints (500 iterations, learning rate 0.01)

### Parameters Used
- **Lookback Window**: 60 trading days (~3 months) for return/covariance estimation
- **Rebalancing Frequency**: Every 21 trading days (~1 month)
- **Risk Aversion Parameter (λ)**: 1.0 (moderate risk aversion)
- **Transaction Cost Model**: 10 basis points (0.1%) linear cost on turnover
- **Initial Portfolio Value**: 1.0 (normalized)
- **Random Seed**: 0 (for reproducibility)

### Data Handling
- **Expected Returns**: Simple historical average of returns over lookback period
- **Covariance Matrix**: Sample covariance of returns over lookback period
- **Constraints**: 
  - Long-only (no short selling, weights ≥ 0)
  - Budget constraint: weights sum to 1.0 (full investment, no cash allocation in MV baseline)
  - No leverage (weights ≤ 1.0)
- **Look-Ahead Bias Prevention**: All estimations use only data available up to the rebalancing point

### Limitations
- Assumes historical averages are good estimates of future returns (estimation risk)
- Sample covariance can be unstable with limited data
- Assumes returns are normally distributed
- Single-period optimization does not account for intertemporal hedging demands
- Sensitivity to inputs (small changes in returns/covariance can lead to large weight changes)

## 📈 Evolving Graph Visualization Assumptions

### Data Source
- **Graph Snapshots**: Loaded from `graphs/saved_graphs/` (graph_*.pkl files)
- **Temporal Frequency**: Daily graph snapshots (based on file naming convention)
- **Total Snapshots**: ~1,952 graphs (2015-03-31 through 2022-12-28)

### Graph Construction
- **Node Representation**: Each node represents a stock (20 stocks total)
- **Edge Creation**: Edges represent significant correlations above threshold
- **Thresholding Logic**: Same as used in Person A's Phase 2 work (checked via graph attributes)
- **Edge Weight**: Represents correlation strength (absolute value used for width)
- **Node Features**: Sector information extracted from node attributes

### Visualization Parameters
- **Layout Algorithm**: Spring layout with fixed seed (42) for consistent positioning across frames
- **Node Coloring**: By sector using discrete colormap (tab10)
- **Sector Categories**: Energy, Finance, Real Estate, Technology, Utilities
- **Node Size**: Fixed at 700 for visibility
- **Font Size**: 9 for node labels (ticker symbols)
- **Figure Size**: 16x9 inches with legend positioned outside graph area
- **Animation Settings**: 
  - Max Frames: 60 (sampled from full dataset if larger)
  - FPS: 2 frames per second
  - Output Format: GIF with white background

### Limitations
- Temporal sampling may miss rapid correlation changes if sampling rate too low
- Spring layout may not perfectly preserve global structure across all frames
- Sector mapping assumes complete and static sector classification
- Correlation thresholding logic relies on attributes stored in graph snapshots

## 📉 Allocation-Over-Time Chart Assumptions

### Data Source
- **Weight History**: Loaded from `results/baselines/mean_variance/weights.csv`
- **Strategy Source**: Mean-variance baseline weights (from Phase 2 work)
- **Time Period**: Full dataset period (2015-03-31 to 2022-12-28)
- **Trading Days**: 2,456 days

### Chart Construction
- **Chart Type**: Stacked area chart showing portfolio weight evolution
- **X-Axis**: Time (dates sorted chronologically)
- **Y-Axis**: Portfolio weight allocation (0 to 1.0)
- **Layers**: Each layer represents weight allocated to a specific stock or cash
- **Number of Assets**: 21 (20 stocks + 1 cash asset)
- **Color Scheme**: Consistent with sector coloring used in graph visualization
- **Transparency**: Alpha = 0.85 for layer visibility
- **Figure Size**: 16x8 inches
- **Y-Limit**: Set to 0-1.05 to show full weight range with small margin
- **Grid**: Horizontal grid lines at alpha 0.3 for readability
- **Legend**: Positioned outside chart area (bbox_to_anchor=(1.01, 1))

### Limitations
- Reflects only mean-variance strategy weights (not RL agent weights as originally intended for Phase 3)
- Uses placeholder/data from Phase 2 mean-variance baseline due to incomplete RL training at Phase 3 stage
- Weights sum to exactly 1.0 at each rebalancing point (due to budget constraint)
- Does not show intra-rebalancing period weight drift (weights held constant between rebalances)
- Cash weight appears as 0.0 in this implementation (all funds invested in stocks)

## 🔄 Cross-Consistency Assumptions

### Sector Mapping
- Sector-color mapping is consistent across all three visualizations
- Sector definitions derived from Node attributes in graph snapshots
- Same color palette (tab10) used for node colors in graph visualization and stack colors in allocation chart

### Temporal Alignment
- All visualizations cover the same time period (2015-03-31 through 2022-12-28)
- Graph snapshots and portfolio weights are aligned by date
- Annualization factors assume 252 trading days per year

### Reproducibility
- All scripts use fixed random seeds where applicable (seed=0)
- Deterministic algorithms used for mean-variance optimization and graph layout
- File paths resolved relative to project root for portability

## 📋 Validation Checks Performed

Each deliverable was validated to ensure:
1. **Mean-Variance Baseline**:
   - No NaN or infinite portfolio values
   - Transaction costs properly accounted for
   - Reasonable Sharpe ratios (positive for sensible strategies)
   - Weights sum to 1.0 at each rebalancing point
   - Correct rebalancing frequency (every 21 days)

2. **Evolving Graph Animation**:
   - All frames generated without errors
   - Consistent node positioning across frames (fixed layout seed)
   - Sector coloring consistent with ticker-sector mapping
   - Timestamp overlay accurate and readable
   - Legend properly displayed outside graph area

3. **Allocation-Over-Time Chart**:
   - Properly formatted stacked area chart
   - Weights sum to approximately 1.0 at each point (exactly 1.0 at rebalancing points)
   - Clear legend and axis labels
   - Sensible weight evolution patterns showing rebalancing behavior

## 🛠️ Technical Implementation Notes

### Software Versions
- Python 3.11.x
- Key Libraries: numpy, pandas, matplotlib, networkx, pickle

### File Structure Outputs
```
results/
├── baselines/
│   └── mean_variance_full/          # Full dataset baseline outputs
│       ├── params.json              # Parameters used
│       ├── metrics.json             # Performance metrics
│       ├── weights.csv              # Portfolio weights over time
│       └── portfolio_value.csv      # Portfolio value and returns over time
├── plots/
│   ├── evolving_graph_animation.gif # Animated correlation graph evolution
│   ├── allocation_over_time.png     # Stacked area weight allocation chart
│   └── mean_variance_equity.png     # Equity curve for mean-variance baseline
└── ASSUMPTIONS.md                   # This file
```

### Parameters File Format (params.json)
```json
{
  "lookback_days": 60,
  "rebalance_days": 21,
  "risk_aversion": 1.0,
  "transaction_cost_bps": 10.0,
  "initial_value": 1.0,
  "seed": 0,
  "optimization": "long-only mean-variance",
  "transaction_cost_model": "linear turnover bps"
}
```

### Metrics File Format (metrics.json)
```json
{
  "final_portfolio_value": 5.030700883003482,
  "annualized_return": 0.1803764107652024,
  "sharpe_ratio": 0.9775092348158985,
  "maximum_drawdown": -0.38612067963896923,
  "mean_daily_return": 0.0007344717009577269,
  "daily_volatility": 0.011927638511454444,
  "average_turnover": 0.006206886836779622,
  "total_transaction_cost": 0.015244114071130754
}
```

---
*Documentation generated as part of Phase 3 completion for Person C.*