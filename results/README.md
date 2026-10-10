# Phase 3 Results - Person C

This directory contains the deliverables completed by Person C for Phase 3 of the Finora project: baseline performance evaluation, graph visualizations, and allocation charts.

## 📁 Directory Structure

```
results/
├── ASSUMPTIONS.md              # Key assumptions, parameters, and limitations
├── README.md                   # This file
├── baselines/
│   └── mean_variance_full/     # Mean-variance baseline on full dataset
│       ├── params.json         # Parameters used for the baseline
│       ├── metrics.json        # Performance metrics (Sharpe, returns, drawdown, etc.)
│       ├── weights.csv         # Portfolio weights over time (date, AAPL, ADBE, ..., CASH)
│       └── portfolio_value.csv # Portfolio value and daily returns over time
└── plots/
    ├── evolving_graph_animation.gif  # Animated correlation graph evolution over time
    ├── allocation_over_time.png      # Stacked area chart of portfolio weight allocation
    └── mean_variance_equity.png      # Equity curve showing portfolio value growth
```

## 🚀 How to Reproduce the Work

### Prerequisites
- Python 3.11+
- Required packages: numpy, pandas, matplotlib, networkx, pickle
- Project directory structure must be maintained

### 1. Reproduce Mean-Variance Baseline (Full Dataset)

```bash
python src/baselines/mean_variance_baseline.py data/clean/full_dataset.parquet
```

Outputs will be saved to:
- `results/baselines/mean_variance_full/` (weights, metrics, params)
- `results/plots/mean_variance_equity.png` (equity curve)

### 2. Generate Evolving Graph Animation

```bash
python scripts/animate_evolving_graph.py
```

Output: `results/plots/evolving_graph_animation.gif`

### 3. Generate Allocation-Over-Time Chart

```bash
python src/analysis/allocation_over_time.py
```

Output: `results/plots/allocation_over_time.png`

## 📊 Deliverables Overview

### Mean-Variance Baseline (Full Dataset)
- **Purpose**: Benchmark performance for comparison with GNN+RL agent
- **Key Metrics** (from metrics.json):
  - Annualized Return: 18.04%
  - Sharpe Ratio: 0.978
  - Maximum Drawdown: -38.61%
  - Final Portfolio Value: 5.03x initial investment
- **Parameters**: 60-day lookback, monthly rebalancing, λ=1.0, 10 bps transaction costs

### Evolving Graph Visualization
- **Purpose**: Show how stock correlation networks change over time
- **Content**: 60-frame animated GIF showing monthly snapshots
- **Features**: 
  - Nodes colored by sector (Energy, Finance, Real Estate, Technology, Utilities)
  - Edge width represents correlation strength
  - Consistent layout using fixed spring layout seed
  - Timestamp overlay showing date for each snapshot
  - Legend showing sector-color mapping

### Allocation-Over-Time Chart
- **Purpose**: Visualize how portfolio weights evolve over time
- **Content**: Stacked area chart showing weight distribution
- **Features**:
  - 21 layers (20 stocks + cash)
  - Colors consistent with sector mapping from graph visualization
  - Shows rebalancing behavior (sharp changes every ~month)
  - X-axis: Date (2015-03-31 to 2022-12-28)
  - Y-axis: Portfolio weight (0 to 1.0)

## 🔧 Usage Notes

### For Person B (RL Environment Integration)
- The mean-variance baseline provides benchmark metrics for evaluating RL agent performance
- Use `metrics.json` to compare annualized return, Sharpe ratio, and drawdown
- The baseline uses the same transaction cost model (10 bps) as the PortfolioEnv

### For Person A (GNN Work)
- The evolving graph animation shows the temporal dynamics that the GNN aims to capture
- Sector coloring in the animation matches the sector features used in GNN node inputs
- Consistent visualization styling allows for easy comparison in final dashboard

### General Usage
- All files are portable and can be copied to other systems
- Relative paths in scripts assume execution from project root
- To change parameters, modify the values in the scripts or create new parameter files
- Output files can be directly used in reports, presentations, or the final Streamlit dashboard

## 📈 Performance Summary

| Metric | Value |
|--------|-------|
| Annualized Return | 18.04% |
| Sharpe Ratio | 0.978 |
| Max Drawdown | -38.61% |
| Final Value (vs $1) | $5.03 |
| Avg Monthly Turnover | 0.62% |
| Total Transaction Costs | 1.52% |

## 📅 Timeline
- **Start Date**: 2015-03-31
- **End Date**: 2022-12-28
- **Total Trading Days**: 2,456
- **Rebalancing Events**: ~117 (every 21 days)

## 🤝 Handoff Information
This package is ready for use by:
- **Person B**: Use baseline metrics for RL agent evaluation and comparison
- **Person A**: Use evolving graph format expectations for GNN embedding visualization
- **Future Phases**: Baselines provide benchmark for Phases 4-6 comparison activities

For questions or clarification, refer to the detailed assumptions in `ASSUMPTIONS.md` or contact Person C.