#!/usr/bin/env python3
"""
Generate comparative visualizations for GAT Hybrid, GCN, and Buy and Hold methods.
Creates at least 10 different graphs comparing performance, risk, and allocation characteristics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Define paths using raw strings to avoid escape issues
BASE_DIR = r"D:/Projects/ML_Package"
RESULTS_DIR = os.path.join(BASE_DIR, "results")
VIS_DIR = os.path.join(BASE_DIR, "results/visualizations_comparison")

# File paths for different methods
GAT_HYBRID_PATH = os.path.join(RESULTS_DIR, "ppo_eval/final_original_gat_5feat_100k")
GCN_PATH = os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k")
DATA_PATH = os.path.join(BASE_DIR, "data", "clean")

def load_portfolio_values(method_path):
    """Load portfolio values from CSV file."""
    pf_file = os.path.join(method_path, "portfolio_value.csv")
    if os.path.exists(pf_file):
        df = pd.read_csv(pf_file)
        # Convert date column to datetime for proper handling
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        print(f"Portfolio value file not found: {pf_file}")
        return None

def load_weights(method_path):
    """Load weights from CSV file."""
    weights_file = os.path.join(method_path, "weights.csv")
    if os.path.exists(weights_file):
        df = pd.read_csv(weights_file)
        # Convert date column to datetime for proper handling
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        # Ensure all weight columns are numeric
        weight_cols = [col for col in df.columns if col not in ['date', 'Unnamed: 0']]
        for col in weight_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    else:
        print(f"Weights file not found: {weights_file}")
        return None

def load_metrics(method_path):
    """Load metrics from JSON file."""
    metrics_file = os.path.join(method_path, "metrics.json")
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            return json.load(f)
    else:
        print(f"Metrics file not found: {metrics_file}")
        return None

def compute_buy_and_hold(test_data_path=None):
    """Compute Buy and Hold portfolio performance from raw data."""
    # Use full dataset to match the date range of GAT/GCN evaluation results
    full_data_path = os.path.join(DATA_PATH, "full_dataset.parquet")

    if not os.path.exists(full_data_path):
        print(f"Full dataset not found: {full_data_path}")
        return None

    print(f"Using full dataset for Buy and Hold: {full_data_path}")

    # Load returns data
    returns_df = pd.read_parquet(full_data_path)

    # Load GAT data to get date range for filtering
    gat_path = os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k")
    gat_portfolio_file = os.path.join(gat_path, "portfolio_value.csv")
    if os.path.exists(gat_portfolio_file):
        gat_df_temp = pd.read_csv(gat_portfolio_file, parse_dates=['date'])
        gat_start = gat_df_temp['date'].min()
        gat_end = gat_df_temp['date'].max()
        # Filter returns_df to date range [gat_start, gat_end]
        # Returns_df index is date (first level)
        returns_df = returns_df.loc[gat_start:gat_end]
        print(f"Filtered Buy and Hold to GAT date range: {gat_start} to {gat_end}")
    else:
        print(f"Warning: GAT portfolio file not found at {gat_portfolio_file}, using full date range")

    # Assuming the data has returns for different tickers
    # We need to compute equal-weighted portfolio returns
    if "Returns Daily" in returns_df.columns:
        # Unstack to get returns by ticker and date
        returns_wide = returns_df["Returns Daily"].unstack(level="Ticker")
        returns_wide = returns_wide.fillna(0.0)

        # Equal weight portfolio (1/N)
        equal_weights = 1.0 / len(returns_wide.columns)
        portfolio_returns = returns_wide.sum(axis=1) * equal_weights

        # Calculate cumulative portfolio value (starting at 1.0)
        portfolio_value = (1 + portfolio_returns).cumprod()

        # Create DataFrame similar to the others
        pf_df = pd.DataFrame({
            'timestep': range(len(portfolio_value)),
            'portfolio_value': portfolio_value.values
        })

        return pf_df
    else:
        print("Could not find Returns Daily column in data")
        return None

def calculate_drawdown(portfolio_values):
    """Calculate drawdown series from portfolio values."""
    running_max = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - running_max) / running_max
    return drawdown

def calculate_returns(portfolio_values):
    """Calculate periodic returns from portfolio values."""
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    # Prepend 0 for the first period
    returns = np.insert(returns, 0, 0)
    return returns

def calculate_rolling_sharpe(returns, window=60):
    """Calculate rolling Sharpe ratio."""
    # Assuming daily returns, annualize with sqrt(252)
    rolling_mean = pd.Series(returns).rolling(window).mean()
    rolling_std = pd.Series(returns).rolling(window).std()
    sharpe = (rolling_mean / rolling_std) * np.sqrt(252)
    # Handle division by zero
    sharpe = sharpe.fillna(0)
    return sharpe.values

def plot_portfolio_comparison(gat_df, gcn_df, bh_df):
    """Plot 1: Portfolio value comparison over time."""
    plt.figure(figsize=(12, 8))

    if gat_df is not None:
        # Use index as timestep if no explicit timestep column
        x_vals = gat_df.index if 'timestep' not in gat_df.columns else gat_df['timestep']
        plt.plot(x_vals, gat_df['portfolio_value'],
                label='GAT Hybrid', linewidth=2, alpha=0.8)

    if gcn_df is not None:
        x_vals = gcn_df.index if 'timestep' not in gcn_df.columns else gcn_df['timestep']
        plt.plot(x_vals, gcn_df['portfolio_value'],
                label='GCN (Baseline)', linewidth=2, alpha=0.8)

    if bh_df is not None:
        x_vals = bh_df.index if 'timestep' not in bh_df.columns else bh_df['timestep']
        plt.plot(x_vals, bh_df['portfolio_value'],
                label='Buy and Hold', linewidth=2, alpha=0.8, linestyle='--')

    plt.title('Portfolio Value Comparison: GAT Hybrid vs GCN vs Buy and Hold',
              fontsize=16, fontweight='bold')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Portfolio Value (Starting at 1.0)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to better show differences
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '01_portfolio_value_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_drawdown_comparison(gat_df, gcn_df, bh_df):
    """Plot 2: Drawdown comparison over time."""
    plt.figure(figsize=(12, 8))

    methods = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    if gat_df is not None:
        pv_gat = gat_df['portfolio_value'].values
        dd_gat = calculate_drawdown(pv_gat)
        x_vals = gat_df.index if 'timestep' not in gat_df.columns else gat_df['timestep']
        plt.plot(x_vals, dd_gat,
                label='GAT Hybrid', linewidth=2, alpha=0.8, color=colors[0])
        methods.append(('GAT Hybrid', dd_gat))

    if gcn_df is not None:
        pv_gcn = gcn_df['portfolio_value'].values
        dd_gcn = calculate_drawdown(pv_gcn)
        x_vals = gcn_df.index if 'timestep' not in gcn_df.columns else gcn_df['timestep']
        plt.plot(x_vals, dd_gcn,
                label='GCN (Baseline)', linewidth=2, alpha=0.8, color=colors[1])
        methods.append(('GCN', dd_gcn))

    if bh_df is not None:
        pv_bh = bh_df['portfolio_value'].values
        dd_bh = calculate_drawdown(pv_bh)
        x_vals = bh_df.index if 'timestep' not in bh_df.columns else bh_df['timestep']
        plt.plot(x_vals, dd_bh,
                label='Buy and Hold', linewidth=2, alpha=0.8,
                color=colors[2], linestyle='--')
        methods.append(('Buy and Hold', dd_bh))

    plt.title('Drawdown Comparison Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Drawdown', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '02_drawdown_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Also create a summary statistics bar chart for max drawdown
    plt.figure(figsize=(10, 6))
    method_names = []
    max_dds = []

    for label, dd_series in methods:
        method_names.append(label)
        max_dds.append(np.min(dd_series) * 100)  # Convert to percentage

    bars = plt.bar(method_names, max_dds, color=colors[:len(method_names)], alpha=0.8)
    plt.title('Maximum Drawdown Comparison', fontsize=16, fontweight='bold')
    plt.ylabel('Maximum Drawdown (%)', fontsize=12)
    plt.xlabel('Method', fontsize=12)

    # Add value labels on bars
    for bar, dd in zip(bars, max_dds):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{dd:.2f}%', ha='center', va='bottom' if height < 0 else 'top')

    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '02b_max_drawdown_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_return_distribution(gat_df, gcn_df, bh_df):
    """Plot 3: Return distribution comparison."""
    plt.figure(figsize=(12, 8))

    returns_data = []
    labels = []
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    if gat_df is not None:
        returns_gat = calculate_returns(gat_df['portfolio_value'].values)
        returns_data.append(returns_gat)
        labels.append('GAT Hybrid')

    if gcn_df is not None:
        returns_gcn = calculate_returns(gcn_df['portfolio_value'].values)
        returns_data.append(returns_gcn)
        labels.append('GCN (Baseline)')

    if bh_df is not None:
        returns_bh = calculate_returns(bh_df['portfolio_value'].values)
        returns_data.append(returns_bh)
        labels.append('Buy and Hold')

    # Plot histograms
    for i, (returns, label, color) in enumerate(zip(returns_data, labels, colors)):
        plt.hist(returns, bins=50, alpha=0.7, label=label, color=color, density=True)

    plt.title('Return Distribution Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('Periodic Return', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '03_return_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_rolling_sharpe_comparison(gat_df, gcn_df, bh_df):
    """Plot 4: Rolling Sharpe ratio comparison."""
    plt.figure(figsize=(12, 8))

    if gat_df is not None:
        returns_gat = calculate_returns(gat_df['portfolio_value'].values)
        sharpe_gat = calculate_rolling_sharpe(returns_gat, window=30)
        valid_idx = ~np.isnan(sharpe_gat)
        x_vals = gat_df.index if 'timestep' not in gat_df.columns else gat_df['timestep']
        plt.plot(x_vals[valid_idx], sharpe_gat[valid_idx],
                label='GAT Hybrid', linewidth=2, alpha=0.8)

    if gcn_df is not None:
        returns_gcn = calculate_returns(gcn_df['portfolio_value'].values)
        sharpe_gcn = calculate_rolling_sharpe(returns_gcn, window=30)
        valid_idx = ~np.isnan(sharpe_gcn)
        x_vals = gcn_df.index if 'timestep' not in gcn_df.columns else gcn_df['timestep']
        plt.plot(x_vals[valid_idx], sharpe_gcn[valid_idx],
                label='GCN (Baseline)', linewidth=2, alpha=0.8)

    if bh_df is not None:
        returns_bh = calculate_returns(bh_df['portfolio_value'].values)
        sharpe_bh = calculate_rolling_sharpe(returns_bh, window=30)
        valid_idx = ~np.isnan(sharpe_bh)
        x_vals = bh_df.index if 'timestep' not in bh_df.columns else bh_df['timestep']
        plt.plot(x_vals[valid_idx], sharpe_bh[valid_idx],
                label='Buy and Hold', linewidth=2, alpha=0.8, linestyle='--')

    plt.title('Rolling Sharpe Ratio Comparison (30-period window)',
              fontsize=16, fontweight='bold')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Sharpe Ratio', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '04_rolling_sharpe_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_weight_allocation_over_time(gat_path, gcn_path):
    """Plot 5-6: Weight allocation heatmaps for GAT and GCN."""
    # GAT Hybrid weights
    gat_weights = load_weights(gat_path)
    if gat_weights is not None:
        plt.figure(figsize=(14, 10))

        # Assuming weights.csv has columns: date, weight_0, weight_1, ..., weight_n, weight_cash
        # We'll exclude the date column and plot the rest as a heatmap
        weight_cols = [col for col in gat_weights.columns if col != 'date' and col != 'Unnamed: 0']
        if len(weight_cols) > 0:
            # Take a sample of timesteps if too many
            n_timesteps = len(gat_weights)
            if n_timesteps > 100:
                sample_indices = np.linspace(0, n_timesteps-1, 100, dtype=int)
                weights_to_plot = gat_weights.iloc[sample_indices][weight_cols]
                # Convert only weight columns to float
                weights_to_plot = weights_to_plot.astype(float)
                # Get timestep values for sampled indices
                if 'date' in gat_weights.columns:
                    timesteps_to_plot = gat_weights.iloc[sample_indices]['date'].values
                else:
                    timesteps_to_plot = sample_indices
            else:
                weights_to_plot = gat_weights[weight_cols]
                # Convert only weight columns to float
                weights_to_plot = weights_to_plot.astype(float)
                # Get timestep values
                if 'date' in gat_weights.columns:
                    timesteps_to_plot = gat_weights['date'].values
                else:
                    timesteps_to_plot = range(len(gat_weights))

            # Create heatmap
            plt.imshow(weights_to_plot.T, aspect='auto', cmap='RdYlBu_r')
            plt.colorbar(label='Weight')
            plt.title('GAT Hybrid Weight Allocation Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Timestep (Sampled)', fontsize=12)
            plt.ylabel('Assets', fontsize=12)

            # Set timestep ticks
            if len(timesteps_to_plot) > 10:
                tick_indices = np.linspace(0, len(timesteps_to_plot)-1, 10, dtype=int)
                if 'timestep' in gat_weights.columns:
                    plt.xticks(tick_indices, [f'{int(timesteps_to_plot[i])}' for i in tick_indices])
                elif 'date' in gat_weights.columns:
                    plt.xticks(tick_indices, [str(timesteps_to_plot[i])[:10] for i in tick_indices])  # Show date part
                else:
                    plt.xticks(tick_indices, [f'{i}' for i in tick_indices])

            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, '05_gat_weight_allocation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

    # GCN weights
    gcn_weights = load_weights(gcn_path)
    if gcn_weights is not None:
        plt.figure(figsize=(14, 10))

        weight_cols = [col for col in gcn_weights.columns if col != 'date' and col != 'Unnamed: 0']
        if len(weight_cols) > 0:
            n_timesteps = len(gcn_weights)
            if n_timesteps > 100:
                sample_indices = np.linspace(0, n_timesteps-1, 100, dtype=int)
                weights_to_plot = gcn_weights.iloc[sample_indices][weight_cols]
                # Convert only weight columns to float
                weights_to_plot = weights_to_plot.astype(float)
                # Get timestep values for sampled indices
                if 'date' in gcn_weights.columns:
                    timesteps_to_plot = gcn_weights.iloc[sample_indices]['date'].values
                else:
                    timesteps_to_plot = sample_indices
            else:
                weights_to_plot = gcn_weights[weight_cols]
                # Convert only weight columns to float
                weights_to_plot = weights_to_plot.astype(float)
                # Get timestep values
                if 'date' in gcn_weights.columns:
                    timesteps_to_plot = gcn_weights['date'].values
                else:
                    timesteps_to_plot = range(len(gcn_weights))

            plt.imshow(weights_to_plot.T, aspect='auto', cmap='RdYlBu_r')
            plt.colorbar(label='Weight')
            plt.title('GCN Weight Allocation Over Time', fontsize=16, fontweight='bold')
            plt.xlabel('Timestep (Sampled)', fontsize=12)
            plt.ylabel('Assets', fontsize=12)

            if len(timesteps_to_plot) > 10:
                tick_indices = np.linspace(0, len(timesteps_to_plot)-1, 10, dtype=int)
                if 'timestep' in gcn_weights.columns:
                    plt.xticks(tick_indices, [f'{int(timesteps_to_plot[i])}' for i in tick_indices])
                elif 'date' in gcn_weights.columns:
                    plt.xticks(tick_indices, [str(timesteps_to_plot[i])[:10] for i in tick_indices])  # Show date part
                else:
                    plt.xticks(tick_indices, [f'{i}' for i in tick_indices])

            plt.tight_layout()
            plt.savefig(os.path.join(VIS_DIR, '06_gcn_weight_allocation_heatmap.png'), dpi=300, bbox_inches='tight')
            plt.close()

def plot_weight_distribution_comparison(gat_path, gcn_path):
    """Plot 7: Weight distribution comparison (final weights)."""
    plt.figure(figsize=(12, 8))

    gat_weights = load_weights(gat_path)
    gcn_weights = load_weights(gcn_path)

    if gat_weights is not None and gcn_weights is not None:
        # Get final weights (last row)
        gat_final_row = gat_weights.iloc[-1]
        gcn_final_row = gcn_weights.iloc[-1]

        # Extract weight columns (excluding timestep and date)
        exclude_cols = ['timestep', 'date', 'Unnamed: 0']
        gat_weight_cols = [col for col in gat_weights.columns if col not in exclude_cols]
        gcn_weight_cols = [col for col in gcn_weights.columns if col not in exclude_cols]

        # Use common columns or just take what's available
        common_cols = list(set(gat_weight_cols) & set(gcn_weight_cols))
        if len(common_cols) == 0:
            # If no common columns, use first few from each
            gat_weights_final = gat_final_row[gat_weight_cols[:5]].values if len(gat_weight_cols) >= 5 else gat_final_row[gat_weight_cols].values
            gcn_weights_final = gcn_final_row[gcn_weight_cols[:5]].values if len(gcn_weight_cols) >= 5 else gcn_final_row[gcn_weight_cols].values
            labels = [f'Asset {i}' for i in range(len(gat_weights_final))]
        else:
            gat_weights_final = gat_final_row[common_cols].values
            gcn_weights_final = gcn_final_row[common_cols].values
            labels = common_cols

        x = np.arange(len(labels))
        width = 0.35

        plt.bar(x - width/2, gat_weights_final, width, label='GAT Hybrid (Best)', alpha=0.8, color='#1f77b4')
        plt.bar(x + width/2, gcn_weights_final, width, label='GCN', alpha=0.8, color='#ff7f0e')

        plt.title('Final Weight Allocation Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Assets', fontsize=12)
        plt.ylabel('Weight', fontsize=12)
        plt.xticks(x, labels, rotation=45 if len(labels) > 5 else 0)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, '07_weight_distribution_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_turnover_analysis(gat_path, gcn_path):
    """Plot 8: Turnover analysis over time."""
    plt.figure(figsize=(12, 8))

    gat_weights = load_weights(gat_path)
    gcn_weights = load_weights(gcn_path)

    def calculate_turnover(weights_df):
        if weights_df is None or len(weights_df) < 2:
            return None
        # Exclude non-weight columns (timestep, date, Unnamed: 0)
        exclude_cols = ['timestep', 'date', 'Unnamed: 0']
        weight_cols = [col for col in weights_df.columns if col not in exclude_cols]
        if len(weight_cols) == 0:
            return None

        weights_only = weights_df[weight_cols].values
        # Calculate turnover as sum of absolute weight changes
        weight_changes = np.diff(weights_only, axis=0)
        turnover = np.sum(np.abs(weight_changes), axis=1)
        # Prepend 0 for first timestep
        turnover = np.insert(turnover, 0, 0)
        return turnover

    gat_turnover = calculate_turnover(gat_weights)
    gcn_turnover = calculate_turnover(gcn_weights)

    if gat_turnover is not None:
        # Get timestep values for x-axis
        if 'date' in gat_weights.columns:
            timesteps_gat = gat_weights['date'].values
        else:
            timesteps_gat = range(len(gat_turnover))
        plt.plot(timesteps_gat, gat_turnover, label='GAT Hybrid (Best)', linewidth=2, alpha=0.8, color='#1f77b4')

    if gcn_turnover is not None:
        # Get timestep values for x-axis
        if 'date' in gcn_weights.columns:
            timesteps_gcn = gcn_weights['date'].values
        else:
            timesteps_gcn = range(len(gcn_turnover))
        plt.plot(timesteps_gcn, gcn_turnover, label='GCN', linewidth=2, alpha=0.8, color='#ff7f0e')

    plt.title('Turnover Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Timestep', fontsize=12)
    plt.ylabel('Turnover (Sum of |Δweights|)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '08_turnover_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Also create average turnover bar chart
    plt.figure(figsize=(10, 6))
    methods = []
    avg_turnovers = []
    colors = ['#1f77b4', '#ff7f0e']

    if gat_turnover is not None:
        methods.append('GAT Hybrid (Best)')
        avg_turnovers.append(np.mean(gat_turnover))

    if gcn_turnover is not None:
        methods.append('GCN')
        avg_turnovers.append(np.mean(gcn_turnover))

    if len(methods) > 0:
        bars = plt.bar(methods, avg_turnovers, color=colors[:len(methods)], alpha=0.8)
        plt.title('Average Turnover Comparison', fontsize=16, fontweight='bold')
        plt.ylabel('Average Turnover', fontsize=12)
        plt.xlabel('Method', fontsize=12)

        for bar, avg in zip(bars, avg_turnovers):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{avg:.4f}', ha='center', va='bottom')

        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, '08b_average_turnover_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_risk_return_scatter(gat_df, gcn_df, bh_df):
    """Plot 9: Risk-return scatter plot."""
    plt.figure(figsize=(10, 8))

    methods = []
    returns = []
    risks = []  # volatility (std of returns)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    if gat_df is not None:
        returns_gat = calculate_returns(gat_df['portfolio_value'].values)
        # Annualize return and volatility (assuming daily data)
        ann_return = np.mean(returns_gat) * 252
        ann_volatility = np.std(returns_gat) * np.sqrt(252)
        methods.append('GAT Hybrid (Best)')
        returns.append(ann_return)
        risks.append(ann_volatility)

    if gcn_df is not None:
        returns_gcn = calculate_returns(gcn_df['portfolio_value'].values)
        ann_return = np.mean(returns_gcn) * 252
        ann_volatility = np.std(returns_gcn) * np.sqrt(252)
        methods.append('GCN')
        returns.append(ann_return)
        risks.append(ann_volatility)

    if bh_df is not None:
        returns_bh = calculate_returns(bh_df['portfolio_value'].values)
        ann_return = np.mean(returns_bh) * 252
        ann_volatility = np.std(returns_bh) * np.sqrt(252)
        methods.append('Buy and Hold')
        returns.append(ann_return)
        risks.append(ann_volatility)

    if len(methods) > 0:
        scatter = plt.scatter(risks, returns, c=colors[:len(methods)], s=100, alpha=0.8, edgecolors='black')

        # Add labels for each point
        for i, method in enumerate(methods):
            plt.annotate(method, (risks[i], returns[i]),
                        xytext=(5, 5), textcoords='offset points', fontsize=10)

        plt.title('Risk-Return Profile Comparison', fontsize=16, fontweight='bold')
        plt.xlabel('Annualized Volatility (Risk)', fontsize=12)
        plt.ylabel('Annualized Return', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(VIS_DIR, '09_risk_return_scatter.png'), dpi=300, bbox_inches='tight')
        plt.close()

def plot_performance_summary_bar(gat_df, gcn_df, bh_df):
    """Plot 10: Performance summary bar chart (final value, Sharpe, etc.)."""
    # Load metrics for accurate numbers
    # CORRECTED: GAT Hybrid is original_5feat_seed123_20k, GCN is final_original_gat_5feat_100k
    gat_metrics = load_metrics(os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k"))
    gcn_metrics = load_metrics(os.path.join(RESULTS_DIR, "ppo_eval/final_original_gat_5feat_100k"))

    # Compute Buy and Hold metrics if data available
    bh_metrics = None
    bh_df_temp = compute_buy_and_hold()
    if bh_df_temp is not None:
        returns_bh = calculate_returns(bh_df_temp['portfolio_value'].values)
        pv_bh = bh_df_temp['portfolio_value'].values

        # Calculate metrics similar to what's in the JSON files
        total_return = pv_bh[-1] - 1.0  # Since we start at 1.0
        ann_return = np.mean(returns_bh) * 252
        ann_volatility = np.std(returns_bh) * np.sqrt(252)
        sharpe = ann_return / ann_volatility if ann_volatility > 0 else 0

        # Calculate max drawdown
        dd_bh = calculate_drawdown(pv_bh)
        max_dd = np.min(dd_bh)

        bh_metrics = {
            'final_portfolio_value': pv_bh[-1],
            'annualized_return': ann_return,
            'sharpe_ratio': sharpe,
            'maximum_drawdown': max_dd
        }

    plt.figure(figsize=(14, 10))

    # Prepare data for subplots
    metrics_to_plot = [
        ('final_portfolio_value', 'Final Portfolio Value', 'Value'),
        ('annualized_return', 'Annualized Return', 'Return'),
        ('sharpe_ratio', 'Sharpe Ratio', 'Ratio'),
        ('maximum_drawdown', 'Maximum Drawdown', 'Drawdown')
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()

    method_names = []
    method_colors = []

    if gat_metrics is not None:
        method_names.append('GAT Hybrid')
        method_colors.append('#1f77b4')

    if gcn_metrics is not None:
        method_names.append('GCN (Baseline)')
        method_colors.append('#ff7f0e')

    if bh_metrics is not None:
        method_names.append('Buy and Hold')
        method_colors.append('#2ca02c')

    if len(method_names) == 0:
        print("No metrics data available for summary plot")
        return

    for idx, (metric_key, title, ylabel) in enumerate(metrics_to_plot):
        ax = axes[idx]

        values = []
        for method_metrics in [gat_metrics, gcn_metrics, bh_metrics]:
            if method_metrics is not None and metric_key in method_metrics:
                val = method_metrics[metric_key]
                # For drawdown, show as positive percentage for clarity
                if metric_key == 'maximum_drawdown':
                    val = abs(val) * 100  # Convert to positive percentage
                elif metric_key in ['annualized_return']:
                    val = val * 100  # Convert to percentage
                values.append(val)
            else:
                values.append(0)  # Default if missing

        bars = ax.bar(method_names, values, color=method_colors[:len(method_names)], alpha=0.8)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            if height != 0:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{val:.2f}', ha='center', va='bottom' if height >= 0 else 'top')

    plt.suptitle('Performance Metrics Summary Comparison', fontsize=18, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(VIS_DIR, '10_performance_summary_bars.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Main function to generate all visualizations."""
    print("Generating comparative visualizations for GAT Hybrid, GCN, and Buy and Hold...")
    print(f"Output directory: {VIS_DIR}")
    print(f"Base directory: {BASE_DIR}")
    print(f"Results directory: {RESULTS_DIR}")
    print(f"Data directory: {DATA_PATH}")

    # Load data
    print("\nLoading portfolio data...")
    # CORRECTED LABELING:
    # final_original_gat_5feat_100k = GCN baseline (original architecture)
    # original_5feat_seed123_20k = GAT Hybrid (our improved architecture)
    gcn_df = load_portfolio_values(os.path.join(RESULTS_DIR, "ppo_eval/final_original_gat_5feat_100k"))
    gat_df = load_portfolio_values(os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k"))
    bh_df = compute_buy_and_hold()

    print(f"GAT Hybrid data: {'Loaded' if gat_df is not None else 'Failed'}")
    print(f"GCN data: {'Loaded' if gcn_df is not None else 'Failed'}")
    print(f"Buy and Hold data: {'Loaded' if bh_df is not None else 'Failed'}")

    if gat_df is not None:
        print(f"GAT Hybrid rows: {len(gat_df)}")
    if gcn_df is not None:
        print(f"GCN rows: {len(gcn_df)}")
    if bh_df is not None:
        print(f"Buy and Hold rows: {len(bh_df)}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # Plot 1: Portfolio value comparison
    print("  1. Portfolio value comparison...")
    plot_portfolio_comparison(gat_df, gcn_df, bh_df)

    # Plot 2: Drawdown comparison
    print("  2. Drawdown comparison...")
    plot_drawdown_comparison(gat_df, gcn_df, bh_df)

    # Plot 3: Return distribution
    print("  3. Return distribution comparison...")
    plot_return_distribution(gat_df, gcn_df, bh_df)

    # Plot 4: Rolling Sharpe ratio
    print("  4. Rolling Sharpe ratio comparison...")
    plot_rolling_sharpe_comparison(gat_df, gcn_df, bh_df)

    # Plots 5-6: Weight allocation heatmaps
    print("  5-6. Weight allocation heatmaps...")
    plot_weight_allocation_over_time(
        os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k"),  # GAT Hybrid
        os.path.join(RESULTS_DIR, "ppo_eval/final_original_gat_5feat_100k")   # GCN
    )

    # Plot 7: Weight distribution comparison
    print("  7. Weight distribution comparison...")
    plot_weight_distribution_comparison(
        os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k"),  # GAT Hybrid
        os.path.join(RESULTS_DIR, "ppo_eval/final_original_gat_5feat_100k")   # GCN
    )

    # Plot 8: Turnover analysis
    print("  8. Turnover analysis...")
    plot_turnover_analysis(
        os.path.join(RESULTS_DIR, "ppo_eval/original_5feat_seed123_20k"),  # GAT Hybrid
        os.path.join(RESULTS_DIR, "ppo_eval/final_original_gat_5feat_100k")   # GCN
    )

    # Plot 9: Risk-return scatter
    print("  9. Risk-return scatter plot...")
    plot_risk_return_scatter(gat_df, gcn_df, bh_df)

    # Plot 10: Performance summary bars
    print("  10. Performance summary bars...")
    plot_performance_summary_bar(gat_df, gcn_df, bh_df)

    print("\nAll visualizations generated successfully!")
    print("Output saved to:", VIS_DIR)

    # List generated files
    generated_files = [f for f in os.listdir(VIS_DIR) if f.endswith('.png')]
    print("Generated", len(generated_files), "visualization files:")
    for f in sorted(generated_files):
        print("   -", f)

if __name__ == "__main__":
    main()