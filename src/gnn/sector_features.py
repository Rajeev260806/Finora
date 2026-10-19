"""
Sector-relative feature computation for GNN node features.
Computes sector-relative signals that are more predictable than raw returns.
"""

import os
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List
import torch
from src.gnn.temporal_graph_dataset import TemporalGraphDataset


def load_returns_data(data_path: str) -> pd.DataFrame:
    """
    Load returns data for feature computation.

    Args:
        data_path: Path to the parquet file containing returns data

    Returns:
        DataFrame with returns (index=date, columns=ticker)
    """
    data = pd.read_parquet(data_path)
    returns = data["Returns Daily"].unstack(level="Ticker")
    returns = returns.fillna(0.0)
    returns = returns.sort_index(axis=1)
    return returns


def compute_sector_relative_features(
    returns_df: pd.DataFrame,
    sector_mapping: Dict[str, int],
    feature_names: List[str] = ['returns_daily', 'volatility_20d', 'volume_normalized', 'volatility_annual', 'log_return']
) -> pd.DataFrame:
    """
    Compute sector-relative features for each stock.

    Args:
        returns_df: DataFrame with returns (index=date, columns=ticker)
        sector_mapping: Dictionary mapping ticker to sector ID
        feature_names: List of feature column names in returns_df

    Returns:
        DataFrame with sector-relative features
    """
    # Create sector DataFrames
    sector_returns = {}
    for sector_id in set(sector_mapping.values()):
        sector_tickers = [ticker for ticker, sec_id in sector_mapping.items() if sec_id == sector_id]
        if sector_tickers:
            sector_returns[sector_id] = returns_df[sector_tickers].mean(axis=1)

    # Compute sector-relative features
    sector_relative_df = returns_df.copy()

    for ticker in returns_df.columns:
        sector_id = sector_mapping.get(ticker, -1)
        if sector_id in sector_returns:
            sector_mean = sector_returns[sector_id]
            # Compute sector-relative returns (stock return minus sector average)
            sector_relative_df[f'{ticker}_sector_relative'] = returns_df[ticker] - sector_mean
        else:
            # If no sector data, set to zero
            sector_relative_df[f'{ticker}_sector_relative'] = 0.0

    # Also compute cross-sectional rankings within sectors
    for date in returns_df.index:
        date_returns = returns_df.loc[date]
        # Group by sector and compute ranks
        sector_groups = {}
        for ticker in date_returns.index:
            sector_id = sector_mapping.get(ticker, -1)
            if sector_id not in sector_groups:
                sector_groups[sector_id] = []
            sector_groups[sector_id].append((ticker, date_returns[ticker]))

        # Compute percentile ranks within each sector
        for sector_id, ticker_returns in sector_groups.items():
            if len(ticker_returns) > 1:
                returns_only = [ret for _, ret in ticker_returns]
                # Compute ranks (0 to 1, where 0 is worst performer, 1 is best)
                sorted_returns = sorted(returns_only)
                for ticker, ret in ticker_returns:
                    rank = sorted_returns.index(ret) / (len(sorted_returns) - 1) if len(sorted_returns) > 1 else 0.5
                    sector_relative_df.loc[date, f'{ticker}_sector_rank'] = rank
            else:
                # Single stock in sector
                ticker = ticker_returns[0][0]
                sector_relative_df.loc[date, f'{ticker}_sector_rank'] = 0.5

    return sector_relative_df


def create_enhanced_node_features(
    base_features: np.ndarray,
    sector_relative_returns: np.ndarray,
    sector_relative_volatility: np.ndarray,
    sector_rankings: np.ndarray,
    sector_ids: np.ndarray
) -> np.ndarray:
    """
    Combine base features with sector-relative features.

    Args:
        base_features: Original node features (n_assets, 5)
        sector_relative_returns: Sector-relative returns (n_assets,)
        sector_relative_volatility: Sector-relative volatility (n_assets,)
        sector_rankings: Cross-sectional rankings within sector (n_assets,)
        sector_ids: Sector IDs for each asset (n_assets,)

    Returns:
        Enhanced node features (n_assets, 9) - original 5 + 4 new features
    """
    # Stack the features: [original_5, sector_rel_return, sector_rel_vol, sector_rank, sector_id_normalized]
    sector_ids_normalized = sector_ids / (sector_ids.max() + 1e-8)  # Normalize to 0-1

    enhanced_features = np.column_stack([
        base_features,
        sector_relative_returns,
        sector_relative_volatility,
        sector_rankings,
        sector_ids_normalized
    ])

    return enhanced_features


def compute_sector_statistics(
    returns_df: pd.DataFrame,
    sector_mapping: Dict[str, int],
    volatility_20d: pd.DataFrame = None,
    volume_normalized: pd.DataFrame = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute sector-relative statistics for all stocks at the latest date.

    Args:
        returns_df: DataFrame with returns (index=date, columns=ticker)
        sector_mapping: Dictionary mapping ticker to sector ID
        volatility_20d: DataFrame with 20-day volatility (optional)
        volume_normalized: DataFrame with normalized volume (optional)

    Returns:
        Tuple of (sector_relative_returns, sector_relative_volatility, sector_rankings)
    """
    # Get latest date
    latest_date = returns_df.index[-1]
    latest_returns = returns_df.loc[latest_date]

    # Compute sector means for latest date
    sector_means = {}
    sector_counts = {}
    for ticker, sector_id in sector_mapping.items():
        if ticker in latest_returns.index:
            if sector_id not in sector_means:
                sector_means[sector_id] = 0.0
                sector_counts[sector_id] = 0
            sector_means[sector_id] += latest_returns[ticker]
            sector_counts[sector_id] += 1

    # Calculate averages
    for sector_id in sector_means:
        if sector_counts[sector_id] > 0:
            sector_means[sector_id] /= sector_counts[sector_id]
        else:
            sector_means[sector_id] = 0.0

    # Compute sector-relative returns
    sector_relative_returns = np.zeros(len(sector_mapping))
    sector_relative_volatility = np.zeros(len(sector_mapping))
    sector_rankings = np.zeros(len(sector_mapping))

    ticker_list = list(sector_mapping.keys())
    for i, ticker in enumerate(ticker_list):
        sector_id = sector_mapping[ticker]
        if ticker in latest_returns.index and sector_id in sector_means:
            # Sector-relative return
            sector_relative_returns[i] = latest_returns[ticker] - sector_means[sector_id]

            # Sector-relative volatility (if available)
            if volatility_20d is not None and ticker in volatility_20d.index:
                sector_vols = []
                for t, sid in sector_mapping.items():
                    if sid == sector_id and t in volatility_20d.index:
                        sector_vols.append(volatility_20d.loc[latest_date, t])
                if sector_vols:
                    sector_mean_vol = np.mean(sector_vols)
                    sector_relative_volatility[i] = volatility_20d.loc[latest_date, ticker] - sector_mean_vol

            # Cross-sectional ranking within sector
            sector_tickers = [t for t, sid in sector_mapping.items() if sid == sector_id]
            if len(sector_tickers) > 1:
                sector_returns = [latest_returns[t] for t in sector_tickers if t in latest_returns.index]
                if len(sector_returns) > 1:
                    ticker_return = latest_returns[ticker] if ticker in latest_returns.index else 0.0
                    sorted_returns = sorted(sector_returns)
                    try:
                        rank = sorted_returns.index(ticker_return) / (len(sorted_returns) - 1)
                    except ValueError:
                        rank = 0.5  # Default if not found
                    sector_rankings[i] = rank
                else:
                    sector_rankings[i] = 0.5
            else:
                sector_rankings[i] = 0.5  # Single stock in sector

    return sector_relative_returns, sector_relative_volatility, sector_rankings


if __name__ == "__main__":
    # Example usage
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    RETURNS_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")

    # Load returns data
    returns_df = load_returns_data(RETURNS_DATA_PATH)
    print(f"Returns data shape: {returns_df.shape}")

    # Create dummy sector mapping for testing
    tickers = returns_df.columns.tolist()
    sector_mapping = {ticker: i % 5 for i, ticker in enumerate(tickers)}  # 5 sectors

    # Compute sector-relative features
    sector_rel_returns, sector_rel_vol, sector_ranks = compute_sector_statistics(
        returns_df, sector_mapping
    )

    print(f"Sector-relative returns shape: {sector_rel_returns.shape}")
    print(f"Sample sector-relative returns: {sector_rel_returns[:5]}")
    print(f"Sample sector-relative volatility: {sector_rel_vol[:5]}")
    print(f"Sample sector rankings: {sector_ranks[:5]}")