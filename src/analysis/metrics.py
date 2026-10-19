"""
Shared Metrics Computation Module for Finora Portfolio Analysis
================================================================
Provides standard, rigorous financial performance metrics used across all
strategies (baselines and RL agents) to ensure 100% comparability.

Definitions:
- Cumulative Return: V_final / V_initial - 1
- Annualized Return: (1 + Cumulative Return) ** (252 / N) - 1
- Annualized Volatility: std(daily_returns, ddof=1) * sqrt(252)
- Annualized Sharpe Ratio: (mean(daily_returns) / std(daily_returns, ddof=1)) * sqrt(252) (assuming 0% risk-free rate)
- Max Drawdown: min_t (V_t / max_{s <= t} V_s - 1)
- Average Daily Turnover (One-Way): mean_t (0.5 * sum_i |w_{i,t} - w_{i,t-1}|)
  Note: One-way turnover represents the fraction of portfolio value reallocated per day.
"""

from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def compute_turnover(weights: np.ndarray) -> np.ndarray:
    """
    Computes daily one-way portfolio turnover:
        turnover_t = 0.5 * sum_i |w_{i,t} - w_{i,t-1}|

    Parameters
    ----------
    weights : np.ndarray
        Array of shape (T, N) or (T, N+1) representing portfolio weights over time.

    Returns
    -------
    turnovers : np.ndarray
        Array of shape (T-1,) of daily one-way turnovers.
    """
    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[0] < 2:
        return np.array([0.0], dtype=np.float64)
    delta_w = np.abs(weights[1:] - weights[:-1])
    # 0.5 factor corresponds to one-way turnover (purchases or sales)
    return 0.5 * np.sum(delta_w, axis=1)


def compute_metrics(
    portfolio_values: np.ndarray | pd.Series,
    daily_returns: Optional[np.ndarray | pd.Series] = None,
    weights: Optional[np.ndarray | pd.DataFrame] = None,
    turnover: Optional[np.ndarray | pd.Series | float] = None,
    risk_free_rate: float = 0.0,
    annualization_factor: int = 252,
) -> Dict[str, float]:
    """
    Computes standardized financial metrics.

    Parameters
    ----------
    portfolio_values : array-like
        Time series of portfolio values (starting at V_0 > 0).
    daily_returns : array-like, optional
        Daily returns. If None, derived from portfolio_values: (V_t / V_{t-1}) - 1.
    weights : array-like, optional
        Weights time series to compute turnover if turnover is not directly passed.
    turnover : array-like or float, optional
        Precomputed turnover series or float average turnover.
    risk_free_rate : float, default=0.0
        Annualized risk-free rate.
    annualization_factor : int, default=252
        Trading days per year.

    Returns
    -------
    metrics : dict
        Standardized metrics dictionary.
    """
    pv = np.asarray(portfolio_values, dtype=np.float64).flatten()
    if len(pv) < 2:
        raise ValueError("At least 2 portfolio value points are required to compute metrics.")

    # 1. Daily returns
    if daily_returns is None:
        ret = pv[1:] / pv[:-1] - 1.0
    else:
        ret = np.asarray(daily_returns, dtype=np.float64).flatten()
        # Clean potential non-finite values
        ret = ret[np.isfinite(ret)]

    n_days = len(ret)
    if n_days == 0:
        raise ValueError("Daily returns series is empty.")

    # 2. Cumulative return and Final Portfolio Value
    initial_val = float(pv[0])
    final_val = float(pv[-1])
    cumulative_return = (final_val / initial_val) - 1.0

    # 3. Annualized Return
    # Using compound annual growth rate (CAGR): (1 + R)^(252 / N) - 1
    if cumulative_return > -1.0:
        annualized_return = (1.0 + cumulative_return) ** (annualization_factor / n_days) - 1.0
    else:
        annualized_return = -1.0

    # 4. Volatility and Sharpe Ratio
    if n_days > 1:
        daily_std = float(np.std(ret, ddof=1))
        annualized_volatility = daily_std * np.sqrt(annualization_factor)
        daily_rf = (1.0 + risk_free_rate) ** (1.0 / annualization_factor) - 1.0
        excess_returns = ret - daily_rf
        if daily_std > 1e-12:
            sharpe_ratio = float(np.mean(excess_returns) / daily_std * np.sqrt(annualization_factor))
        else:
            sharpe_ratio = 0.0
    else:
        annualized_volatility = 0.0
        sharpe_ratio = 0.0

    # 5. Maximum Drawdown
    running_max = np.maximum.accumulate(pv)
    drawdowns = (pv - running_max) / running_max
    max_drawdown = float(np.min(drawdowns))

    # 6. Average Turnover
    if turnover is not None:
        if isinstance(turnover, (int, float, np.floating)):
            avg_turnover = float(turnover)
        else:
            to_arr = np.asarray(turnover, dtype=np.float64)
            avg_turnover = float(np.mean(to_arr)) if len(to_arr) > 0 else 0.0
    elif weights is not None:
        w_arr = np.asarray(weights, dtype=np.float64)
        to_arr = compute_turnover(w_arr)
        avg_turnover = float(np.mean(to_arr)) if len(to_arr) > 0 else 0.0
    else:
        avg_turnover = 0.0

    return {
        "Final Portfolio Value": final_val,
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Volatility": annualized_volatility,
        "Max Drawdown": max_drawdown,
        "Average Turnover": avg_turnover,
    }


def compute_metrics_from_history(history_df: pd.DataFrame, risk_free_rate: float = 0.0) -> Dict[str, float]:
    """
    Convenience function to compute metrics directly from an environment history DataFrame.
    """
    pv = history_df["portfolio_value"].values
    
    # Check for daily return column
    if "portfolio_return" in history_df.columns:
        ret = history_df["portfolio_return"].dropna().values
    elif "net_return" in history_df.columns:
        ret = history_df["net_return"].dropna().values
    elif "raw_return" in history_df.columns:
        ret = history_df["raw_return"].dropna().values
    else:
        ret = None

    turnover = None
    if "turnover" in history_df.columns:
        turnover = history_df["turnover"].values

    return compute_metrics(portfolio_values=pv, daily_returns=ret, turnover=turnover, risk_free_rate=risk_free_rate)
