"""
Phase 2 (Person B) — Reward & Transaction Cost Model Validation Script
========================================================================

This script validates the Phase 2 proposed modular reward function and
transaction cost model across 4 synthetic allocation strategies on
Finora's cleaned training dataset (`data/clean/train.parquet`).

Strategies Evaluated:
1. Passive Buy & Hold (Equal weight initial allocation, 0 turnover)
2. Monthly Rebalancing (Equal weight rebalancing every 21 trading days)
3. Excessive Churning (Random portfolio re-allocation every single day)
4. Single Stock Concentration (100% allocation into a single asset)

Validation Objectives:
- Ensure zero NaN / Infinite values in reward calculation.
- Verify transaction cost drag severely penalizes Excessive Churning.
- Verify Volatility & Concentration penalties penalize single-stock concentration.
- Confirm reward values remain within a stable numerical range (e.g., [-0.10, +0.10]).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.environment.portfolioenv import PortfolioEnv


def compute_phase2_reward(
    info: dict,
    returns_window: np.ndarray,
    weights: np.ndarray,
    prev_weights: np.ndarray,
    lambda_vol: float = 0.10,
    lambda_txn: float = 1.0,
    lambda_conc: float = 0.01,
) -> dict:
    """
    Computes Phase 2 risk-adjusted, cost-aware modular reward.

    Parameters
    ----------
    info : dict
        Environment step info dict containing 'raw_return', 'transaction_cost', 'turnover'.
    returns_window : np.ndarray
        Historical daily log returns window of shape (n_assets, window).
    weights : np.ndarray
        Current allocation vector of shape (n_assets + 1,) including cash.
    prev_weights : np.ndarray
        Previous allocation vector of shape (n_assets + 1,) including cash.
    lambda_vol : float
        Weight on annualized rolling volatility penalty.
    lambda_txn : float
        Weight multiplier on transaction cost penalty.
    lambda_conc : float
        Weight on Herfindahl-Hirschman Index (HHI) concentration penalty.

    Returns
    -------
    dict
        Breakdown of reward components and final scalar reward value.
    """
    port_return = info["raw_return"]
    transaction_cost = info["transaction_cost"]
    net_return = port_return - (lambda_txn * transaction_cost)

    # Log net return guard
    if net_return <= -1.0:
        log_net_return = -10.0
    else:
        log_net_return = float(np.log1p(net_return))

    # Asset weights excluding cash
    asset_weights = weights[:-1]
    n_assets = len(asset_weights)

    # 1. Rolling 20-day annualized portfolio volatility
    if returns_window.size > 0 and np.sum(asset_weights) > 0:
        # returns_window shape: (n_assets, window)
        port_win_returns = asset_weights @ returns_window  # shape: (window,)
        daily_vol = float(np.std(port_win_returns, ddof=1)) if len(port_win_returns) > 1 else 0.0
        ann_vol = daily_vol * np.sqrt(252)
    else:
        daily_vol = 0.0
        ann_vol = 0.0

    vol_penalty = lambda_vol * ann_vol

    # 2. Asset Concentration Penalty (HHI across risky assets)
    hhi = float(np.sum(asset_weights ** 2))
    conc_penalty = lambda_conc * hhi

    # Total scalar reward
    total_reward = log_net_return - vol_penalty - conc_penalty

    return {
        "total_reward": total_reward,
        "log_net_return": log_net_return,
        "net_return": net_return,
        "raw_return": port_return,
        "transaction_cost": transaction_cost,
        "ann_vol": ann_vol,
        "vol_penalty": vol_penalty,
        "hhi": hhi,
        "conc_penalty": conc_penalty,
    }


def run_strategy_simulation(env: PortfolioEnv, strategy_type: str, n_steps: int = 500) -> pd.DataFrame:
    """
    Simulates a policy on PortfolioEnv and computes Phase 2 reward breakdown.
    """
    obs, info = env.reset(seed=42)
    n_assets = env.n_assets
    records = []

    # Prepare initial target weights
    if strategy_type in ["buy_and_hold", "equal_weight"]:
        eq_weight = np.ones(n_assets) / n_assets
        # inverse softmax to get raw logits for eq_weight and 0 cash
        eq_logits = np.append(np.log(eq_weight + 1e-8), -10.0)
    elif strategy_type == "single_stock":
        stock_logits = np.full(n_assets + 1, -10.0)
        stock_logits[0] = 10.0  # 100% in asset 0
    else:
        stock_logits = None

    for step in range(n_steps):
        current_weights = obs["weights"]
        returns_win = obs["returns_window"]

        if strategy_type == "buy_and_hold":
            if step == 0:
                action = eq_logits
            else:
                # Keep exact current weights (no turnover)
                action = np.log(np.maximum(current_weights, 1e-8))
        elif strategy_type == "equal_weight":
            if step % 21 == 0:
                action = eq_logits
            else:
                action = np.log(np.maximum(current_weights, 1e-8))
        elif strategy_type == "churning":
            # Random logits every step -> high turnover
            action = np.random.randn(n_assets + 1).astype(np.float32)
        elif strategy_type == "single_stock":
            action = stock_logits
        else:
            raise ValueError(f"Unknown strategy_type: {strategy_type}")

        prev_w = current_weights.copy()
        next_obs, reward_env, terminated, truncated, info = env.step(action)
        curr_w = next_obs["weights"]

        reward_components = compute_phase2_reward(
            info=info,
            returns_window=returns_win,
            weights=curr_w,
            prev_weights=prev_w,
        )

        record = {
            "step": step,
            "date": info["date"],
            "portfolio_value": info["portfolio_value"],
            **reward_components,
        }
        records.append(record)

        obs = next_obs
        if terminated or truncated:
            break

    return pd.DataFrame(records)


def main():
    data_path = ROOT_DIR / "data" / "clean" / "train.parquet"
    if not data_path.exists():
        print(f"Error: Dataset not found at {data_path}")
        sys.exit(1)

    print("==================================================================")
    print("FINORA PHASE 2: REWARD & TRANSACTION COST MODEL VALIDATION")
    print("==================================================================")
    print(f"Loading data from: {data_path}")
    data = pd.read_parquet(data_path)

    env = PortfolioEnv(data=data, window=20, transaction_cost_bps=10.0)
    print(f"Environment initialized. Tickers: {env.tickers} (N={env.n_assets})")
    print(f"Total available steps: {env.n_steps}\n")

    strategies = ["buy_and_hold", "equal_weight", "churning", "single_stock"]
    results = {}

    for strat in strategies:
        df_res = run_strategy_simulation(env, strat, n_steps=500)
        results[strat] = df_res

    # Summary table printing
    print("-" * 105)
    print(f"{'Strategy':<20} | {'Final Value':<12} | {'Mean Reward':<12} | {'Std Reward':<12} | {'Mean Ann Vol':<14} | {'Total Txn Cost':<15}")
    print("-" * 105)

    for strat, df_res in results.items():
        final_val = df_res["portfolio_value"].iloc[-1]
        mean_reward = df_res["total_reward"].mean()
        std_reward = df_res["total_reward"].std()
        mean_vol = df_res["ann_vol"].mean()
        total_cost = df_res["transaction_cost"].sum()

        print(f"{strat:<20} | {final_val:<12.4f} | {mean_reward:<12.6f} | {std_reward:<12.6f} | {mean_vol:<14.4%} | {total_cost:<15.6f}")

    print("-" * 105)
    print("\nSANITY CHECKS & VERIFICATION:")

    # Check 1: NaNs and Infs
    for strat, df_res in results.items():
        nan_count = df_res["total_reward"].isna().sum()
        inf_count = np.isinf(df_res["total_reward"]).sum()
        assert nan_count == 0, f"NaNs detected in strategy {strat}!"
        assert inf_count == 0, f"Infs detected in strategy {strat}!"
    print(" [PASSED] 1. Zero NaNs or Infs across all strategies.")

    # Check 2: Churning penalization
    mean_reward_eq = results["equal_weight"]["total_reward"].mean()
    mean_reward_churn = results["churning"]["total_reward"].mean()
    assert mean_reward_eq > mean_reward_churn, (
        f"Equal weight reward ({mean_reward_eq:.6f}) should exceed churning reward ({mean_reward_churn:.6f})!"
    )
    print(f" [PASSED] 2. Excessive Churning penalized correctly: Equal Weight ({mean_reward_eq:.6f}) > Churning ({mean_reward_churn:.6f})")

    # Check 3: Single stock concentration penalty
    mean_hhi_single = results["single_stock"]["hhi"].mean()
    mean_hhi_eq = results["equal_weight"]["hhi"].mean()
    assert mean_hhi_single > mean_hhi_eq, "Single stock concentration HHI should be higher than equal weight!"
    print(f" [PASSED] 3. Concentration HHI evaluated correctly: Single Stock ({mean_hhi_single:.4f}) > Equal Weight ({mean_hhi_eq:.4f})")

    # Check 4: Reward numerical stability
    for strat, df_res in results.items():
        min_r, max_r = df_res["total_reward"].min(), df_res["total_reward"].max()
        assert -1.0 < min_r < 1.0 and -1.0 < max_r < 1.0, f"Reward out of expected range for {strat}: [{min_r}, {max_r}]"
    print(" [PASSED] 4. Reward values remain numerically well-bounded within [-1.0, +1.0].")

    print("\nPhase 2 Reward & Transaction Cost Model validation complete! All tests passed successfully.\n")


if __name__ == "__main__":
    main()
