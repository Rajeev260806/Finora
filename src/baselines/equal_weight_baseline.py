"""
Equal-weight baseline — companion to validate_env.py.

Purpose: a random policy re-trades almost completely every day, so its
portfolio-value decay mostly reflects transaction costs, not whether
the underlying return math is correct. An equal-weight policy barely
trades after the first step (weights stay ~constant), so its value
path isolates "is the return calculation right?" from "is turnover
cost realistic?" — this is the check the Phase 1 checklist asks for.

Usage:
    python equal_weight_baseline.py data/clean/full_dataset.parquet
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.environment.portfolioenv import PortfolioEnv


def run_equal_weight_episode(env: PortfolioEnv) -> pd.DataFrame:
    obs, info = env.reset(seed=0)
    # equal weight across all assets, ~0 in cash — same logits every step
    # (softmax of all-equal logits = uniform weights)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[-1] = -10.0  # push cash logit down so ~all weight goes to assets
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_history_df()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python equal_weight_baseline.py <path to full_dataset.parquet>")

    data = pd.read_parquet(sys.argv[1])
    env = PortfolioEnv(data, window=20, transaction_cost_bps=10.0)
    history = run_equal_weight_episode(env)

    print(f"Final portfolio value (equal-weight): {history['portfolio_value'].iloc[-1]:.4f}")
    print(f"Mean daily raw_return: {history['raw_return'].mean():.6f}")
    print(f"Std daily raw_return: {history['raw_return'].std():.6f}")
    print(f"Mean turnover after step 1: {history['turnover'].iloc[1:].mean():.6f}")
    print(f"Total transaction cost paid: {history['transaction_cost'].sum():.6f}")

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history.index, history["portfolio_value"])
    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Equal-weight baseline — portfolio value")
    ax.set_ylabel("Portfolio value (start = 1.0)")
    fig.tight_layout()
    fig.savefig("equal_weight_baseline.png", dpi=150)
    print("Plot saved to equal_weight_baseline.png")