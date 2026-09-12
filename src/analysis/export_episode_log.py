"""
Export a sample episode log for Person C.

Runs the equal-weight baseline (the "known good" policy validated
against the volatility-drag check) through PortfolioEnv and saves the
full step-by-step history as a CSV. Person C can inspect this without
running any of your code themselves.

Usage:
    python export_episode_log.py data/clean/full_dataset.parquet
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.portfolioenv import PortfolioEnv


def run_equal_weight_episode(env: PortfolioEnv) -> pd.DataFrame:
    obs, info = env.reset(seed=0)
    action = np.zeros(env.action_space.shape, dtype=np.float32)
    action[-1] = -10.0  # push cash logit down -> ~equal weight across assets
    done = False
    while not done:
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_history_df()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: python export_episode_log.py <path to full_dataset.parquet>")

    data = pd.read_parquet(sys.argv[1])
    env = PortfolioEnv(data, window=20, transaction_cost_bps=10.0)
    history = env_history = run_equal_weight_episode(env)

    out_path = "episode_log_sample.csv"
    history.to_csv(out_path)
    print(f"Saved {len(history)} rows to {out_path}")
    print(f"Columns: {history.columns.tolist()}")
    print(history.head())