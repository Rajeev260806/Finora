"""
Step 7 — Validate PortfolioEnv with a random policy.

Before handing this env to Person C or plugging in a GNN+RL agent,
confirm it behaves sensibly: no exploding/negative-nonsensical
portfolio values, finite rewards, and plausible observation ranges.

Usage:
    python validate_env.py data/clean/full_dataset.parquet
"""

import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to sys.path to allow imports from src
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.environment.portfolioenv import PortfolioEnv


def run_random_episode(env: PortfolioEnv, seed: int) -> pd.DataFrame:
    obs, info = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        action = rng.normal(size=env.action_space.shape)  # random logits
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return env.get_history_df()


def sanity_checks(history: pd.DataFrame) -> list[str]:
    problems = []
    if history["portfolio_value"].isnull().any():
        problems.append("NaN in portfolio_value")
    if (history["portfolio_value"] <= 0).any():
        problems.append("Non-positive portfolio_value encountered")
    if not np.isfinite(history["raw_return"]).all():
        problems.append("Non-finite raw_return encountered")
    if (history["raw_return"].abs() > 0.5).any():
        problems.append("raw_return outside expected +/-50% range — check data/units")
    return problems


def make_synthetic_data(n_dates=500, n_assets=6, seed=0) -> pd.DataFrame:
    """Fallback synthetic dataset with the same shape Person A's pipeline
    would produce, for testing this script before the real parquet exists."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n_dates)
    tickers = [f"TICK{i}" for i in range(n_assets)]
    idx = pd.MultiIndex.from_product([dates, tickers], names=["Date", "Ticker"])
    daily_return = rng.normal(loc=0.0003, scale=0.015, size=len(idx))
    return pd.DataFrame({"daily_return": daily_return}, index=idx)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        data = pd.read_parquet(sys.argv[1])
    else:
        print("No parquet path given — using synthetic data for a dry run.\n")
        data = make_synthetic_data()

    env = PortfolioEnv(data, window=20, transaction_cost_bps=10.0)

    n_episodes = 10
    all_final_values = []
    fig, ax = plt.subplots(figsize=(9, 5))

    for ep in range(n_episodes):
        history = run_random_episode(env, seed=ep)
        problems = sanity_checks(history)
        if problems:
            print(f"[episode {ep}] ISSUES: {problems}")
        else:
            print(f"[episode {ep}] OK — final value {history['portfolio_value'].iloc[-1]:.4f}")
        all_final_values.append(history["portfolio_value"].iloc[-1])
        ax.plot(history.index, history["portfolio_value"], alpha=0.7, label=f"ep {ep}")

    ax.axhline(1.0, color="black", linestyle="--", linewidth=0.8)
    ax.set_title("Random-policy validation episodes — portfolio value")
    ax.set_ylabel("Portfolio value (start = 1.0)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig("validation_episodes.png", dpi=150)

    print(f"\nMean final value across {n_episodes} random episodes: {np.mean(all_final_values):.4f}")
    print("Plot saved to validation_episodes.png")