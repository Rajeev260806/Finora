"""
Validation Tuning for Residual Policy Alpha
===========================================
PROJECT_ACTION_PLAN.md:
"Select hyperparameters on validation only. Run test once."
"w = softmax( log(1/N) + alpha * tanh(a) ), alpha around 0.1 to 0.5 (tune on validation)."

Trains PPO on train.parquet (2015-2021) and evaluates on val.parquet (2022).
"""

import os
import sys
import numpy as np
import pandas as pd
from stable_baselines3 import PPO

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.environment.gnn_portfolio_env import GNNPortfolioEnv
from src.analysis.metrics import compute_metrics, compute_metrics_from_history

TRAIN_PATH = os.path.join(BASE_DIR, "data", "clean", "train.parquet")
VAL_PATH = os.path.join(BASE_DIR, "data", "clean", "val.parquet")


def evaluate_agent(model, env):
    obs, info = env.reset(seed=42)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    return compute_metrics_from_history(env.get_history_df())


def main():
    print("=" * 65)
    print("  TUNING RESIDUAL ALPHA ON VALIDATION PERIOD (2022)")
    print("=" * 65)

    train_data = pd.read_parquet(TRAIN_PATH)
    val_data = pd.read_parquet(VAL_PATH)

    alphas = [0.1, 0.25, 0.5]
    results = {}

    for alpha in alphas:
        print(f"\n--- Testing residual_alpha = {alpha} ---")
        train_env = GNNPortfolioEnv(
            data=train_data,
            embedding_dir="src/gnn/embeddings_gat",
            residual_alpha=alpha,
            seed=42
        )
        model = PPO(
            "MultiInputPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=64,
            ent_coef=0.005,
            seed=42,
            verbose=0,
        )
        # Train for 50k steps to compare on val
        model.learn(total_timesteps=50000)

        # Evaluate on validation period
        val_env = GNNPortfolioEnv(
            data=val_data,
            embedding_dir="src/gnn/embeddings_gat",
            residual_alpha=alpha,
            seed=42
        )
        val_metrics = evaluate_agent(model, val_env)
        results[alpha] = val_metrics
        print(f"Val Result for alpha={alpha}: Sharpe={val_metrics['Sharpe Ratio']:.4f}, Return={val_metrics['Cumulative Return']:.2%}, MaxDD={val_metrics['Max Drawdown']:.2%}")

    print("\n" + "=" * 65)
    print("  VALIDATION TUNING SUMMARY")
    print("=" * 65)
    best_alpha = max(results.keys(), key=lambda a: results[a]["Sharpe Ratio"])
    for a, m in results.items():
        star = " <-- BEST" if a == best_alpha else ""
        print(f"Alpha {a:4.2f}: Final Val={m['Final Portfolio Value']:.4f} | Ann Ret={m['Annualized Return']:>7.2%} | Sharpe={m['Sharpe Ratio']:>7.4f} | Max DD={m['Max Drawdown']:>7.2%}{star}")
    print(f"\nSelected best alpha for test evaluation: {best_alpha}")


if __name__ == "__main__":
    main()
