"""
Finora Phase 4 (Person B -> Person C Handoff) — Checkpoint Evaluator
===================================================================
Loads any trained PPO model/checkpoint and evaluates it deterministically
over a specified market period (train, test, or full dataset).

Outputs saved to target directory (e.g. results/ppo_eval/<checkpoint_id>/):
1. portfolio_value.csv: date, portfolio_value, daily_return, turnover, transaction_cost
2. weights.csv: date, AAPL, ADBE, ..., CASH
3. metrics.json: standardized performance metrics (Sharpe, return, drawdown, etc.)
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

# Add workspace root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from stable_baselines3 import PPO
from src.environment.gnn_portfolio_env import GNNPortfolioEnv


def evaluate_checkpoint(
    checkpoint_path: str,
    output_dir: str,
    data_path: str = "data/clean/full_dataset.parquet",
    embedding_dir: str = "src/gnn/embeddings",
    deterministic: bool = True,
    seed: int = 42,
) -> dict:
    """
    Evaluates a PPO checkpoint and writes standardized outputs for Person C.
    """
    print("=" * 65)
    print("  EVALUATING PPO CHECKPOINT")
    print("=" * 65)
    print(f"Checkpoint:          {checkpoint_path}")
    print(f"Dataset:             {data_path}")
    print(f"Output directory:    {output_dir}")
    print(f"Deterministic:       {deterministic}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load data
    full_data_path = os.path.join(BASE_DIR, data_path) if not os.path.isabs(data_path) else data_path
    if not os.path.exists(full_data_path):
        raise FileNotFoundError(f"Data file not found: {full_data_path}")
    data = pd.read_parquet(full_data_path)

    # 2. Setup environment
    full_emb_dir = os.path.join(BASE_DIR, embedding_dir) if not os.path.isabs(embedding_dir) else embedding_dir
    env = GNNPortfolioEnv(
        data=data,
        embedding_dir=full_emb_dir,
        window=20,
        transaction_cost_bps=10.0,
        initial_cash=1.0,
        return_col="Returns Daily",
        embedding_dim=32,
        lambda_vol=0.10,
        lambda_conc=0.01,
        lambda_txn=1.0,
        seed=seed,
    )

    # 3. Load model
    full_model_path = os.path.join(BASE_DIR, checkpoint_path) if not os.path.isabs(checkpoint_path) else checkpoint_path
    if not os.path.exists(full_model_path) and not full_model_path.endswith(".zip"):
        full_model_path += ".zip"
    if not os.path.exists(full_model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {full_model_path}")

    model = PPO.load(full_model_path, env=env)
    print(f"Loaded policy from:  {full_model_path}")

    # 4. Run deterministic trajectory
    obs, info = env.reset(seed=seed)
    tickers = env.tickers
    columns_weights = ["date"] + list(tickers) + ["CASH"]

    records_pv = []
    records_weights = []

    # Record initial step (t=0)
    init_date = env.dates[env.current_idx]
    init_date_str = str(init_date)[:10]
    init_weights = env.prev_weights

    records_pv.append({
        "date": init_date_str,
        "portfolio_value": 1.0,
        "daily_return": 0.0,
        "turnover": 0.0,
        "transaction_cost": 0.0,
    })
    row_w = [init_date_str] + [float(w) for w in init_weights]
    records_weights.append(row_w)

    done = False
    step_count = 0

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, step_info = env.step(action)
        step_count += 1
        done = terminated or truncated

        step_date_str = str(step_info["date"])[:10]
        records_pv.append({
            "date": step_date_str,
            "portfolio_value": float(step_info["portfolio_value"]),
            "daily_return": float(step_info["net_return"]),
            "turnover": float(step_info["turnover"]),
            "transaction_cost": float(step_info["transaction_cost"]),
        })

        current_w = [step_date_str] + [float(w) for w in env.prev_weights]
        records_weights.append(current_w)

    # 5. Format DataFrames
    pv_df = pd.DataFrame(records_pv).drop_duplicates(subset=["date"], keep="last")
    weights_df = pd.DataFrame(records_weights, columns=columns_weights).drop_duplicates(subset=["date"], keep="last")

    pv_csv_path = os.path.join(output_dir, "portfolio_value.csv")
    weights_csv_path = os.path.join(output_dir, "weights.csv")

    pv_df.to_csv(pv_csv_path, index=False)
    weights_df.to_csv(weights_csv_path, index=False)

    # 6. Compute standard performance metrics
    pv_series = pv_df["portfolio_value"].values
    daily_returns = pv_df["daily_return"].iloc[1:].values
    final_pv = float(pv_series[-1])

    n_days = len(daily_returns)
    ann_factor = 252.0

    if n_days > 1:
        cum_ret = final_pv - 1.0
        ann_return = float((1.0 + cum_ret) ** (ann_factor / n_days) - 1.0) if final_pv > 0 else -1.0
        daily_vol = float(np.std(daily_returns, ddof=1))
        ann_vol = daily_vol * np.sqrt(ann_factor)
        mean_daily = float(np.mean(daily_returns))
        sharpe = float((ann_return / ann_vol)) if ann_vol > 1e-8 else 0.0

        # Maximum drawdown
        running_max = np.maximum.accumulate(pv_series)
        drawdowns = (pv_series - running_max) / np.maximum(running_max, 1e-8)
        max_drawdown = float(np.min(drawdowns))
    else:
        ann_return = 0.0
        daily_vol = 0.0
        mean_daily = 0.0
        sharpe = 0.0
        max_drawdown = 0.0

    avg_turnover = float(pv_df["turnover"].mean())
    total_cost = float(pv_df["transaction_cost"].sum())

    metrics = {
        "final_portfolio_value": final_pv,
        "annualized_return": ann_return,
        "sharpe_ratio": sharpe,
        "maximum_drawdown": max_drawdown,
        "mean_daily_return": mean_daily,
        "daily_volatility": daily_vol,
        "average_turnover": avg_turnover,
        "total_transaction_cost": total_cost,
        "steps_evaluated": step_count,
    }

    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n[OK] Evaluation outputs successfully generated:")
    print(f"     - Portfolio Value: {pv_csv_path} ({len(pv_df)} rows)")
    print(f"     - Weights:         {weights_csv_path} ({len(weights_df)} rows)")
    print(f"     - Metrics:         {metrics_json_path}")
    print("\nMetrics Summary:")
    print(f"  Final Portfolio Value: {final_pv:.4f}")
    print(f"  Annualized Return:     {ann_return * 100:.2f}%")
    print(f"  Sharpe Ratio:          {sharpe:.3f}")
    print(f"  Max Drawdown:          {max_drawdown * 100:.2f}%")
    print(f"  Average Turnover:      {avg_turnover:.4f}")
    print(f"  Total Transaction Cost:{total_cost:.4f}")
    print("=" * 65)

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate PPO Checkpoint for Person C")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to PPO checkpoint .zip")
    parser.add_argument("--output_dir", type=str, required=True, help="Output folder to write csv & json")
    parser.add_argument("--data", type=str, default="data/clean/full_dataset.parquet", help="Parquet dataset path")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic action")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        data_path=args.data,
        deterministic=args.deterministic,
        seed=args.seed,
    )
