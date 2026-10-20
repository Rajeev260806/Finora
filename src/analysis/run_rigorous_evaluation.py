"""
Finora Rigorous Strategy Evaluation & Ablation Runner
=====================================================
Addresses Phases 1.2, 3.1-3.3, 4, and 5 of PROJECT_ACTION_PLAN.md:
- Evaluates Baselines (Buy & Hold, Equal Weight, Momentum, Mean-Variance)
- Evaluates PPO Agents across 5 seeds:
    1. GAT Hybrid + Residual Policy (Our primary proposed system)
    2. GCN + Residual Policy
    3. Random Embeddings + Residual Policy (Sanity ablation)
    4. No Embeddings (Raw returns only) + Residual Policy
    5. GAT Hybrid + Free-form Policy (Ablation to test residual policy contribution)
- Strict train (2015-2021) and test (2022-12-29 to 2024-12-31) separation: NO LEAKAGE
- Standardized metrics (CAGR annualization, Sharpe, Max Drawdown, one-way turnover)
- Cost robustness evaluation (5 bps, 10 bps, 20 bps)
- Aggregates Mean +/- Std across seeds
- Generates markdown summary and JSON records
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from stable_baselines3 import PPO
from src.environment.gnn_portfolio_env import GNNPortfolioEnv
from src.environment.portfolioenv import PortfolioEnv
from src.analysis.metrics import compute_metrics, compute_metrics_from_history
from src.baselines.evaluate_all_baselines import evaluate_baselines

TRAIN_PATH = os.path.join(BASE_DIR, "data", "clean", "train.parquet")
TEST_PATH = os.path.join(BASE_DIR, "data", "clean", "test.parquet")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def train_and_eval_ppo(
    variant_name: str,
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    embedding_dir: str | None,
    residual_alpha: float,
    seed: int,
    residual_mode: str = "equal_weight",
    rebalance_interval: int = 1,
    momentum_lookback: int = 60,
    momentum_temp: float = 0.2,
    downside_only: bool = True,
    timesteps: int = 50000,
    tc_bps: float = 10.0,
):
    """
    Trains a PPO policy on train_data and evaluates out-of-sample on test_data.
    """
    print(f"  [Train & Eval] Variant: {variant_name} | Seed: {seed} | alpha: {residual_alpha} | mode: {residual_mode} | interval: {rebalance_interval} | steps: {timesteps}")
    
    use_embeddings = embedding_dir is not None
    
    # 1. Training Environment
    if use_embeddings:
        train_env = GNNPortfolioEnv(
            data=train_data,
            embedding_dir=embedding_dir,
            window=20,
            transaction_cost_bps=tc_bps,
            initial_cash=1.0,
            residual_alpha=residual_alpha,
            residual_mode=residual_mode,
            rebalance_interval=rebalance_interval,
            momentum_lookback=momentum_lookback,
            momentum_temp=momentum_temp,
            downside_only=downside_only,
            seed=seed,
            lambda_vol=0.10,
            lambda_conc=0.001 if residual_mode == "momentum" else 0.01,
            lambda_txn=1.0,
        )
    else:
        train_env = PortfolioEnv(
            data=train_data,
            window=20,
            transaction_cost_bps=tc_bps,
            initial_cash=1.0,
            residual_alpha=residual_alpha,
            residual_mode=residual_mode,
            rebalance_interval=rebalance_interval,
            momentum_lookback=momentum_lookback,
            momentum_temp=momentum_temp,
            downside_only=downside_only,
            seed=seed,
            use_embeddings=False,
            lambda_vol=0.10,
            lambda_conc=0.001 if residual_mode == "momentum" else 0.01,
            lambda_txn=1.0,
        )

    # 2. Train PPO Agent (Both environments use Dict observation space)
    policy_name = "MultiInputPolicy"
    model = PPO(
        policy=policy_name,
        env=train_env,
        learning_rate=3e-4,
        n_steps=512,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        seed=seed,
        verbose=0,
    )
    model.learn(total_timesteps=timesteps)

    # 3. Test Evaluation Environment (strictly out-of-sample)
    if use_embeddings:
        test_env = GNNPortfolioEnv(
            data=test_data,
            embedding_dir=embedding_dir,
            window=20,
            transaction_cost_bps=tc_bps,
            initial_cash=1.0,
            residual_alpha=residual_alpha,
            residual_mode=residual_mode,
            rebalance_interval=rebalance_interval,
            momentum_lookback=momentum_lookback,
            momentum_temp=momentum_temp,
            downside_only=downside_only,
            seed=seed,
        )
    else:
        test_env = PortfolioEnv(
            data=test_data,
            window=20,
            transaction_cost_bps=tc_bps,
            initial_cash=1.0,
            residual_alpha=residual_alpha,
            residual_mode=residual_mode,
            rebalance_interval=rebalance_interval,
            momentum_lookback=momentum_lookback,
            momentum_temp=momentum_temp,
            downside_only=downside_only,
            seed=seed,
            use_embeddings=False,
        )

    obs, info = test_env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = test_env.step(action)
        done = terminated or truncated

    hist_df = test_env.get_history_df()
    metrics = compute_metrics_from_history(hist_df, risk_free_rate=0.0)
    return metrics, hist_df


def format_mean_std(values: list[float], is_percent: bool = False, precision: int = 4) -> str:
    m = float(np.mean(values))
    s = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    if is_percent:
        return f"{m * 100:.2f}% +/- {s * 100:.2f}%"
    else:
        fmt = f"{{:.{precision}f}} +/- {{:.{precision}f}}"
        return fmt.format(m, s)


def main():
    parser = argparse.ArgumentParser(description="Finora Rigorous Multi-Seed Strategy Evaluation")
    parser.add_argument("--timesteps", type=int, default=50000, help="Timesteps per seed per variant")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 101, 202, 303, 404], help="List of random seeds")
    parser.add_argument("--alpha", type=float, default=0.25, help="Selected residual alpha")
    args = parser.parse_args()

    print("=" * 75)
    print("  FINORA RIGOROUS MULTI-SEED STRATEGY EVALUATION & ABLATION STUDY")
    print("=" * 75)
    print(f"Seeds:        {args.seeds}")
    print(f"Timesteps:    {args.timesteps}")
    print(f"Alpha:        {args.alpha}")

    train_data = pd.read_parquet(TRAIN_PATH)
    test_data = pd.read_parquet(TEST_PATH)

    # ---------------------------------------------------------
    # 1. Evaluate All Baselines (Deterministic)
    # ---------------------------------------------------------
    print("\n--- Evaluating Baselines on Test Period (2022-12-29 to 2024-12-31, 10 bps) ---")
    baseline_metrics, baseline_histories = evaluate_baselines(
        data_path=os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet"),
        transaction_cost_bps=10.0,
        test_start="2022-12-29"
    )

    # ---------------------------------------------------------
    # 2. Ablation & Strategy Variants to Train Across Seeds
    # ---------------------------------------------------------
    variants = [
        {
            "name": "PPO + GAT (Enhanced Momentum, Weekly)",
            "embedding_dir": "src/gnn/embeddings_gat",
            "residual_alpha": 0.15,
            "residual_mode": "momentum",
            "rebalance_interval": 5,
        },
        {
            "name": "PPO + GAT (Residual)",
            "embedding_dir": "src/gnn/embeddings_gat",
            "residual_alpha": args.alpha,
            "residual_mode": "equal_weight",
            "rebalance_interval": 1,
        },
        {
            "name": "PPO + GCN (Residual)",
            "embedding_dir": "src/gnn/embeddings_gcn",
            "residual_alpha": args.alpha,
            "residual_mode": "equal_weight",
            "rebalance_interval": 1,
        },
        {
            "name": "PPO + Random GNN (Residual)",
            "embedding_dir": "src/gnn/embeddings_random",
            "residual_alpha": args.alpha,
            "residual_mode": "equal_weight",
            "rebalance_interval": 1,
        },
        {
            "name": "PPO + No Embeddings (Residual)",
            "embedding_dir": None,
            "residual_alpha": args.alpha,
            "residual_mode": "equal_weight",
            "rebalance_interval": 1,
        },
        {
            "name": "PPO + GAT (Free-form Softmax)",
            "embedding_dir": "src/gnn/embeddings_gat",
            "residual_alpha": 0.0,
            "residual_mode": "equal_weight",
            "rebalance_interval": 1,
        },
    ]

    all_results = {}
    representative_histories = {}

    for var in variants:
        v_name = var["name"]
        print(f"\n=======================================================")
        print(f"Running Variant: {v_name}")
        print(f"=======================================================")
        all_results[v_name] = {
            "Final Portfolio Value": [],
            "Cumulative Return": [],
            "Annualized Return": [],
            "Sharpe Ratio": [],
            "Volatility": [],
            "Max Drawdown": [],
            "Average Turnover": [],
        }

        for seed in args.seeds:
            metrics, hist_df = train_and_eval_ppo(
                variant_name=v_name,
                train_data=train_data,
                test_data=test_data,
                embedding_dir=var["embedding_dir"],
                residual_alpha=var["residual_alpha"],
                seed=seed,
                residual_mode=var.get("residual_mode", "equal_weight"),
                rebalance_interval=var.get("rebalance_interval", 1),
                timesteps=args.timesteps,
                tc_bps=10.0,
            )
            for k in all_results[v_name]:
                if k in metrics:
                    all_results[v_name][k].append(metrics[k])

            if seed == args.seeds[0]:
                representative_histories[v_name] = hist_df

    # ---------------------------------------------------------
    # 3. Transaction Cost Robustness Check (5, 10, 20 bps)
    # ---------------------------------------------------------
    print("\n--- Running Cost Robustness Check (5, 10, 20 bps) on GAT Residual (Seed 42) ---")
    cost_levels = [5.0, 10.0, 20.0]
    cost_results = {}
    for c in cost_levels:
        m, _ = train_and_eval_ppo(
            variant_name="GAT Residual",
            train_data=train_data,
            test_data=test_data,
            embedding_dir="src/gnn/embeddings_gat",
            residual_alpha=args.alpha,
            seed=42,
            timesteps=min(args.timesteps, 30000),
            tc_bps=c,
        )
        cost_results[f"{int(c)} bps"] = m

    # ---------------------------------------------------------
    # 4. Compile Unified Results Table
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("  COMPREHENSIVE OUT-OF-SAMPLE TEST RESULTS (2022-12-29 to 2024-12-31, 10 bps)")
    print("=" * 90)
    header = f"| {'Strategy / Variant':<30} | {'Final Value':<17} | {'Ann Return':<17} | {'Sharpe':<17} | {'Max DD':<17} | {'Turnover':<15} |"
    sep = f"|{'-'*32}|{'-'*19}|{'-'*19}|{'-'*19}|{'-'*19}|{'-'*17}|"
    print(header)
    print(sep)

    table_rows = []

    # Baselines
    for b_name, bm in baseline_metrics.items():
        fv = f"{bm['Final Portfolio Value']:.4f}"
        ar = f"{bm['Annualized Return']*100:.2f}%"
        sr = f"{bm['Sharpe Ratio']:.4f}"
        md = f"{bm['Max Drawdown']*100:.2f}%"
        to = f"{bm['Average Turnover']:.4f}"
        row_str = f"| {b_name:<30} | {fv:<17} | {ar:<17} | {sr:<17} | {md:<17} | {to:<15} |"
        print(row_str)
        table_rows.append({
            "Strategy": b_name,
            "Final Value": fv,
            "Annualized Return": ar,
            "Sharpe": sr,
            "Max Drawdown": md,
            "Turnover": to,
        })

    print(sep)

    # RL Variants (Mean +/- Std)
    for v_name, res in all_results.items():
        fv = format_mean_std(res["Final Portfolio Value"], precision=4)
        ar = format_mean_std(res["Annualized Return"], is_percent=True)
        sr = format_mean_std(res["Sharpe Ratio"], precision=4)
        md = format_mean_std(res["Max Drawdown"], is_percent=True)
        to = format_mean_std(res["Average Turnover"], precision=4)
        row_str = f"| {v_name:<30} | {fv:<17} | {ar:<17} | {sr:<17} | {md:<17} | {to:<15} |"
        print(row_str)
        table_rows.append({
            "Strategy": v_name,
            "Final Value": fv,
            "Annualized Return": ar,
            "Sharpe": sr,
            "Max Drawdown": md,
            "Turnover": to,
        })
    print("=" * 90)

    # ---------------------------------------------------------
    # 5. Cost Robustness Summary
    # ---------------------------------------------------------
    print("\nTransaction Cost Robustness (PPO + GAT Residual):")
    for c_str, cm in cost_results.items():
        print(f"  {c_str}: Final Value = {cm['Final Portfolio Value']:.4f} | Ann Ret = {cm['Annualized Return']:.2%} | Sharpe = {cm['Sharpe Ratio']:.4f} | Turnover = {cm['Average Turnover']:.4f}")

    # ---------------------------------------------------------
    # 6. Save Artifacts & Visualizations
    # ---------------------------------------------------------
    summary_path = os.path.join(RESULTS_DIR, "rigorous_evaluation_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "baselines": {k: {m: float(v) for m, v in metrics.items()} for k, metrics in baseline_metrics.items()},
            "rl_variants": all_results,
            "cost_robustness": {k: {m: float(v) for m, v in metrics.items()} for k, metrics in cost_results.items()},
        }, f, indent=2)
    print(f"\n[OK] Saved results JSON to: {summary_path}")

    # Plot portfolio trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    for b_name in ["Buy & Hold", "Equal Weight", "Mean-Variance"]:
        if b_name in baseline_histories:
            df = baseline_histories[b_name]
            ax.plot(pd.to_datetime(df.index), df["portfolio_value"], label=b_name, lw=2.0)

    for v_name in ["PPO + GAT (Residual)", "PPO + GAT (Free-form Softmax)"]:
        if v_name in representative_histories:
            df = representative_histories[v_name]
            dates = pd.to_datetime(df.index)
            ax.plot(dates, df["portfolio_value"], label=f"{v_name} (Seed 42)", lw=2.2, ls="--")

    ax.set_title("Out-of-Sample Portfolio Value Trajectories (2022-12-29 to 2024-12-31, 10 bps)", fontsize=13)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    plot_path = os.path.join(PLOTS_DIR, "rigorous_strategy_trajectories_test.png")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"[OK] Saved comparison trajectory plot to: {plot_path}")


if __name__ == "__main__":
    main()
