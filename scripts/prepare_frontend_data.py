"""
Script to extract, compute, and assemble comprehensive model outputs,
metrics, trajectories, weights, and graph networks for the Finora Frontend.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from stable_baselines3 import PPO
from src.environment.gnn_portfolio_env import GNNPortfolioEnv
from src.analysis.metrics import compute_metrics, compute_metrics_from_history
from src.baselines.evaluate_all_baselines import evaluate_baselines

FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_DIR = os.path.join(FRONTEND_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# 1. Sector Map
SECTOR_MAP = {
    "AAPL": "Technology",
    "ADBE": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "V": "Financials",
    "AEP": "Utilities",
    "DUK": "Utilities",
    "NEE": "Utilities",
    "SO": "Utilities",
    "AMT": "Real Estate",
    "O": "Real Estate",
    "PLD": "Real Estate",
    "SPG": "Real Estate",
    "BAC": "Financials",
    "GS": "Financials",
    "JPM": "Financials",
    "COP": "Energy",
    "CVX": "Energy",
    "SLB": "Energy",
    "XOM": "Energy",
}

SECTOR_COLORS = {
    "Technology": "#3b82f6",     # Blue
    "Financials": "#10b981",     # Emerald
    "Energy": "#f59e0b",         # Amber
    "Utilities": "#8b5cf6",      # Purple
    "Real Estate": "#ec4899",    # Pink
    "CASH": "#64748b"            # Slate
}

def extract_all():
    print("[1/6] Evaluating Baselines on Holdout Test Period (2022-12-29 to 2024-12-31)...")
    baseline_metrics, baseline_histories = evaluate_baselines(
        data_path=os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet"),
        transaction_cost_bps=10.0,
        test_start="2022-12-29"
    )

    test_dates = [str(d)[:10] for d in baseline_histories["Buy & Hold"].index]

    trajectories_test = {
        "dates": test_dates,
        "strategies": {}
    }

    # Add baselines to test trajectories
    for name, hist in baseline_histories.items():
        pv = [round(float(v), 4) for v in hist["portfolio_value"].values]
        # compute drawdown series
        arr = np.array(pv)
        peak = np.maximum.accumulate(arr)
        dd = [round(float(d), 4) for d in (arr - peak) / peak]
        
        trajectories_test["strategies"][name] = {
            "name": name,
            "category": "Baseline",
            "portfolio_value": pv,
            "drawdown": dd,
            "daily_return": [round(float(r), 5) for r in hist["daily_return"].values],
            "turnover": [round(float(t), 5) for t in hist["turnover"].values] if "turnover" in hist.columns else [0.0]*len(pv),
            "metrics": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in baseline_metrics[name].items()}
        }

    # 2. Evaluate Enhanced GNN + PPO (Momentum residual weekly)
    print("[2/6] Evaluating Enhanced GNN + PPO model on holdout test set...")
    test_data = pd.read_parquet(os.path.join(BASE_DIR, "data", "clean", "test.parquet"))
    emb_dir = os.path.join(BASE_DIR, "src", "gnn", "embeddings_gat" if os.path.exists(os.path.join(BASE_DIR, "src", "gnn", "embeddings_gat")) else "embeddings")
    
    enhanced_model_path = os.path.join(BASE_DIR, "src", "gnn", "models", "gnn_ppo")
    env_enhanced = GNNPortfolioEnv(
        data=test_data,
        embedding_dir=emb_dir,
        window=20,
        transaction_cost_bps=10.0,
        initial_cash=1.0,
        return_col="Returns Daily",
        residual_alpha=0.15,
        residual_mode="momentum",
        rebalance_interval=5,
        momentum_lookback=60,
        momentum_temp=0.2,
        downside_only=True,
        seed=0,
    )

    model_enh = PPO.load(enhanced_model_path, env=env_enhanced)
    obs, _ = env_enhanced.reset(seed=0)
    while True:
        action, _ = model_enh.predict(obs, deterministic=True)
        obs, reward, term, trunc, info = env_enhanced.step(action)
        if term or trunc:
            break
    
    hist_enh = env_enhanced.get_history_df()
    m_enh = compute_metrics_from_history(hist_enh)
    pv_enh = [round(float(v), 4) for v in hist_enh["portfolio_value"].values]
    arr = np.array(pv_enh)
    peak = np.maximum.accumulate(arr)
    dd_enh = [round(float(d), 4) for d in (arr - peak) / peak]

    trajectories_test["strategies"]["PPO + GAT (Enhanced Momentum)"] = {
        "name": "PPO + GAT (Enhanced Momentum)",
        "category": "RL Strategy",
        "portfolio_value": pv_enh,
        "drawdown": dd_enh,
        "daily_return": [round(float(r), 5) for r in hist_enh["net_return"].values],
        "turnover": [round(float(t), 5) for t in hist_enh["turnover"].values],
        "metrics": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in m_enh.items()}
    }

    # 3. Evaluate PPO + GAT Residual (Equal Weight anchor)
    print("[3/6] Evaluating PPO + GAT (Residual Equal Weight)...")
    env_gat_eq = GNNPortfolioEnv(
        data=test_data,
        embedding_dir=emb_dir,
        window=20,
        transaction_cost_bps=10.0,
        initial_cash=1.0,
        residual_alpha=0.10,
        residual_mode="equal_weight",
        seed=42,
    )
    
    final_model_path = os.path.join(BASE_DIR, "results", "ppo_final_model.zip")
    if os.path.exists(final_model_path):
        model_gat_eq = PPO.load(final_model_path, env=env_gat_eq)
        obs, _ = env_gat_eq.reset(seed=42)
        while True:
            action, _ = model_gat_eq.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env_gat_eq.step(action)
            if term or trunc:
                break
        hist_gat_eq = env_gat_eq.get_history_df()
        m_gat_eq = compute_metrics_from_history(hist_gat_eq)
        pv_eq = [round(float(v), 4) for v in hist_gat_eq["portfolio_value"].values]
        arr = np.array(pv_eq)
        peak = np.maximum.accumulate(arr)
        dd_eq = [round(float(d), 4) for d in (arr - peak) / peak]

        trajectories_test["strategies"]["PPO + GAT (Residual Equal-Weight)"] = {
            "name": "PPO + GAT (Residual Equal-Weight)",
            "category": "RL Strategy",
            "portfolio_value": pv_eq,
            "drawdown": dd_eq,
            "daily_return": [round(float(r), 5) for r in hist_gat_eq["net_return"].values],
            "turnover": [round(float(t), 5) for t in hist_gat_eq["turnover"].values],
            "metrics": {k: round(float(v), 4) if isinstance(v, (int, float, np.floating)) else v for k, v in m_gat_eq.items()}
        }

    # 4. Load ablation & rigorous evaluation summary
    print("[4/6] Loading rigorous evaluation summary & cost sensitivity...")
    summary_path = os.path.join(BASE_DIR, "results", "rigorous_evaluation_summary.json")
    with open(summary_path, "r") as f:
        summary_data = json.load(f)

    # 5. Extract In-Sample 2015-2024 Historical Data (1994 days)
    print("[5/6] Extracting In-Sample 10-year trajectories & weights...")
    ppo_final_pv_csv = os.path.join(BASE_DIR, "results", "ppo_eval", "final", "portfolio_value.csv")
    ppo_weights_csv = os.path.join(BASE_DIR, "results", "ppo_eval", "final", "weights.csv")
    mv_pv_csv = os.path.join(BASE_DIR, "results", "baselines", "mean_variance_full", "portfolio_value.csv")

    in_sample_data = {
        "dates": [],
        "strategies": {},
        "weights": {
            "dates": [],
            "assets": list(SECTOR_MAP.keys()) + ["CASH"],
            "series": []
        }
    }

    if os.path.exists(ppo_final_pv_csv):
        df_ppo_pv = pd.read_csv(ppo_final_pv_csv).dropna(subset=["date"])
        in_sample_data["dates"] = df_ppo_pv["date"].astype(str).tolist()
        pv_arr = df_ppo_pv["portfolio_value"].values
        peak = np.maximum.accumulate(pv_arr)
        dd = (pv_arr - peak) / peak
        
        in_sample_data["strategies"]["GNN + PPO (In-Sample)"] = {
            "portfolio_value": [round(float(x), 4) for x in pv_arr],
            "drawdown": [round(float(x), 4) for x in dd],
            "daily_return": [round(float(x), 5) for x in df_ppo_pv["daily_return"].values],
            "turnover": [round(float(x), 5) for x in df_ppo_pv["turnover"].values],
            "final_value": round(float(pv_arr[-1]), 4)
        }

    if os.path.exists(mv_pv_csv):
        df_mv_pv = pd.read_csv(mv_pv_csv).dropna(subset=["date"])
        pv_arr = df_mv_pv["portfolio_value"].values
        peak = np.maximum.accumulate(pv_arr)
        dd = (pv_arr - peak) / peak
        in_sample_data["strategies"]["Mean-Variance (In-Sample)"] = {
            "portfolio_value": [round(float(x), 4) for x in pv_arr],
            "drawdown": [round(float(x), 4) for x in dd],
            "final_value": round(float(pv_arr[-1]), 4)
        }

    # Sample weights every 5 days for responsive browser rendering (400 points instead of 2000)
    if os.path.exists(ppo_weights_csv):
        df_w = pd.read_csv(ppo_weights_csv).dropna(subset=["date"])
        sampled_w = df_w.iloc[::5].copy()
        in_sample_data["weights"]["dates"] = sampled_w["date"].astype(str).tolist()
        
        weight_records = []
        for _, row in sampled_w.iterrows():
            rec = {col: round(float(row[col]), 4) for col in df_w.columns if col not in ["date", "Unnamed: 0"]}
            # compute sector totals
            sec_totals = {sec: 0.0 for sec in SECTOR_COLORS.keys()}
            for ticker, sec in SECTOR_MAP.items():
                if ticker in rec:
                    sec_totals[sec] += rec[ticker]
            if "CASH" in rec:
                sec_totals["CASH"] = rec["CASH"]
            rec["_sectors"] = {k: round(v, 4) for k, v in sec_totals.items()}
            weight_records.append(rec)
        in_sample_data["weights"]["series"] = weight_records

    # 6. Extract Asset Universe & Correlation Matrix
    print("[6/6] Computing Correlation Graph & Asset Statistics...")
    full_df = pd.read_parquet(os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet"))
    wide_ret = full_df["Returns Daily"].unstack(level="Ticker").fillna(0.0)
    corr_matrix = wide_ret.corr().round(4)
    tickers = corr_matrix.columns.tolist()

    # Asset stats
    asset_stats = []
    for t in tickers:
        series = wide_ret[t]
        ann_ret = float((1 + series.mean()) ** 252 - 1)
        ann_vol = float(series.std() * np.sqrt(252))
        sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0.0
        sec = SECTOR_MAP.get(t, "Other")
        asset_stats.append({
            "ticker": t,
            "sector": sec,
            "color": SECTOR_COLORS.get(sec, "#64748b"),
            "annualized_return": round(ann_ret * 100, 2),
            "volatility": round(ann_vol * 100, 2),
            "sharpe": round(sharpe, 2)
        })

    # Graph links (correlation > 0.40)
    nodes = [{"id": t, "sector": SECTOR_MAP.get(t, "Other"), "color": SECTOR_COLORS.get(SECTOR_MAP.get(t, "Other"), "#64748b")} for t in tickers]
    links = []
    for i, t1 in enumerate(tickers):
        for j, t2 in enumerate(tickers):
            if i < j:
                c = float(corr_matrix.loc[t1, t2])
                if abs(c) >= 0.40:
                    links.append({
                        "source": t1,
                        "target": t2,
                        "weight": round(c, 3),
                        "positive": c > 0
                    })

    graph_data = {
        "nodes": nodes,
        "links": links,
        "matrix": {
            "tickers": tickers,
            "values": corr_matrix.values.tolist()
        },
        "sectors": SECTOR_MAP,
        "sector_colors": SECTOR_COLORS
    }

    # Training Checkpoints
    training_metrics_csv = os.path.join(BASE_DIR, "results", "ppo_logs", "training_metrics.csv")
    checkpoints = []
    if os.path.exists(training_metrics_csv):
        df_tm = pd.read_csv(training_metrics_csv)
        for _, r in df_tm.iterrows():
            checkpoints.append({
                "timestep": int(r["timestep"]),
                "episode_reward": round(float(r["episode_reward"]), 2),
                "episode_length": int(r["episode_length"])
            })

    output_payload = {
        "meta": {
            "title": "Finora AI Portfolio Management Intelligence",
            "version": "2.0.0",
            "test_period": "2022-12-29 to 2024-12-31 (504 Trading Days)",
            "transaction_cost_bps": 10.0,
            "evaluation_seeds": [42, 101, 202, 303, 404]
        },
        "trajectories_test": trajectories_test,
        "rigorous_summary": summary_data,
        "in_sample": in_sample_data,
        "asset_universe": asset_stats,
        "graph_data": graph_data,
        "checkpoints": checkpoints
    }

    out_file = os.path.join(DATA_DIR, "model_outputs.json")
    with open(out_file, "w") as f:
        json.dump(output_payload, f, indent=2)
    print(f"[SUCCESS] Saved frontend data to: {out_file}")

    # Also generate data.js for static / zero-dependency browser loading
    js_file = os.path.join(FRONTEND_DIR, "js", "data.js")
    os.makedirs(os.path.dirname(js_file), exist_ok=True)
    with open(js_file, "w", encoding="utf-8") as f:
        f.write(f"// Auto-generated Finora Model Data\nconst FINORA_DATA = {json.dumps(output_payload)};\n")
    print(f"[SUCCESS] Saved standalone data script to: {js_file}")

if __name__ == "__main__":
    extract_all()
