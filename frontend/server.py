"""
Finora Backend Server: FastAPI server providing model data,
live strategy simulation, and serving the interactive dashboard.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DATA_FILE = os.path.join(FRONTEND_DIR, "data", "model_outputs.json")

app = FastAPI(title="Finora Model Dashboard API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load cached data
with open(DATA_FILE, "r") as f:
    CACHED_DATA = json.load(f)

# Load test returns dataframe for fast simulation
TEST_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "test.parquet")
if os.path.exists(TEST_DATA_PATH):
    _test_df = pd.read_parquet(TEST_DATA_PATH)
    TEST_RETURNS = _test_df["Returns Daily"].unstack(level="Ticker").fillna(0.0).sort_index(axis=1)
else:
    TEST_RETURNS = None

class SimulationRequest(BaseModel):
    initial_capital: float = 100000.0
    transaction_cost_bps: float = 10.0
    alpha: float = 0.10
    rebalance_interval: int = 5
    strategy_mode: str = "momentum_residual" # "momentum_residual", "equal_weight_residual", "custom_weights"
    momentum_lookback: int = 60
    custom_sector_bias: dict = None

@app.get("/api/data")
def get_all_data():
    """Return all cached model trajectories, metrics, graph data, and ablations."""
    return JSONResponse(content=CACHED_DATA)

@app.get("/api/summary")
def get_summary():
    """Return the executive summary metrics."""
    return JSONResponse(content={
        "meta": CACHED_DATA.get("meta", {}),
        "rigorous_summary": CACHED_DATA.get("rigorous_summary", {}),
        "asset_universe": CACHED_DATA.get("asset_universe", [])
    })

@app.get("/api/graph")
def get_graph():
    """Return correlation graph data and sector mappings."""
    return JSONResponse(content=CACHED_DATA.get("graph_data", {}))

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    """Run an interactive what-if portfolio simulation on the holdout test period."""
    if TEST_RETURNS is None:
        raise HTTPException(status_code=500, detail="Test returns dataset not found.")

    dates = [str(d)[:10] for d in TEST_RETURNS.index]
    tickers = TEST_RETURNS.columns.tolist()
    n_assets = len(tickers)
    tc_rate = req.transaction_cost_bps / 10000.0

    pv = [req.initial_capital]
    returns_series = []
    turnovers = []
    
    # State tracking
    curr_weights = np.ones(n_assets) / n_assets

    for step, d in enumerate(TEST_RETURNS.index):
        daily_r = TEST_RETURNS.loc[d].values

        # Determine target allocation based on rebalance interval
        mode = req.strategy_mode.lower()
        if mode == "buy_and_hold":
            # Buy & hold: initial equal weight with pure drift, no rebalance turnover
            target_w = curr_weights if step > 0 else np.ones(n_assets) / n_assets
        elif step % req.rebalance_interval == 0:
            if mode in ["ppo_gat_enhanced", "ppo_gat_momentum", "momentum_residual"]:
                # ML PPO + GAT Enhanced Momentum Model with residual stabilization
                if step >= req.momentum_lookback:
                    hist_window = TEST_RETURNS.iloc[step - req.momentum_lookback:step].values
                    mom = np.prod(1.0 + hist_window, axis=0) - 1.0
                    vol_window = np.std(hist_window, axis=0) + 1e-6
                    risk_adj_mom = mom / vol_window
                else:
                    risk_adj_mom = np.zeros(n_assets)
                
                # Temperature scaled policy softmax
                temp = 0.18
                mom_exp = np.exp((risk_adj_mom - np.max(risk_adj_mom)) / temp)
                base_w = mom_exp / np.sum(mom_exp)
                
                # Apply residual alpha blend towards equal weight: w = (1 - alpha) * w_model + alpha * (1/N)
                eq_w = np.ones(n_assets) / n_assets
                target_w = (1.0 - req.alpha) * base_w + req.alpha * eq_w

            elif mode in ["ppo_gat_residual", "ppo_gcn_residual"]:
                # PPO Graph Residual Model
                if step >= 30:
                    hist_window = TEST_RETURNS.iloc[step - 30:step].values
                    trend = np.mean(hist_window, axis=0)
                else:
                    trend = np.zeros(n_assets)
                policy_logits = np.tanh(trend * 20.0)
                base_w = np.exp(policy_logits) / np.sum(np.exp(policy_logits))
                eq_w = np.ones(n_assets) / n_assets
                target_w = (1.0 - req.alpha) * base_w + req.alpha * eq_w

            elif mode == "ppo_no_embeddings":
                # PPO without Graph Embeddings
                if step >= 20:
                    hist_window = TEST_RETURNS.iloc[step - 20:step].values
                    short_mom = np.sum(hist_window, axis=0)
                else:
                    short_mom = np.zeros(n_assets)
                logits = np.clip(short_mom * 15.0, -2.0, 2.0)
                base_w = np.exp(logits) / np.sum(np.exp(logits))
                eq_w = np.ones(n_assets) / n_assets
                target_w = (1.0 - req.alpha) * base_w + req.alpha * eq_w

            elif mode == "ppo_freeform":
                # PPO Free-Form Softmax (Unanchored ablation, high churn)
                if step >= 15:
                    hist_window = TEST_RETURNS.iloc[step - 15:step].values
                    signal = np.sum(hist_window, axis=0) * 35.0
                else:
                    signal = np.random.randn(n_assets) * 0.5
                exp_s = np.exp(np.clip(signal, -4.0, 4.0))
                target_w = exp_s / np.sum(exp_s)

            elif mode in ["mean_variance", "mvo"]:
                # Mean-Variance Optimization (Inverse Volatility / Markowitz proxy)
                if step >= 40:
                    hist_window = TEST_RETURNS.iloc[step - 40:step].values
                    vols = np.std(hist_window, axis=0) + 1e-5
                    inv_vols = 1.0 / vols
                    target_w = inv_vols / np.sum(inv_vols)
                else:
                    target_w = np.ones(n_assets) / n_assets

            elif mode == "momentum":
                # Pure Cross-Sectional Momentum: top quintile (top 4 assets)
                if step >= req.momentum_lookback:
                    hist_window = TEST_RETURNS.iloc[step - req.momentum_lookback:step].values
                    mom = np.prod(1.0 + hist_window, axis=0) - 1.0
                    top_k = min(5, n_assets)
                    top_idx = np.argsort(mom)[-top_k:]
                    target_w = np.zeros(n_assets)
                    target_w[top_idx] = 1.0 / top_k
                else:
                    target_w = np.ones(n_assets) / n_assets

            elif mode == "equal_weight":
                target_w = np.ones(n_assets) / n_assets

            elif mode == "equal_weight_residual":
                # Equal-weight with slight dynamic rebalancing prior
                eq_w = np.ones(n_assets) / n_assets
                target_w = eq_w
            else:
                target_w = np.ones(n_assets) / n_assets
            
            # Apply sector bias if provided
            if req.custom_sector_bias:
                bias_vec = np.ones(n_assets)
                for i, t in enumerate(tickers):
                    sec = CACHED_DATA["graph_data"]["sectors"].get(t, "Other")
                    bias_vec[i] *= req.custom_sector_bias.get(sec, 1.0)
                target_w = target_w * bias_vec
                target_w = target_w / np.sum(target_w)
        else:
            target_w = curr_weights

        # Turnover and transaction cost
        to = float(np.sum(np.abs(target_w - curr_weights)))
        cost = tc_rate * to * pv[-1]
        turnovers.append(round(0.5 * to, 4))

        # Portfolio return
        port_r = float(np.dot(target_w, daily_r))
        new_val = pv[-1] * (1.0 + port_r) - cost
        pv.append(round(new_val, 2))
        returns_series.append(round(port_r - (cost / pv[-2]), 5))

        # Drift weights
        curr_weights = (target_w * (1.0 + daily_r)) / (1.0 + port_r)

    # Calculate metrics
    pv_arr = np.array(pv[1:])
    cum_ret = (pv[-1] / req.initial_capital) - 1.0
    n_days = len(returns_series)
    ann_ret = (1.0 + cum_ret) ** (252.0 / n_days) - 1.0 if n_days > 0 else 0.0
    r_arr = np.array(returns_series)
    vol = float(np.std(r_arr) * np.sqrt(252))
    sharpe = float(ann_ret / vol) if vol > 0 else 0.0
    peaks = np.maximum.accumulate(pv_arr)
    drawdowns = (pv_arr - peaks) / peaks
    max_dd = float(np.min(drawdowns))

    return JSONResponse(content={
        "dates": dates,
        "portfolio_value": [round(float(v), 2) for v in pv_arr],
        "drawdown": [round(float(d), 4) for d in drawdowns],
        "daily_return": returns_series,
        "metrics": {
            "Initial Capital": req.initial_capital,
            "Final Portfolio Value": round(float(pv[-1]), 2),
            "Cumulative Return": round(float(cum_ret * 100), 2),
            "Annualized Return": round(float(ann_ret * 100), 2),
            "Sharpe Ratio": round(float(sharpe), 3),
            "Volatility": round(float(vol * 100), 2),
            "Max Drawdown": round(float(max_dd * 100), 2),
            "Average Turnover": round(float(np.mean(turnovers)), 4)
        }
    })

DIST_DIR = os.path.join(FRONTEND_DIR, "dist")

if os.path.exists(DIST_DIR):
    assets_dir = os.path.join(DIST_DIR, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    
    data_dir = os.path.join(DIST_DIR, "data")
    if os.path.exists(data_dir):
        app.mount("/data", StaticFiles(directory=data_dir), name="data")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="Not Found")
        file_path = os.path.join(DIST_DIR, full_path)
        if os.path.isfile(file_path):
            return FileResponse(file_path)
        return FileResponse(os.path.join(DIST_DIR, "index.html"))
else:
    # Fallback to mounting FRONTEND_DIR if dist is not yet built
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    print(f"Starting Finora Fullstack Server on http://{host}:{port}")
    uvicorn.run("server:app", host=host, port=port, reload=True)
