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
        if step % req.rebalance_interval == 0:
            if req.strategy_mode == "momentum_residual":
                # Compute historical momentum from test_returns up to this step
                if step >= req.momentum_lookback:
                    hist_window = TEST_RETURNS.iloc[step - req.momentum_lookback:step].values
                    mom = np.prod(1.0 + hist_window, axis=0) - 1.0
                else:
                    mom = np.zeros(n_assets)
                
                # Temperature scaled softmax
                temp = 0.2
                mom_exp = np.exp((mom - np.max(mom)) / temp)
                base_w = mom_exp / np.sum(mom_exp)
                
                # Apply residual alpha blend towards equal weight
                eq_w = np.ones(n_assets) / n_assets
                target_w = (1.0 - req.alpha) * base_w + req.alpha * eq_w
            elif req.strategy_mode == "equal_weight_residual":
                target_w = np.ones(n_assets) / n_assets
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

# Mount static frontend
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
