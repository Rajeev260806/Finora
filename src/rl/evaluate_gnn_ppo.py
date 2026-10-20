import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from stable_baselines3 import PPO

from src.environment.gnn_portfolio_env import GNNPortfolioEnv
from src.analysis.metrics import compute_metrics_from_history

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clean",
    "test.parquet"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "src",
    "gnn",
    "embeddings_gat" if os.path.exists(os.path.join(BASE_DIR, "src", "gnn", "embeddings_gat")) else "embeddings"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "src",
    "gnn",
    "models",
    "gnn_ppo"
)


data = pd.read_parquet(DATA_PATH)

env = GNNPortfolioEnv(
    data=data,
    embedding_dir=EMBEDDING_DIR,
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

model = PPO.load(MODEL_PATH, env=env)

obs, info = env.reset(seed=0)

total_reward = 0.0
steps = 0

while True:
    action, _ = model.predict(obs, deterministic=True)

    obs, reward, terminated, truncated, info = env.step(action)

    total_reward += reward
    steps += 1

    if terminated or truncated:
        break

history = env.get_history_df()
metrics = compute_metrics_from_history(history)

print("=" * 60)
print("ENHANCED GNN + PPO OUT-OF-SAMPLE EVALUATION (TEST PERIOD)")
print("=" * 60)

for k, v in metrics.items():
    if isinstance(v, float):
        print(f"{k:<25}: {v:.4f}")
    else:
        print(f"{k:<25}: {v}")

print("Total reward:", total_reward)
print("=" * 60)