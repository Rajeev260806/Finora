import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from stable_baselines3 import PPO

from src.environment.gnn_portfolio_env import GNNPortfolioEnv

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clean",
    "full_dataset.parquet"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "src",
    "gnn",
    "embeddings"
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

print("=" * 60)
print("GNN + PPO EVALUATION")
print("=" * 60)

print("Steps:", steps)
print("Final portfolio value:", info["portfolio_value"])
print("Total reward:", total_reward)

if len(history) > 0:
    print("Total transaction cost:", history["transaction_cost"].sum())
    print("Average turnover:", history["turnover"].mean())

print("=" * 60)