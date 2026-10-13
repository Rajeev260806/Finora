import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pandas as pd
from src.environment.gnn_portfolio_env import GNNPortfolioEnv

data = pd.read_parquet("data/clean/full_dataset.parquet")

env = GNNPortfolioEnv(
    data=data,
    embedding_dir="src/gnn/embeddings",
    embedding_dim=32,
    window=20,
    transaction_cost_bps=10.0,
    initial_cash=1.0,
    return_col="Returns Daily",
    lambda_vol=0.10,
    lambda_conc=0.01,
    lambda_txn=1.0,
    seed=42
)

obs, info = env.reset(seed=42)

print("===== GNN ENVIRONMENT TEST =====")
print("Assets:", env.n_assets)
print("Embedding dimension:", env.embedding_dim)
print("Embedding shape:", obs["gnn_embeddings"].shape)
print("Weights shape:", obs["weights"].shape)
print("Portfolio value:", info["portfolio_value"])

action = env.action_space.sample()

obs, reward, terminated, truncated, info = env.step(action)

print()
print("===== STEP TEST =====")
print("Reward:", reward)
print("Portfolio value:", info["portfolio_value"])
print("Embedding shape:", obs["gnn_embeddings"].shape)
print("Weights shape:", obs["weights"].shape)
print("Terminated:", terminated)
print("Truncated:", truncated)