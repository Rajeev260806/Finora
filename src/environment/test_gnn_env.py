import pandas as pd
from src.environment.gnn_portfolio_env import GNNPortfolioEnv

data = pd.read_parquet("data/clean/full_dataset.parquet")

env = GNNPortfolioEnv(
    data=data,
    embedding_dir="graphs/gnn_models/embeddings",
    window=20,
    transaction_cost_bps=10.0,
    initial_cash=1.0,
    return_col="Returns Daily",
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