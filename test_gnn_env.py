import pandas as pd

from src.environment.gnn_portfolio_env import GNNPortfolioEnv

DATA_PATH = "data/clean/full_dataset.parquet"
EMBEDDING_DIR = "graphs/gnn_models/embeddings"

data = pd.read_parquet(DATA_PATH)

env = GNNPortfolioEnv(
    data=data,
    embedding_dir=EMBEDDING_DIR
)

obs, info = env.reset(seed=0)

print("=" * 60)
print("GNN PORTFOLIO ENVIRONMENT TEST")
print("=" * 60)

print("Environment created")
print("Assets:", env.n_assets)
print("Observation type:", type(obs))
print("Observation keys:", obs.keys())

for key, value in obs.items():
    print(f"{key} shape:", value.shape)

print("Initial portfolio value:", info["portfolio_value"])

action = env.action_space.sample()

obs, reward, terminated, truncated, info = env.step(action)

print("\n===== STEP TEST =====")
print("Reward:", reward)
print("Portfolio value:", info["portfolio_value"])
print("Terminated:", terminated)
print("Truncated:", truncated)

print("\n===== COMPLETED =====")
