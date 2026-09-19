import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO

from src.environment.gnn_portfolio_env import GNNPortfolioEnv


DATA_PATH = "data/clean/full_dataset.parquet"
EMBEDDING_DIR = "graphs/gnn_models/embeddings"
MODEL_PATH = "models/gnn_ppo"

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

values = [1.0]

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

    values.append(info["portfolio_value"])

    if terminated or truncated:
        break

ppo_final = values[-1]

results = pd.DataFrame({
    "Strategy": [
        "Buy and Hold",
        "Equal Weight",
        "Mean Variance",
        "Momentum",
        "GNN + PPO"
    ],
    "Final Value": [
        17.3495,
        4.8701,
        5.0307,
        6.8297,
        ppo_final
    ]
})

print("=" * 60)
print("PHASE 2 STRATEGY COMPARISON")
print("=" * 60)
print(results.to_string(index=False))

results.to_csv(
    "gnn_ppo_comparison.csv",
    index=False
)

plt.figure(figsize=(10, 6))
plt.bar(results["Strategy"], results["Final Value"])
plt.ylabel("Final Portfolio Value")
plt.title("Phase 1 Baselines vs GNN + PPO")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("gnn_ppo_comparison.png")

print("\nSaved:")
print("gnn_ppo_comparison.csv")
print("gnn_ppo_comparison.png")