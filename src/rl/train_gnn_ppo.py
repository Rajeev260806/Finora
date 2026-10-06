import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.environment.gnn_portfolio_env import GNNPortfolioEnv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "src",
    "gnn",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 60)
print("GNN + PPO TRAINING")
print("=" * 60)


# Load dataset
data = pd.read_parquet(DATA_PATH)

print("Dataset loaded")
print("Rows:", len(data))


# Create GNN portfolio environment
env = GNNPortfolioEnv(
    data=data,
    embedding_dir=EMBEDDING_DIR,
    window=20,
    transaction_cost_bps=10.0,
    initial_cash=1.0,
    return_col="Returns Daily",
    seed=0,
)

print("Environment created")
print("Assets:", env.n_assets)
print("Embedding dimension:", env.embedding_dim)


# Validate environment
print("\nChecking environment...")
check_env(env, warn=True)

print("Environment check passed")


# Create PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    gamma=0.99,
    seed=0,
)


print("\nStarting PPO training...")


# Train
model.learn(
    total_timesteps=10000
)


# Save model
model_path = os.path.join(
    MODEL_DIR,
    "gnn_ppo"
)

model.save(model_path)

print("\n" + "=" * 60)
print("PPO TRAINING COMPLETED")
print("=" * 60)
print("Model saved to:", model_path)