import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from src.environment.gnn_portfolio_env import GNNPortfolioEnv


DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clean",
    "train.parquet"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "src",
    "gnn",
    "embeddings_gat" if os.path.exists(os.path.join(BASE_DIR, "src", "gnn", "embeddings_gat")) else "embeddings"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "src",
    "gnn",
    "models"
)

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 60)
print("ENHANCED GNN + PPO TRAINING (Momentum Residual, Weekly Rebalance)")
print("=" * 60)


# Load dataset
data = pd.read_parquet(DATA_PATH)

print("Dataset loaded:", DATA_PATH)
print("Rows:", len(data))


# Create GNN portfolio environment
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
    lambda_vol=0.10,
    lambda_conc=0.001,
    lambda_txn=1.0,
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
    total_timesteps=20000
)


# Save model
model_path = os.path.join(
    MODEL_DIR,
    "gnn_ppo"
)
model.save(model_path)

root_model_path = os.path.join(BASE_DIR, "models", "gnn_ppo")
os.makedirs(os.path.dirname(root_model_path), exist_ok=True)
model.save(root_model_path)

print("\n" + "=" * 60)
print("PPO TRAINING COMPLETED")
print("=" * 60)
print("Model saved to:", model_path + ".zip")
print("Root model saved to:", root_model_path + ".zip")