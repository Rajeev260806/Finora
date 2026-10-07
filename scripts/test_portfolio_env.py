"""
Phase 3 (Person B) — Comprehensive PortfolioEnv Test Suite
==========================================================
Validates PortfolioEnv implementation according to Phase 3 objectives:
1. Basic functionality & Gymnasium compliance
2. Dual observation modes (returns_window vs placeholder/real GNN embeddings)
3. Loading of real GNN embedding files (.npy)
4. Economic logic (turnover costs, cash 0 return, portfolio growth, softmax property)
5. Phase 2 risk-adjusted reward integration
"""

from __future__ import annotations

import os
import shutil
import tempfile
import numpy as np
import pandas as pd

from src.environment.portfolioenv import PortfolioEnv


def load_sample_data() -> pd.DataFrame:
    data_path = os.path.join("data", "clean", "train.parquet")
    if not os.path.exists(data_path):
        data_path = os.path.join("data", "clean", "full_dataset.parquet")
    return pd.read_parquet(data_path)


def test_basic_functionality():
    print("--- Running Test 1: Basic Functionality & Shapes ---")
    data = load_sample_data()
    env = PortfolioEnv(data, window=20, transaction_cost_bps=10.0)

    obs, info = env.reset(seed=42)
    n_assets = env.n_assets

    assert "returns_window" in obs, "Observation must contain 'returns_window'"
    assert "weights" in obs, "Observation must contain 'weights'"
    assert obs["returns_window"].shape == (n_assets, 20), f"Expected shape {(n_assets, 20)}, got {obs['returns_window'].shape}"
    assert obs["weights"].shape == (n_assets + 1,), f"Expected shape {(n_assets + 1,)}, got {obs['weights'].shape}"
    assert np.isclose(obs["weights"][-1], 1.0), "Initial allocation should be 100% cash"
    assert info["portfolio_value"] == 1.0

    done = False
    step_count = 0
    while not done and step_count < 20:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        step_count += 1
        done = terminated or truncated

        assert isinstance(reward, float)
        assert not np.isnan(reward), "Reward must not be NaN"
        assert not np.isinf(reward), "Reward must not be Inf"
        assert obs["returns_window"].shape == (n_assets, 20)
        assert obs["weights"].shape == (n_assets + 1,)
        assert np.isclose(np.sum(obs["weights"]), 1.0, atol=1e-5), "Weights must sum to 1.0"

    print(f"[OK] Test 1 Passed: Executed {step_count} steps successfully.")


def test_embedding_mode_placeholder():
    print("--- Running Test 2: Placeholder GNN Embedding Mode ---")
    data = load_sample_data()
    embedding_dim = 32
    env = PortfolioEnv(data, window=20, use_embeddings=True, embedding_dim=embedding_dim, seed=42)

    obs, info = env.reset()
    n_assets = env.n_assets

    assert "gnn_embeddings" in obs, "Observation must contain 'gnn_embeddings' when use_embeddings=True"
    assert "weights" in obs, "Observation must contain 'weights'"
    assert obs["gnn_embeddings"].shape == (n_assets, embedding_dim), f"Expected shape {(n_assets, embedding_dim)}, got {obs['gnn_embeddings'].shape}"
    assert obs["gnn_embeddings"].dtype == np.float32

    # Step through environment
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs["gnn_embeddings"].shape == (n_assets, embedding_dim)
        assert not np.isnan(obs["gnn_embeddings"]).any()
        if terminated or truncated:
            break

    print("[OK] Test 2 Passed: Placeholder GNN Embedding mode operates correctly.")


def test_real_embedding_files_loading():
    print("--- Running Test 3: Real Embedding File (.npy) Loading ---")
    data = load_sample_data()
    n_assets = len(data.index.get_level_values(-1).unique())
    embedding_dim = 32

    # Create temp directory with mock GNN embedding files
    tmp_dir = tempfile.mkdtemp()
    try:
        # Create embedding file for step index 20
        mock_emb_20 = np.full((n_assets, embedding_dim), 0.75, dtype=np.float32)
        np.save(os.path.join(tmp_dir, "embeddings_graph_0020.npy"), mock_emb_20)

        # Create embedding file for step index 21
        mock_emb_21 = np.full((n_assets, embedding_dim), -0.50, dtype=np.float32)
        np.save(os.path.join(tmp_dir, "embeddings_graph_0021.npy"), mock_emb_21)

        env = PortfolioEnv(
            data,
            window=20,
            use_embeddings=True,
            embedding_dim=embedding_dim,
            embedding_dir=tmp_dir,
            seed=42,
        )

        obs, _ = env.reset()
        np.testing.assert_array_almost_equal(obs["gnn_embeddings"], mock_emb_20)

        # Execute 1 step -> should load embeddings_graph_0021.npy
        action = np.zeros(n_assets + 1, dtype=np.float32)
        obs, _, _, _, _ = env.step(action)
        np.testing.assert_array_almost_equal(obs["gnn_embeddings"], mock_emb_21)

        print("[OK] Test 3 Passed: Successfully loaded pre-computed GNN embedding .npy files.")
    finally:
        shutil.rmtree(tmp_dir)


def test_economic_logic():
    print("--- Running Test 4: Economic Logic Verification ---")
    data = load_sample_data()
    env = PortfolioEnv(data, window=20, transaction_cost_bps=10.0, initial_cash=1.0)
    n_assets = env.n_assets

    # Scenario A: All Cash Action
    # Action logits with huge value for cash slot
    all_cash_action = np.full(n_assets + 1, -10.0, dtype=np.float32)
    all_cash_action[-1] = 10.0  # cash logit

    env.reset(seed=42)
    # Step 1: initial allocation was already 100% cash, staying 100% cash -> 0 turnover, 0 cost, 0 return
    obs, reward, terminated, truncated, info = env.step(all_cash_action)
    assert np.isclose(info["turnover"], 0.0, atol=1e-5), f"Turnover staying in cash should be 0, got {info['turnover']}"
    assert np.isclose(info["transaction_cost"], 0.0, atol=1e-5)
    assert np.isclose(info["portfolio_value"], 1.0, atol=1e-5), f"Cash portfolio value should remain 1.0, got {info['portfolio_value']}"

    # Scenario B: High Turnover Cost Drag
    env.reset(seed=42)
    total_cost = 0.0
    for i in range(10):
        # Alternate 100% asset 0 and 100% asset 1
        action = np.full(n_assets + 1, -10.0, dtype=np.float32)
        target_asset = i % 2
        action[target_asset] = 10.0
        _, _, _, _, info = env.step(action)
        total_cost += info["transaction_cost"]

    assert total_cost > 0.0, "Frequent rebalancing must accumulate positive transaction costs"
    print(f"[OK] Test 4 Passed: Economic logic verified (Cash value preserved, total churn cost = {total_cost:.6f}).")


def test_phase2_reward_integration():
    print("--- Running Test 5: Phase 2 Risk-Adjusted Reward Model ---")
    data = load_sample_data()

    # Env with Phase 2 volatility and concentration penalties active
    env_p2 = PortfolioEnv(
        data,
        window=20,
        transaction_cost_bps=10.0,
        lambda_vol=0.10,
        lambda_conc=0.01,
        seed=42,
    )
    n_assets = env_p2.n_assets

    # Single-stock action (100% asset 0) -> high HHI concentration (1.0)
    single_stock_action = np.full(n_assets + 1, -10.0, dtype=np.float32)
    single_stock_action[0] = 10.0

    # Equal weight action -> low HHI concentration (1/N)
    equal_weight_action = np.full(n_assets + 1, 0.0, dtype=np.float32)
    equal_weight_action[-1] = -10.0  # exclude cash, split among stocks

    env_p2.reset()
    _, reward_single, _, _, _ = env_p2.step(single_stock_action)

    env_p2.reset()
    _, reward_eq, _, _, _ = env_p2.step(equal_weight_action)

    # Both rewards should be finite numbers
    assert not np.isnan(reward_single) and not np.isnan(reward_eq)
    print(f"[OK] Test 5 Passed: Phase 2 reward calculated (Single stock reward: {reward_single:.5f}, Equal weight reward: {reward_eq:.5f}).")


def test_gym_compliance():
    print("--- Running Test 6: Gymnasium Space Compliance ---")
    data = load_sample_data()

    # Test returns_window mode
    env1 = PortfolioEnv(data, use_embeddings=False)
    obs1, _ = env1.reset()
    assert env1.observation_space.contains(obs1), "Observation 1 must be contained within observation_space"
    sample_act1 = env1.action_space.sample()
    assert env1.action_space.contains(sample_act1), "Sample action 1 must be contained within action_space"

    # Test embeddings mode
    env2 = PortfolioEnv(data, use_embeddings=True, embedding_dim=32)
    obs2, _ = env2.reset()
    assert env2.observation_space.contains(obs2), "Observation 2 must be contained within observation_space"
    sample_act2 = env2.action_space.sample()
    assert env2.action_space.contains(sample_act2), "Sample action 2 must be contained within action_space"

    print("[OK] Test 6 Passed: Gymnasium spaces compliance verified.")


if __name__ == "__main__":
    print("=========================================================")
    print("  FINORA PHASE 3: PORTFOLIO ENVIRONMENT TEST SUITE  ")
    print("=========================================================\n")
    test_basic_functionality()
    test_embedding_mode_placeholder()
    test_real_embedding_files_loading()
    test_economic_logic()
    test_phase2_reward_integration()
    test_gym_compliance()
    print("\n[SUCCESS] ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
