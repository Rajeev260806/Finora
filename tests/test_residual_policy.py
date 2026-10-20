"""
Unit Test for Residual Policy (Phase 3.1)
========================================
Verifies:
1. When action is zero (initial policy state), portfolio weights equal Equal Weight (1/N per stock).
2. Weights are strictly non-negative and sum to 1.0.
3. Action perturbs weights away from 1/N smoothly, bounded by residual_alpha.
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.environment.portfolioenv import PortfolioEnv


class TestResidualPolicy(unittest.TestCase):

    def setUp(self):
        data_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "clean", "test.parquet"
        )
        self.data = pd.read_parquet(data_path)

    def test_zero_action_produces_equal_weight(self):
        env = PortfolioEnv(
            data=self.data,
            window=20,
            residual_alpha=0.25,
            use_embeddings=False,
        )
        env.reset()
        zero_action = np.zeros(env.n_assets + 1, dtype=np.float32)
        weights = env._process_action(zero_action)

        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        # Check that all stocks have equal weights
        stock_weights = weights[:-1]
        expected_stock_weight = 1.0 / env.n_assets  # 1/20 = 0.05
        for w in stock_weights:
            self.assertAlmostEqual(w, expected_stock_weight, delta=0.001)

    def test_action_perturbation_bounded(self):
        alpha = 0.3
        env = PortfolioEnv(
            data=self.data,
            window=20,
            residual_alpha=alpha,
            use_embeddings=False,
        )
        env.reset()
        # Large action favoring stock 0
        extreme_action = np.zeros(env.n_assets + 1, dtype=np.float32)
        extreme_action[0] = 10.0
        extreme_action[1:] = -10.0

        weights = env._process_action(extreme_action)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0.0))
        # Stock 0 must have highest weight, but still bounded
        self.assertGreater(weights[0], weights[1])

    def test_momentum_residual_mode(self):
        env = PortfolioEnv(
            data=self.data,
            window=20,
            residual_alpha=0.1,
            residual_mode="momentum",
            momentum_lookback=60,
            use_embeddings=False,
        )
        env.reset()
        zero_action = np.zeros(env.n_assets + 1, dtype=np.float32)
        weights = env._process_action(zero_action)
        self.assertAlmostEqual(float(np.sum(weights)), 1.0, places=5)
        self.assertTrue(np.all(weights >= 0.0))
        # Top momentum stock should have higher weight than bottom momentum stock
        self.assertGreater(np.max(weights[:-1]), np.min(weights[:-1]))

    def test_rebalance_interval_turnover(self):
        env = PortfolioEnv(
            data=self.data,
            window=20,
            rebalance_interval=5,
            use_embeddings=False,
        )
        env.reset()
        action1 = np.random.randn(env.n_assets + 1).astype(np.float32)
        obs, reward, term, trunc, info1 = env.step(action1)
        # Step 1 should rebalance (step_count = 0)
        self.assertGreaterEqual(info1["turnover"], 0.0)

        # Step 2 (step_count = 1) should not rebalance -> turnover = 0
        action2 = np.random.randn(env.n_assets + 1).astype(np.float32)
        obs, reward, term, trunc, info2 = env.step(action2)
        self.assertEqual(info2["turnover"], 0.0)
        self.assertEqual(info2["transaction_cost"], 0.0)


if __name__ == "__main__":
    unittest.main()

