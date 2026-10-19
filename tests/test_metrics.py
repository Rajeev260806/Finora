"""
Unit Tests for Shared Metrics Module
===================================
Tests required by PROJECT_ACTION_PLAN.md:
1. For constant daily return r, annualized return must equal (1+r)^252 - 1.
2. Sharpe ratio equals (mean/std) * sqrt(252) under known distribution.
3. Max drawdown accurately measures drop from peak.
4. One-way turnover calculation equals 0.5 * sum(|w_t - w_{t-1}|).
"""

import sys
import os
import unittest
import numpy as np
import pandas as pd

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.analysis.metrics import compute_metrics, compute_turnover, compute_metrics_from_history


class TestMetrics(unittest.TestCase):

    def test_constant_return_annualization(self):
        """
        Action Plan requirement:
        For a constant daily return r, annualized return must equal (1+r)^252 - 1.
        """
        r = 0.001  # 10 bps daily return
        n_days = 252  # exactly 1 year of trading days
        
        # Construct portfolio values: V_0 = 1.0, V_t = V_{t-1} * (1 + r)
        pv = np.cumprod(np.insert(np.full(n_days, 1.0 + r), 0, 1.0))
        ret = np.full(n_days, r)

        metrics = compute_metrics(portfolio_values=pv, daily_returns=ret)
        
        expected_ann_ret = (1.0 + r) ** 252 - 1.0
        self.assertAlmostEqual(metrics["Annualized Return"], expected_ann_ret, places=6)
        self.assertAlmostEqual(metrics["Cumulative Return"], expected_ann_ret, places=6)

        # Test for different duration (e.g. 504 days = 2 years)
        n_days_2y = 504
        pv_2y = np.cumprod(np.insert(np.full(n_days_2y, 1.0 + r), 0, 1.0))
        ret_2y = np.full(n_days_2y, r)
        metrics_2y = compute_metrics(portfolio_values=pv_2y, daily_returns=ret_2y)
        self.assertAlmostEqual(metrics_2y["Annualized Return"], expected_ann_ret, places=6)

    def test_sharpe_ratio_formula(self):
        """
        Check: Sharpe = mean/std * sqrt(252).
        """
        np.random.seed(42)
        n_days = 1000
        # Synthetic daily returns with known mean and std
        ret = np.random.normal(loc=0.0005, scale=0.01, size=n_days)
        pv = np.cumprod(np.insert(1.0 + ret, 0, 1.0))

        metrics = compute_metrics(portfolio_values=pv, daily_returns=ret)
        expected_sharpe = float(np.mean(ret) / np.std(ret, ddof=1) * np.sqrt(252))
        expected_vol = float(np.std(ret, ddof=1) * np.sqrt(252))

        self.assertAlmostEqual(metrics["Sharpe Ratio"], expected_sharpe, places=6)
        self.assertAlmostEqual(metrics["Volatility"], expected_vol, places=6)

    def test_max_drawdown(self):
        """
        Verify peak-to-trough max drawdown calculation.
        """
        # Values: 100 -> 120 (peak) -> 90 (trough: -25%) -> 110 -> 100
        pv = np.array([100.0, 120.0, 90.0, 110.0, 100.0])
        metrics = compute_metrics(portfolio_values=pv)
        # Expected max drawdown: (90 - 120) / 120 = -30 / 120 = -0.25 (-25%)
        self.assertAlmostEqual(metrics["Max Drawdown"], -0.25, places=6)

    def test_one_way_turnover(self):
        """
        Check: turnover = 0.5 * sum(|w_t - w_{t-1}|).
        """
        # Day 0: 100% Asset A, 0% Asset B
        # Day 1: 50% Asset A, 50% Asset B -> |0.5 - 1.0| + |0.5 - 0.0| = 0.5 + 0.5 = 1.0 total churn.
        # One-way turnover: 0.5 * 1.0 = 0.5 (sold 50% of A, bought 50% of B).
        # Day 2: 0% Asset A, 100% Asset B -> One-way turnover = 0.5 * (0.5 + 0.5) = 0.5.
        weights = np.array([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.0, 1.0]
        ])
        turnovers = compute_turnover(weights)
        self.assertEqual(len(turnovers), 2)
        self.assertAlmostEqual(turnovers[0], 0.5, places=6)
        self.assertAlmostEqual(turnovers[1], 0.5, places=6)

        # Average turnover in metrics:
        pv = np.array([1.0, 1.01, 1.02])
        metrics = compute_metrics(portfolio_values=pv, weights=weights)
        self.assertAlmostEqual(metrics["Average Turnover"], 0.5, places=6)

    def test_compute_metrics_from_history(self):
        """
        Test compatibility with environment history DataFrame.
        """
        df = pd.DataFrame({
            "portfolio_value": [1.0, 1.02, 1.01, 1.05],
            "portfolio_return": [0.0, 0.02, -0.0098, 0.0396],
            "turnover": [0.0, 0.1, 0.05, 0.02],
        })
        metrics = compute_metrics_from_history(df)
        self.assertAlmostEqual(metrics["Final Portfolio Value"], 1.05, places=6)
        self.assertAlmostEqual(metrics["Average Turnover"], np.mean([0.0, 0.1, 0.05, 0.02]), places=6)


if __name__ == "__main__":
    unittest.main()
