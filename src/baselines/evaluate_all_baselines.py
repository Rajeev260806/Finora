"""
Unified Baseline Evaluation Module for Finora
=============================================
Runs all 4 baselines on the EXACT same test period (2022-12-29 to 2024-12-31),
same initial value ($1.0), and same transaction costs (10 bps).
Uses the shared metrics module `src/analysis/metrics.py`.

Strategies:
1. Buy & Hold: Initial equal weight across all 20 assets, zero rebalancing.
2. Equal Weight: Monthly (21 trading days) rebalancing to 1/N across all 20 assets.
3. Momentum: 12-1 month lookback (252 days lookback, skip 21 days), top 5 assets, 80% stocks, 20% cash, monthly rebalancing.
4. Mean-Variance: 60-day historical window, risk_aversion=1.0, long-only, monthly rebalancing.
"""

import os
import sys
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from src.analysis.metrics import compute_metrics

DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")


def evaluate_baselines(data_path=DATA_PATH, transaction_cost_bps=10.0, test_start="2022-12-29"):
    full_df = pd.read_parquet(data_path)
    wide_returns = full_df["Returns Daily"].unstack(level="Ticker").fillna(0.0).sort_index(axis=1)
    tickers = wide_returns.columns.tolist()
    n_assets = len(tickers)
    dates = wide_returns.index.tolist()

    test_dates = [d for d in dates if str(d)[:10] >= test_start]
    tc_rate = transaction_cost_bps / 10000.0

    results = {}
    histories = {}

    # 1. Buy & Hold
    bh_pv = [1.0]
    bh_ret = []
    asset_vals = np.ones(n_assets) / n_assets
    for d in test_dates:
        r = wide_returns.loc[d].values
        asset_vals = asset_vals * (1.0 + r)
        new_v = float(asset_vals.sum())
        bh_ret.append((new_v / bh_pv[-1]) - 1.0)
        bh_pv.append(new_v)

    bh_metrics = compute_metrics(np.array(bh_pv), daily_returns=np.array(bh_ret))
    results["Buy & Hold"] = bh_metrics
    histories["Buy & Hold"] = pd.DataFrame({"portfolio_value": bh_pv[1:], "daily_return": bh_ret}, index=test_dates)

    # 2. Equal Weight (Monthly rebalancing, 10 bps)
    ew_pv = [1.0]
    ew_ret = []
    curr_w = np.zeros(n_assets + 1)
    curr_w[-1] = 1.0
    ew_to = []

    for step, d in enumerate(test_dates):
        r = wide_returns.loc[d].values
        if step % 21 == 0:
            target_w = np.zeros(n_assets + 1)
            target_w[:n_assets] = 1.0 / n_assets
        else:
            target_w = curr_w

        to = float(np.sum(np.abs(target_w - curr_w)))
        cost = tc_rate * to
        ew_to.append(0.5 * to)

        port_r = float(np.dot(target_w[:n_assets], r))
        net_r = port_r - cost
        new_v = ew_pv[-1] * (1.0 + net_r)
        ew_ret.append(net_r)
        ew_pv.append(new_v)

        # Weight drift
        drift = target_w[:n_assets] * (1.0 + r)
        tot = drift.sum() + target_w[-1]
        curr_w[:n_assets] = drift / tot
        curr_w[-1] = target_w[-1] / tot

    ew_metrics = compute_metrics(np.array(ew_pv), daily_returns=np.array(ew_ret), turnover=ew_to)
    results["Equal Weight"] = ew_metrics
    histories["Equal Weight"] = pd.DataFrame({"portfolio_value": ew_pv[1:], "daily_return": ew_ret, "turnover": ew_to}, index=test_dates)

    # 3. Momentum (12-1 month lookback, top 5 stocks, 80/20 cash, monthly rebalance, 10 bps)
    mom_pv = [1.0]
    mom_ret = []
    curr_w = np.zeros(n_assets + 1)
    curr_w[-1] = 1.0
    mom_to = []

    for step, d in enumerate(test_dates):
        curr_idx = dates.index(d)
        r = wide_returns.loc[d].values

        if step % 21 == 0:
            # Lookback 252 days, skip 21 days
            start = curr_idx - 252
            end = curr_idx - 21
            hist_r = wide_returns.iloc[start:end].values
            cum_r = np.prod(1.0 + hist_r, axis=0) - 1.0
            top5_idx = np.argsort(cum_r)[-5:]

            target_w = np.zeros(n_assets + 1)
            target_w[top5_idx] = 0.80 / 5.0
            target_w[-1] = 0.20
        else:
            target_w = curr_w

        to = float(np.sum(np.abs(target_w - curr_w)))
        cost = tc_rate * to
        mom_to.append(0.5 * to)

        port_r = float(np.dot(target_w[:n_assets], r))
        net_r = port_r - cost
        new_v = mom_pv[-1] * (1.0 + net_r)
        mom_ret.append(net_r)
        mom_pv.append(new_v)

        drift = target_w[:n_assets] * (1.0 + r)
        tot = drift.sum() + target_w[-1]
        curr_w[:n_assets] = drift / tot
        curr_w[-1] = target_w[-1] / tot

    mom_metrics = compute_metrics(np.array(mom_pv), daily_returns=np.array(mom_ret), turnover=mom_to)
    results["Momentum"] = mom_metrics
    histories["Momentum"] = pd.DataFrame({"portfolio_value": mom_pv[1:], "daily_return": mom_ret, "turnover": mom_to}, index=test_dates)

    # 4. Mean-Variance (60-day lookback, risk_aversion=1.0, monthly rebalance, 10 bps)
    mv_pv = [1.0]
    mv_ret = []
    curr_w = np.zeros(n_assets + 1)
    curr_w[-1] = 1.0
    mv_to = []

    for step, d in enumerate(test_dates):
        curr_idx = dates.index(d)
        r = wide_returns.loc[d].values

        if step % 21 == 0:
            hist_r = wide_returns.iloc[curr_idx - 60: curr_idx].values
            mu = np.mean(hist_r, axis=0)
            cov = np.cov(hist_r, rowvar=False)

            # Optimization: maximize mu^T w - lambda * w^T cov w, s.t. w >= 0, sum(w) = 1
            w = np.ones(n_assets) / n_assets
            lr = 0.01
            for _ in range(500):
                grad = mu - 2.0 * 1.0 * cov @ w
                w = np.clip(w + lr * grad, 0.0, None)
                s = w.sum()
                w = w / s if s > 0 else np.ones(n_assets) / n_assets

            target_w = np.zeros(n_assets + 1)
            target_w[:n_assets] = w
            target_w[-1] = 0.0
        else:
            target_w = curr_w

        to = float(np.sum(np.abs(target_w - curr_w)))
        cost = tc_rate * to
        mv_to.append(0.5 * to)

        port_r = float(np.dot(target_w[:n_assets], r))
        net_r = port_r - cost
        new_v = mv_pv[-1] * (1.0 + net_r)
        mv_ret.append(net_r)
        mv_pv.append(new_v)

        drift = target_w[:n_assets] * (1.0 + r)
        tot = drift.sum() + target_w[-1]
        curr_w[:n_assets] = drift / tot
        curr_w[-1] = target_w[-1] / tot

    mv_metrics = compute_metrics(np.array(mv_pv), daily_returns=np.array(mv_ret), turnover=mv_to)
    results["Mean-Variance"] = mv_metrics
    histories["Mean-Variance"] = pd.DataFrame({"portfolio_value": mv_pv[1:], "daily_return": mv_ret, "turnover": mv_to}, index=test_dates)

    return results, histories


def print_baseline_table(results):
    print("=" * 88)
    print(f"| {'Strategy':<14} | {'Final Value':<11} | {'Cum Ret':<8} | {'Ann Ret':<8} | {'Sharpe':<7} | {'Vol':<7} | {'Max DD':<8} | {'Turnover':<8} |")
    print("|" + "-" * 16 + "|" + "-" * 13 + "|" + "-" * 10 + "|" + "-" * 10 + "|" + "-" * 9 + "|" + "-" * 9 + "|" + "-" * 10 + "|" + "-" * 10 + "|")
    for name, m in results.items():
        print(f"| {name:<14} | {m['Final Portfolio Value']:>11.4f} | {m['Cumulative Return']:>7.2%} | {m['Annualized Return']:>7.2%} | {m['Sharpe Ratio']:>7.4f} | {m['Volatility']:>7.4f} | {m['Max Drawdown']:>8.2%} | {m['Average Turnover']:>8.4f} |")
    print("=" * 88)


if __name__ == "__main__":
    res, _ = evaluate_baselines()
    print_baseline_table(res)
