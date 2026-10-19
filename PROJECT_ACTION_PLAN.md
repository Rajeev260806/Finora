# GNN + PPO Portfolio Project: Action Plan

Confidence tags: **[Certain]** = visible in your files or simple arithmetic. **[Likely]** = strong inference, verify. **[Guessing]** = untested idea.

---

## 0. Uncomfortable truth first

**[Certain]** Right now you cannot claim a win over any baseline, even if a number looks higher. Your test results are in-sample (leakage), your metrics contain bugs, and your best result (4.018x) is not measured on the same period as the baselines. Fix evaluation before touching the model.

---

## Phase 1: Fix the evaluation (do this first, ~1-2 days)

### 1.1 Remove train/test leakage
- **[Certain]** `ASSUMPTIONS_TEST.md` says GNN+PPO was trained on 2015-01-02 to 2024-12-31. The test period (2022-12-29 to 2024-12-31) is inside it.
- Use a strict split:
  - Train: 2015-01-02 to 2021-12-31
  - Validation: 2022-01-01 to 2022-12-28 (tune here only)
  - Test: 2022-12-29 to 2024-12-31 (touch once at the end)
- Regenerate GNN embeddings using **train data only** (graph construction, normalization, and GNN training must not see val/test).
- Fit all scalers/normalizers on train only.

### 1.2 Make all results comparable
- **[Certain]** 4.018x at 14.82% annualized implies about 10 years (ln 4.018 / ln 1.1482 ≈ 10.07). It is a full-period number. The baselines in the test table are 2-year numbers. They cannot be compared.
- **[Likely]** The test script loads `src/gnn/embeddings_original/` (probably GCN), not `embeddings_original_gat`. Check which embeddings `run_test_comparison.py` actually loads.
- Re-run every strategy (baselines + GCN + GAT Hybrid) on the **same test period, same costs (10 bps), same initial value**.

### 1.3 Fix metric bugs
- **[Certain]** Momentum: final value 1.2154 over 2 years is about 10% annualized, but the file reports 23.83%.
- **[Certain]** Buy & Hold: 17.35x with 64.12% annualized implies about 5.8 years, not 10.
- Write one shared `compute_metrics(returns)` function used by every strategy. Add a unit test: for a constant daily return r, annualized return must equal (1+r)^252 - 1.
- Check: Sharpe = mean/std * sqrt(252), max drawdown from the cumulative curve, turnover = 0.5 * sum(|w_t - w_{t-1}|) (state which definition you use).

### 1.4 Fix the proxy task
- **[Likely]** R² of -110 to -130 is a bug, not "no signal." No signal gives R² near 0 (your sector-rank task at -0.0024 shows this).
- Likely causes: unnormalized targets, train/eval scaling mismatch, wrong shape/broadcasting, predicting raw returns with a large output scale.
- Fix: standardize targets (z-score per day, cross-sectionally), then report **Spearman rank IC** per day (mean IC and IC/std). That is the metric that matters for allocation.
- Until this is fixed, remove the claim "the graph has no predictive signal" from your write-up. **[Certain]** It is unsupported.

---

## Phase 2: Set realistic targets

| Target | Verdict |
|---|---|
| Buy & Hold (1.698x, Sharpe 1.77) | **[Certain]** Do not target. NVDA in the basket over 2023-24 makes this nearly unbeatable for a rebalancing strategy. |
| Mean-Variance (Sharpe 1.67) | **[Likely]** Hard. Stretch goal. |
| Equal Weight (1.3246x, Sharpe 1.31) | **[Likely]** Realistic primary target. You are about 3% behind with similar turnover. |
| Momentum (1.2154x) | **[Certain]** Nominally beaten, but its row is buggy and Sharpe is a tie (1.2319 vs 1.2363). Do not claim until Phase 1 is done. |

Primary goal: beat **Equal Weight** on Sharpe and/or max drawdown, out-of-sample, across multiple seeds.

---

## Phase 3: Improve the model

### 3.1 Residual policy around Equal Weight (highest priority)
- Replace free-form log-weight actions with:
  `w = softmax( log(1/N) + alpha * tanh(a) )`, alpha around 0.1 to 0.5 (tune on validation).
- **[Likely]** The agent starts at Equal Weight and only deviates when it learned something, so the floor is about Equal Weight performance instead of relearning 1/N from scratch.

### 3.2 Training budget and reward
- **[Likely]** 60,000 timesteps is only about 30 episodes of ~1,800 days. Try 300k to 1M.
- **[Likely]** `lambda_txn = 1.0` is huge relative to a 10 bps cost. Make the penalty consistent with real costs (reward = portfolio return - actual cost).
- **[Guessing]** Try a differential Sharpe ratio reward or a risk-adjusted reward.
- Normalize observations (running mean/std), try a smaller learning rate, and log entropy and value loss to check the agent is learning.

### 3.3 Sanity ablations (cheap and informative)
- PPO with **no embeddings** (returns only) vs GCN embeddings vs GAT embeddings vs **random embeddings**.
- **[Certain]** If GAT does not beat random embeddings, the embeddings are not contributing, and you should say so.

### 3.4 Only after 3.1 to 3.3: bigger ideas
- Hierarchical action space (sector weights first, then stocks within sectors).
- Different graph construction (rolling correlation, sector membership graph).
- Different proxy targets (volatility, sector returns, regime).
- Hybrid: GNN for risk estimation + simple model for returns.

---

## Phase 4: Statistical rigor

- Run **5 to 10 seeds** per configuration. Report mean ± std for final value, Sharpe, max drawdown, turnover.
- **[Likely]** Your lambda and seed comparisons are noise: variants differ by under 3% (4.018x vs 3.891x) with one seed each, and seed 789 was picked after seeing results.
- Select hyperparameters on **validation only**. Run test once.
- Optional: block bootstrap the daily returns to give a confidence interval on the Sharpe difference vs Equal Weight.
- Add a robustness check: results at 5, 10, and 20 bps costs.
- Add a second test window if data allows (for example train to 2019, test 2020-2021) so the result is not one market regime.

---

## Phase 5: Reporting

- One results table: all strategies, same period, same costs, mean ± std over seeds.
- Include a leakage statement: exactly what data each stage saw.
- Be honest about outcomes:
  - If you beat Equal Weight on Sharpe or drawdown out-of-sample: report it, with the caveat that Buy & Hold still wins on return.
  - If you do not: report a clean negative result (GAT gives 2.03x better sector separation than GCN, evaluated without leakage, and it did/did not help RL). **[Certain]** This is defensible. A leaky win is not.
- Remove or correct claims that depend on the buggy numbers (README "Momentum showed good annualized return", "no predictive signal").
- Keep a single reproducible script: `python run_all.py --seeds 0..9` that regenerates every table and plot.

---

## Checklist

- [x] Split fixed: train 2015-2021, val 2022, test 2023-2024 (`data/clean/{train,val,test}.parquet`)
- [x] Embeddings regenerated on train data only (`src/gnn/build_all_graphs.py`, `src/gnn/train_proxy_models.py`)
- [x] Confirmed which embeddings the test script loads (Confirmed GAT Hybrid was generated into `src/gnn/embeddings_original`)
- [x] Shared metrics function + unit test; Momentum and Buy & Hold numbers recomputed (`src/analysis/metrics.py`, `tests/test_metrics.py`)
- [x] Proxy task targets standardized; Spearman IC reported (`src/gnn/train_proxy_models.py`)
- [x] All strategies re-run on the same period and costs (`src/analysis/run_rigorous_evaluation.py`)
- [x] Residual policy around Equal Weight implemented (`src/environment/portfolioenv.py`, `tests/test_residual_policy.py`)
- [x] Timesteps raised to 300k+; reward penalty aligned with 10 bps cost (`PortfolioEnv` cost penalty matches exact 10 bps)
- [x] Ablation: no embeddings / random embeddings / GCN / GAT (`src/analysis/run_rigorous_evaluation.py`)
- [x] 5 to 10 seeds, mean ± std reported (5 seeds: `[42, 101, 202, 303, 404]`)
- [x] Final test run done once, after all tuning (Holdout test evaluated once: 2022-12-29 to 2024-12-31)
- [x] README and claims updated to match the corrected results (`results/FINAL_EVALUATION_REPORT.md`)

---

## Order of work if time is short

1. Phase 1.1 to 1.3 (split, comparability, metrics). Nothing else is trustworthy without these.
2. Phase 3.1 (residual policy). Biggest expected gain for least effort.
3. Phase 4 (seeds). Required to make any claim.
4. Everything else.
