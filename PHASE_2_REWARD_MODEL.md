# Finora Phase 2: Person B — Reward & Transaction Cost Model Specification

> **Deliverable Package**: Phase 2 Checkpoint (Reward & Transaction Cost Model Specification)  
> **Author**: Person B (Portfolio Environment & Reward Design)  
> **Target Execution**: Phase 4 (GNN+RL Integration & Agent Training)  

---

## 🎯 Executive Summary

In Phase 2, Person B completed the formal mathematical design, empirical validation, and handoff specification for the **Reward Function** and **Transaction Cost Model** to be used in Finora's RL portfolio allocation environment (`PortfolioEnv`).

The model balances three critical quantitative objectives:
1. **Capital Growth**: Maximizing net log portfolio returns after transaction costs ($r_{\text{net, log}}$).
2. **Risk Mitigation**: Penalizing 20-day annualized rolling portfolio volatility ($\sigma_{20}^{\text{ann}}$).
3. **Diversification & Churn Drag**: Penalizing portfolio turnover drag ($c_t$) and asset concentration ($\text{HHI}$).

---

## 📋 Summary of Deliverables Created

1. **Validation Script (`scripts/validate_reward_model.py`)**:
   - Runs 4 synthetic strategies over 2,012 daily steps on `data/clean/train.parquet`.
   - Verified 0 NaNs / 0 Infs and confirmed penalty mechanisms operate as expected.
2. **Detailed Handoff Specification (`src/environment/reward_model.md`)**:
   - Complete mathematical equations, hyperparameter tuning ranges, and reference code.
3. **Root Handoff Spec (`PHASE_2_REWARD_MODEL.md`)**:
   - Project-level reference for Person A (GNN lead) and Person C (Baselines lead).

---

## 📐 Mathematical Model & Reference Function

$$R_t = \ln(1 + r_{\text{port}, t} - c_t) - \lambda_{\text{vol}} \cdot \sigma_{20, t}^{\text{ann}} - \lambda_{\text{conc}} \cdot \sum_{i=1}^{N} w_{i, t}^2$$

Where:
- $r_{\text{port}, t} = \mathbf{w}_{t, 1:N}^T \mathbf{r}_t$
- $c_t = 0.0010 \cdot \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$ (10 bps linear cost)
- $\sigma_{20, t}^{\text{ann}} = \sqrt{252} \cdot \text{std}(\mathbf{w}_{t, 1:N}^T \mathbf{R}_{\text{win}})$
- $\lambda_{\text{vol}} = 0.10$, $\lambda_{\text{conc}} = 0.01$, $\text{bps} = 10.0$

### Reference Implementation
```python
def compute_phase2_reward(info: dict, returns_window: np.ndarray, weights: np.ndarray, prev_weights: np.ndarray) -> float:
    port_return = info["raw_return"]
    transaction_cost = info["transaction_cost"]
    net_return = port_return - transaction_cost
    log_net_return = float(np.log1p(net_return)) if net_return > -1.0 else -10.0

    asset_weights = weights[:-1]
    if returns_window.size > 0 and np.sum(asset_weights) > 0:
        port_win_returns = asset_weights @ returns_window
        daily_vol = float(np.std(port_win_returns, ddof=1)) if len(port_win_returns) > 1 else 0.0
        ann_vol = daily_vol * np.sqrt(252)
    else:
        ann_vol = 0.0

    vol_penalty = 0.10 * ann_vol
    hhi = float(np.sum(asset_weights ** 2))
    conc_penalty = 0.01 * hhi

    return log_net_return - vol_penalty - conc_penalty
```

---

## 📊 Empirical Validation Results

Tested over 2,012 trading steps (2015-2022) on `data/clean/train.parquet`:

| Strategy | Final Value | Mean Reward | Mean Ann Volatility | Total Txn Cost |
| :--- | :---: | :---: | :---: | :---: |
| **Buy & Hold (1/N)** | **1.3533x** | **-0.013756** | **13.86%** | **0.0020 (0.2%)** |
| **Equal Weight (Monthly)**| 1.3533x | -0.013756 | 13.86% | 0.0020 (0.2%) |
| **Excessive Churning** | 0.8065x | -0.015239 | 13.81% | **0.4977 (49.8%)** |
| **Single Stock (100% NVDA)**| 1.0794x | -0.033191 | **23.34%** | 0.0020 (0.2%) |

---

## 🤝 Team Handoff Summary

- **Person A (GNN Lead)**: GNN node features take `"returns_window"` of shape `(20, 20)`. Node embeddings $\mathbf{H}_t$ will fuse into observation space in Phase 4.
- **Person C (Baselines Lead)**: All benchmark strategies in `src/baselines/` share the exact 10 bps transaction cost model.
- **Person B (Phase 4)**: Ready to plug `compute_phase2_reward` into `portfolioenv.py` during RL policy training.
