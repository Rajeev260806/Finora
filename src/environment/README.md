# Finora — Phase 2: Person B (Reward & Transaction Cost Model) README

## 📌 Overview & Status

**Status**: ✅ **100% COMPLETE**

This module documents the Phase 2 responsibilities of **Person B** for Finora's GNN+RL Portfolio Allocation system. Person B is responsible for **designing, formalizing, and empirically validating the reward function** and **specifying the transaction-cost model** used in the Gymnasium-style `PortfolioEnv` (`src/environment/portfolioenv.py`).

---

## 🎯 Phase 2 Objectives & Achievements Summary

| Objective | Status | Implementation / Deliverable File |
| :--- | :---: | :--- |
| **1. Review `PortfolioEnv` Architecture** | ✅ Complete | Summary of Observation Dict, Action Box, Info fields documented below |
| **2. Draft Risk-Adjusted Reward Function** | ✅ Complete | Multi-component model combining log net return, rolling volatility penalty & HHI concentration penalty |
| **3. Specify Transaction-Cost Model** | ✅ Complete | Linear 10 bps baseline ($\beta = 0.0010$) + Quadratic market impact model |
| **4. Validate with Real Market Data** | ✅ Complete | Built & executed [`scripts/validate_reward_model.py`](file:///e:/Finora/scripts/validate_reward_model.py) over 2,012 trading steps |
| **5. Document Assumptions & Limitations** | ✅ Complete | Formally documented execution, stationarity, and long-only constraints |
| **6. Produce Handoff Package** | ✅ Complete | Published [`PHASE_2_REWARD_MODEL.md`](file:///e:/Finora/PHASE_2_REWARD_MODEL.md) & [`src/environment/reward_model.md`](file:///e:/Finora/src/environment/reward_model.md) |

---

## 🏗️ 1. Gymnasium Environment Contract (`PortfolioEnv`)

The environment (`src/environment/portfolioenv.py`) operates as follows:

```text
               ┌───────────────────────────────────────────────┐
               │                PortfolioEnv                   │
               │                                               │
   Action  ───►│  Logits (N+1,) ──► Softmax ──► Weights (N+1,) │───► Next Observation
               │                                               │     - returns_window (20, 20)
               │  Step: Net Return & Cost Calculation           │     - weights (21,)
               │  Reward: R_t = LogNetRet - VolPen - ConcPen   │
               └───────────────────────────────────────────────┘
                                       │
                                       ▼
                             Step Info Dictionary
                   (date, portfolio_value, turnover, 
                    transaction_cost, raw_return, cash_weight)
```

### Key Spaces & Specifications
- **Observation Space (`spaces.Dict`)**:
  - `"returns_window"`: Array of shape `(N, W)` = `(20, 20)` containing the past $W=20$ days of log returns for all $N=20$ equities. Designed specifically to serve as GNN node feature matrices.
  - `"weights"`: Vector of shape `(N+1,)` = `(21,)` containing the portfolio weights from step $t-1$, with slot `[-1]` reserved for cash.
- **Action Space (`spaces.Box`)**:
  - Continuous action logits $\mathbf{z} \in [-10, 10]^{21}$.
  - Processed internally via Softmax: $\mathbf{w}_t = \text{softmax}(\mathbf{z})$, guaranteeing $\sum w_{i, t} = 1.0$ and $w_{i, t} \ge 0$.
- **Look-Ahead Bias Prevention**: Observation slice at index $t$ ends strictly at $t$ (exclusive of $t+1$), guaranteeing the policy acts only on past realized returns.

---

## 📐 2. Reward Function Specification

To train a deep RL agent (e.g. PPO / DDPG) that balances profit, risk, and rebalancing efficiency, the reward function is defined as:

$$R_t = r_{\text{net, log}, t} - \lambda_{\text{vol}} \cdot \sigma_{20, t}^{\text{ann}} - \lambda_{\text{conc}} \cdot \text{HHI}_t$$

### Component Mathematical Formulations

#### 1. Log Net Return ($r_{\text{net, log}, t}$)
Encourages portfolio value growth while deducting transaction expenses:
$$r_{\text{net}, t} = r_{\text{port}, t} - \lambda_{\text{txn}} \cdot c_t = (\mathbf{w}_{t, 1:N}^T \mathbf{r}_t) - \lambda_{\text{txn}} \cdot c_t$$
$$r_{\text{net, log}, t} = \begin{cases} \ln(1 + r_{\text{net}, t}) & \text{if } r_{\text{net}, t} > -1.0 \\ -10.0 & \text{if } r_{\text{net}, t} \le -1.0 \end{cases}$$

#### 2. Rolling 20-Day Annualized Portfolio Volatility Penalty ($\sigma_{20, t}^{\text{ann}}$)
Computes annualized volatility directly from the observation matrix $\mathbf{R}_{\text{win}} \in \mathbb{R}^{20 \times 20}$:
$$\mathbf{r}_{\text{port, win}} = \mathbf{w}_{t, 1:N}^T \mathbf{R}_{\text{win}} \quad \in \mathbb{R}^{20}$$
$$\sigma_{20, t} = \text{std}(\mathbf{r}_{\text{port, win}}), \quad \sigma_{20, t}^{\text{ann}} = \sigma_{20, t} \cdot \sqrt{252}$$
$$\text{Penalty}_{\text{vol}} = \lambda_{\text{vol}} \cdot \sigma_{20, t}^{\text{ann}}$$

#### 3. Asset Concentration Regularization ($\text{HHI}_t$)
Penalizes non-diversified single-stock allocations using the Herfindahl-Hirschman Index across risky assets:
$$\text{HHI}_t = \sum_{i=1}^{N} w_{i, t}^2$$
$$\text{Penalty}_{\text{conc}} = \lambda_{\text{conc}} \cdot \text{HHI}_t$$

---

## 💸 3. Transaction Cost Model

- **Baseline Model (Linear Basis Points)**:
  $$\text{Turnover}_t = \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1 = \sum_{i=1}^{N+1} |w_{i, t} - w_{i, t-1}|$$
  $$c_t = \left(\frac{\text{bps}}{10\,000}\right) \cdot \text{Turnover}_t = 0.0010 \cdot \text{Turnover}_t \quad (\text{10 bps} = 0.10\%)$$
- **Extended Model (Quadratic Slippage)**:
  $$c_{t, \text{extended}} = 0.0010 \cdot \text{Turnover}_t + \gamma \cdot \text{Turnover}_t^2 \quad (\gamma = 10^{-4})$$

---

## 📊 4. Empirical Validation Results

Validated over 2,012 trading days on `data/clean/train.parquet` via [`scripts/validate_reward_model.py`](file:///e:/Finora/scripts/validate_reward_model.py).

### Strategy Performance Summary Table

| Strategy | Description | Final Value | Mean Reward | Mean Ann Vol | Total Txn Drag |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Buy & Hold (1/N)** | Initial equal allocation, 0 turnover | **1.3533x** | **-0.013756** | **13.86%** | **0.0020 (0.2%)** |
| **Equal Weight** | Monthly rebalancing to $1/N$ | **1.3533x** | **-0.013756** | **13.86%** | **0.0020 (0.2%)** |
| **Excessive Churning** | Random logit updates every day | 0.8065x | -0.015239 | 13.81% | **0.4977 (49.8%)** |
| **Single Stock (NVDA)**| 100% allocation to single asset | 1.0794x | -0.033191 | **23.34%** | 0.0020 (0.2%) |

### Verified Sanity Checks
- ✅ **Check 1**: **0 NaNs / 0 Infs** across 2,012 steps.
- ✅ **Check 2**: **Excessive Churning penalized**: High turnover cost drag ($49.8\%$) reduces reward and portfolio value significantly.
- ✅ **Check 3**: **Single Stock Concentration penalized**: Heavy concentration penalty ($\text{HHI} = 1.000$) and higher volatility reduce reward to $-0.033191$.
- ✅ **Check 4**: **Numerical Boundedness**: Total step rewards remain well-behaved in $[-0.10, +0.10]$.

---

## 🛠️ 5. Hyperparameter Reference & Tuning Table

| Hyperparameter | Code Name | Default | Phase 4 Tuning Range | Purpose |
| :--- | :--- | :---: | :---: | :--- |
| **Volatility Penalty** | `lambda_vol` | `0.10` | `[0.01, 0.50]` | Controls penalty strength for portfolio volatility |
| **Transaction Multiplier**| `lambda_txn` | `1.00` | `[0.50, 2.00]` | Multiplier for transaction cost deduction |
| **Concentration Weight** | `lambda_conc` | `0.01` | `[0.001, 0.05]`| Multiplier for asset concentration HHI penalty |
| **Transaction Bps** | `tc_bps` | `10.0` | `[5.0, 30.0]` | Transaction cost rate in basis points (10 bps = 0.10%) |

---

## 💻 6. Reference Python Implementation

For Phase 4 implementation into `PortfolioEnv._calculate_reward()`:

```python
import numpy as np

def compute_phase2_reward(
    info: dict,
    returns_window: np.ndarray,
    weights: np.ndarray,
    prev_weights: np.ndarray,
    lambda_vol: float = 0.10,
    lambda_txn: float = 1.0,
    lambda_conc: float = 0.01,
) -> float:
    """Computes Phase 2 risk-adjusted, cost-aware modular reward."""
    port_return = info["raw_return"]
    transaction_cost = info["transaction_cost"]
    net_return = port_return - (lambda_txn * transaction_cost)

    log_net_return = float(np.log1p(net_return)) if net_return > -1.0 else -10.0

    asset_weights = weights[:-1]
    if returns_window.size > 0 and np.sum(asset_weights) > 0:
        port_win_returns = asset_weights @ returns_window
        daily_vol = float(np.std(port_win_returns, ddof=1)) if len(port_win_returns) > 1 else 0.0
        ann_vol = daily_vol * np.sqrt(252)
    else:
        ann_vol = 0.0

    vol_penalty = lambda_vol * ann_vol
    hhi = float(np.sum(asset_weights ** 2))
    conc_penalty = lambda_conc * hhi

    return log_net_return - vol_penalty - conc_penalty
```

---

## 🤝 7. Team Handoff & Next Steps (Phase 4)

- **Person A (GNN Lead)**: GNN node encoder will take `"returns_window"` of shape `(20, 20)` alongside correlation graph adjacency matrices from `graphs/saved_graphs/`.
- **Person C (Baselines Lead)**: Baselines suite in `src/baselines/` uses the identical $10\text{ bps}$ cost model.
- **Person B (Phase 4 RL Training)**:
  1. Substitute `compute_phase2_reward` into `PortfolioEnv`.
  2. Integrate PyTorch Geometric GNN node embeddings into observation space.
  3. Train PPO agent using Stable-Baselines3 / PyTorch RL.
  4. Compare out-of-sample Sharpe ratio & drawdown against Person C's baselines on `data/clean/test.parquet`.
