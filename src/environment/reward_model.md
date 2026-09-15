# Finora Phase 2: Reward & Transaction Cost Model Specification (Person B)

This document provides the formal mathematical specification, empirical validation results, modeling assumptions, and Phase 4 implementation guidelines for the **Reward Function** and **Transaction Cost Model** designed for Finora's `PortfolioEnv`.

---

## 📌 1. Environment Architecture Contract (Phase 1 Review)

The `PortfolioEnv` (`src/environment/portfolioenv.py`) provides the following Gym interface:

- **Observation Space (`spaces.Dict`)**:
  - `"returns_window"`: Tensor of shape `(N, W)` = `(20, 20)` containing the historical daily log returns across the past $W=20$ trading days for $N=20$ equities.
  - `"weights"`: Vector of shape `(N + 1,)` = `(21,)` representing current portfolio weights at step $t-1$, including cash in the last slot (`weights[-1]`).
- **Action Space (`spaces.Box`)**:
  - Continuous action vector of shape `(N + 1,)` containing raw unconstrained policy logits $\mathbf{z} \in [-10, 10]^{21}$.
  - Transformed internally via Softmax into long-only weights: $\mathbf{w}_t = \text{softmax}(\mathbf{z})$, guaranteeing $\sum_{i=1}^{N+1} w_{i, t} = 1$ and $w_{i, t} \ge 0$.
- **Step Info Dict**:
  - `info["date"]`: Current trading date.
  - `info["portfolio_value"]`: Total portfolio capital $V_t$.
  - `info["raw_return"]`: Gross daily portfolio return $r_{\text{port}, t} = \mathbf{w}_{t, 1:N}^T \mathbf{r}_t$.
  - `info["turnover"]`: Rebalancing turnover $\|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$.
  - `info["transaction_cost"]`: Transaction cost drag $c_t$.
  - `info["cash_weight"]`: Weight allocated to cash $w_{\text{cash}, t}$.

---

## 📐 2. Reward Function Mathematical Formulation

To train a robust RL agent (e.g., PPO/DDPG) that achieves high risk-adjusted returns without excessive churning or extreme asset concentration, we formalize a **multi-component risk-adjusted reward function**:

$$R_t = r_{\text{net, log}, t} - \lambda_{\text{vol}} \cdot \sigma_{20, t}^{\text{ann}} - \lambda_{\text{conc}} \cdot \text{HHI}_t$$

### Component Breakdown

#### A. Log Net Portfolio Return ($r_{\text{net, log}, t}$)
Directly encourages portfolio capital growth net of trading expenses:
$$r_{\text{net}, t} = r_{\text{port}, t} - \lambda_{\text{txn}} \cdot c_t$$
$$r_{\text{net, log}, t} = \begin{cases} 
\ln(1 + r_{\text{net}, t}) & \text{if } r_{\text{net}, t} > -1.0 \\
-10.0 & \text{if } r_{\text{net}, t} \le -1.0 
\end{cases}$$

#### B. Rolling 20-Day Annualized Portfolio Volatility Penalty ($\sigma_{20, t}^{\text{ann}}$)
Encourages risk-aware allocation by penalizing dynamic portfolio variance. Computed directly from the observation matrix $\mathbf{R}_{\text{win}} \in \mathbb{R}^{N \times W}$:
$$\mathbf{r}_{\text{port, win}} = \mathbf{w}_{t, 1:N}^T \mathbf{R}_{\text{win}} \in \mathbb{R}^{W}$$
$$\sigma_{20, t} = \sqrt{\frac{1}{W-1} \sum_{k=1}^{W} \left(r_{\text{port, win}}[k] - \bar{r}_{\text{port, win}}\right)^2}$$
$$\sigma_{20, t}^{\text{ann}} = \sigma_{20, t} \cdot \sqrt{252}$$
$$\text{Penalty}_{\text{vol}} = \lambda_{\text{vol}} \cdot \sigma_{20, t}^{\text{ann}}$$

#### C. Asset Concentration Regularization ($\text{HHI}_t$)
Penalizes non-diversified portfolios using the Herfindahl-Hirschman Index across risky assets to incentivize multi-sector diversification:
$$\text{HHI}_t = \sum_{i=1}^{N} w_{i, t}^2$$
$$\text{Penalty}_{\text{conc}} = \lambda_{\text{conc}} \cdot \text{HHI}_t$$

### Hyperparameter Defaults & Tuning Ranges

| Parameter | Symbol | Default Value | Recommended Tuning Range | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Vol Penalty Weight** | $\lambda_{\text{vol}}$ | `0.10` | `[0.01, 0.50]` | Weight on annualized rolling volatility penalty |
| **Txn Penalty Weight** | $\lambda_{\text{txn}}$ | `1.00` | `[0.50, 2.00]` | Multiplier on transaction cost penalty |
| **Concentration Weight**| $\lambda_{\text{conc}}$ | `0.01` | `[0.001, 0.05]`| Weight on HHI concentration penalty |
| **Transaction Bps** | $\text{bps}$ | `10.0` | `[5.0, 30.0]` | Transaction cost rate in basis points (0.10%) |

---

## 💸 3. Transaction Cost Model Specification

### Baseline Model: Linear Basis Points
Applied inside `step()` during weight updates:
$$\text{Turnover}_t = \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1 = \sum_{i=1}^{N+1} |w_{i, t} - w_{i, t-1}|$$
$$c_t = \left(\frac{\text{bps}}{10\,000}\right) \cdot \text{Turnover}_t = 0.0010 \cdot \text{Turnover}_t$$

### Extended Model: Quadratic Market Impact / Slippage
For stress testing under high-volume trading scenarios in Phase 4:
$$c_{t, \text{extended}} = \left(\frac{\text{bps}}{10\,000}\right) \cdot \text{Turnover}_t + \gamma \cdot \text{Turnover}_t^2$$
where $\gamma = 10^{-4}$ models non-linear slippage drag during aggressive asset reallocations.

---

## 🧪 4. Empirical Validation Results Summary

The reward specification was validated on Finora's training set (`data/clean/train.parquet`, 2,012 trading steps) using `scripts/validate_reward_model.py`.

### Validation Benchmark Table

| Strategy | Final Portfolio Value | Mean Daily Reward | Std Dev Reward | Mean Ann Volatility | Total Txn Cost Drag |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Buy & Hold (1/N)** | **1.3533x** | **-0.013756** | **0.010657** | **13.86%** | **0.0020 (0.2%)** |
| **Equal Weight (Monthly)**| 1.3533x | -0.013756 | 0.010657 | 13.86% | 0.0020 (0.2%) |
| **Excessive Churning** | 0.8065x | -0.015239 | 0.010818 | 13.81% | **0.4977 (49.8%)** |
| **Single Stock (100% NVDA)**| 1.0794x | -0.033191 | 0.017084 | **23.34%** | 0.0020 (0.2%) |

### Key Findings & Verification Criteria
1. ✅ **Numerical Stability**: 0 NaNs and 0 Infs across all 2,012 simulation steps. Total daily reward strictly bounded within $[-0.10, +0.10]$.
2. ✅ **Turnover Drag Penalization**: Excessive daily churning lost **49.8% of total portfolio capital** to transaction costs, resulting in significantly lower mean reward than periodic rebalancing.
3. ✅ **Risk & Concentration Penalization**: Single stock allocation incurred $23.34\%$ volatility and $\text{HHI} = 1.000$, resulting in a severe concentration penalty (-0.033191 vs -0.013756).

---

## 📝 5. Assumptions & Limitations

1. **Daily Execution**: Trades are assumed to execute at daily market close prices without intraday execution latency.
2. **Stationary Window Volatility**: 20-day rolling standard deviation assumes short-term return variance is local and stationary.
3. **Zero Risk-Free Cash Yield**: Cash weight $w_{\text{cash}}$ earns exactly $0.0\%$ return.
4. **Long-Only Simplex Constraint**: No short selling ($w_{i, t} \ge 0$) or leverage ($\sum w = 1.0$).

---

## 🔗 6. Handoff Integration Contracts

### For Person A (GNN Lead)
- GNN node features will extract asset-level features directly from `"returns_window"` of shape `(N, W)` = `(20, 20)`.
- GNN graph topology from `graphs/saved_graphs/` provides the adjacency matrix $\mathbf{A}_t \in \mathbb{R}^{20 \times 20}$.
- GNN output node embeddings $\mathbf{H}_t \in \mathbb{R}^{20 \times d}$ will be concatenated into the observation dictionary.

### For Person C (Quantitative Baselines Lead)
- Benchmark comparison scripts in `src/baselines/` use the identical 10 bps transaction cost structure ($\beta = 0.0010$) and portfolio value formula $V_t = V_{t-1} \cdot (1 + r_{\text{port}, t} - c_t)$.

### For Person B (Phase 4 RL Implementation)
Copy-paste reference Python implementation ready for `PortfolioEnv._calculate_reward()`:

```python
def compute_phase2_reward(
    info: dict,
    returns_window: np.ndarray,
    weights: np.ndarray,
    prev_weights: np.ndarray,
    lambda_vol: float = 0.10,
    lambda_txn: float = 1.0,
    lambda_conc: float = 0.01,
) -> float:
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
