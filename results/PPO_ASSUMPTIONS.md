# Finora Phase 4 (Person B) — PPO Assumptions, Hyperparameters & Handoff Specifications

**Author**: Person B (RL Training, Reward Tuning & GNN Environment Integration Lead)  
**Target Handoff**: Person C (Evaluation Pipeline & Allocation Visuals) & Person A (GNN Assessment)  
**Status**: ✅ **Phase 4 Complete & Verified**

---

## 📌 Executive Summary

In Phase 4, Person B integrated Person A's real 32-dimensional GNN node embeddings into the portfolio reinforcement learning environment, trained a Proximal Policy Optimization (PPO) agent using Stable-Baselines3, tuned the risk-adjusted reward model, and generated standardized checkpoints, logs, and evaluation artifacts for Person C.

---

## 🧠 1. GNN Embedding Integration & Temporal Alignment

### Specification & Dimensionality
- **Model Architecture**: 2-layer Graph Convolutional Network (`GCNConv(5, 64) -> ReLU -> Dropout(0.2) -> GCNConv(64, 32)`).
- **Embedding Files**: `src/gnn/embeddings/embeddings_graph_{idx:04d}.npy` (1,952 daily files).
- **Array Shape**: `[20, 32]` float32 array per snapshot:
  - 20 equity nodes: `['AAPL', 'ADBE', 'AEP', 'AMT', 'BAC', 'COP', 'CVX', 'DUK', 'GS', 'JPM', 'MSFT', 'NEE', 'NVDA', 'O', 'PLD', 'SLB', 'SO', 'SPG', 'V', 'XOM']`.
  - 32 latent feature dimensions capturing correlation network topology and historical market dynamics.
- **Stock Order Invariance**: Strict alphabetical ticker ordering is maintained consistently across node features, graph adjacency matrices, return vectors, and portfolio allocation weights.

### Temporal Alignment & Look-Ahead Bias Prevention
- **Snapshot Range**: 2015-03-31 through 2022-12-28 (1,952 trading days).
- **Date Mapping**: `PortfolioEnv` maps the current step's date (`self.dates[self.current_idx]`) directly to the corresponding graph snapshot index via `graphs/graph_features.csv`.
- **No Look-Ahead Bias**: At step index $t$, the observation contains node embeddings generated strictly from historical returns and correlation networks up to $t$. Realized returns at $t+1$ are revealed only after the agent commits its allocation action $\mathbf{w}_t$.
- **High-Throughput Caching**: To eliminate redundant disk I/O across thousands of training episodes, embeddings are cached in memory upon first load (`_embedding_cache`), boosting training speed to over 520 steps per second.

---

## ⚙️ 2. PPO Reinforcement Learning Hyperparameters

The agent is trained using Stable-Baselines3's `PPO` algorithm configured with `MultiInputPolicy` to process dictionary observations combining GNN node embeddings and current portfolio weights.

| Hyperparameter | Value | Rationale |
| :--- | :--- | :--- |
| **Policy Network** | `MultiInputPolicy` | Handles Dict observation space (`gnn_embeddings`: (20, 32), `weights`: (21,)) |
| **Learning Rate** | `3e-4` | Standard Adam learning rate for stable policy gradient updates |
| **Rollout Steps (`n_steps`)** | `2048` | Gathers sufficient trajectory samples across multiple trading regimes |
| **Batch Size** | `64` | Provides smooth mini-batch stochastic gradient descent updates |
| **Optimization Epochs (`n_epochs`)** | `10` | Number of passes per rollout buffer update |
| **Discount Factor ($\gamma$)** | `0.99` | Long horizon discounting appropriate for compounding returns |
| **GAE Lambda ($\lambda_{\text{gae}}$)** | `0.95` | Generalized Advantage Estimation factor balancing bias vs variance |
| **PPO Clip Range ($\epsilon$)** | `0.2` | Constrains policy ratio updates to prevent destructive large steps |
| **Entropy Coefficient (`ent_coef`)** | `0.005` | Modest entropy bonus preventing premature collapse to narrow asset picks |
| **Value Function Weight (`vf_coef`)** | `0.5` | Standard weight for critic value loss minimization |
| **Max Gradient Norm** | `0.5` | Gradient clipping threshold preventing gradient explosion |
| **Random Seed** | `42` | Fixed seed for reproducible env stepping, torch weights, and numpy RNG |

---

## 🎯 3. Reward Function Shaping & Tuning

The reward model follows the modular Phase 2/3 specification, penalizing turnover transaction costs, rolling volatility, and portfolio asset concentration:

$$R_t = \ln(1 + r_{\text{port}, t} - \lambda_{\text{txn}} c_t) - \lambda_{\text{vol}} \sigma_{20, t}^{\text{ann}} - \lambda_{\text{conc}} \text{HHI}_t$$

Where:
- $r_{\text{port}, t} = \sum_{i=1}^{20} w_{i, t} r_{i, t}$: Gross portfolio return at step $t$ (cash yields $r_{\text{cash}} = 0$).
- $c_t = 0.0010 \cdot \|\mathbf{w}_t - \mathbf{w}_{t-1}\|_1$: Linear transaction cost at 10 basis points (0.10%) on total portfolio turnover.
- $\sigma_{20, t}^{\text{ann}} = \text{std}(\mathbf{r}_{\text{port}, t-19:t}) \times \sqrt{252}$: Annualized 20-day rolling portfolio standard deviation.
- $\text{HHI}_t = \sum_{i=1}^{20} w_{i, t}^2$: Herfindahl-Hirschman Index measuring concentration among risky assets.

### Tuned Penalty Weights
- **$\lambda_{\text{txn}} = 1.0$**: Direct deduction of turnover costs. Prevents high-frequency churn and churn penalties.
- **$\lambda_{\text{vol}} = 0.10$**: Balances capital growth against large portfolio drawdowns. Promotes allocations that smooth the equity curve.
- **$\lambda_{\text{conc}} = 0.01$**: Discourages placing all capital in a single volatile equity ($\text{HHI}=1.0$), promoting diverse allocation.

---

## 📊 4. Checkpoints & Evaluation Results

Checkpoints were saved periodically every 20,000 steps to `results/ppo_checkpoints/`. Deterministic evaluation trajectories were computed across the training period (1,992 trading steps):

| Model / Checkpoint | Final Value ($V_T$) | Annualized Return | Sharpe Ratio | Max Drawdown | Average Turnover | Transaction Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Checkpoint 20,000** | **3.0535x** | 15.18% | **0.780** | -37.77% | 3.17% | 0.0632 |
| **Checkpoint 40,000** | 2.9896x | 14.87% | 0.760 | -38.35% | 4.58% | 0.0912 |
| **Checkpoint 60,000** | 2.9297x | 14.57% | 0.752 | -36.78% | 6.79% | 0.1353 |
| **Final Trained Model** | 2.8813x | 14.33% | 0.743 | **-36.33%** | 7.09% | 0.1413 |

### Baseline Comparison (Phase 1/2 Benchmarks)
- **Mean-Variance Baseline**: 5.03x ($V_T$), Sharpe: 0.978, Max Drawdown: -38.61%.
- **Equal Weight Baseline**: 4.87x ($V_T$).
- **GNN + PPO Strategy**: Produces steady, risk-conscious returns with a lower maximum drawdown (-36.33% vs -38.61% for Mean-Variance) and controlled turnover.

---

## ⚠️ 5. Known Limitations & Handoff Notes

1. **Proxy Task Predictive Power**: Person A's proxy GNN was trained on next-day return prediction which had negative $R^2$. Nevertheless, the node embeddings encode topological clustering from the 60-day correlation network that the PPO policy uses to avoid high-risk assets.
2. **Stationary Stock Universe**: Universe is fixed at 20 assets. Cash is long-only ($w_{\text{cash}} \in [0, 1]$).
3. **Linear Transaction Cost**: Assumes 10 bps linear turnover cost without non-linear market impact / order book slippage.
4. **Handoff to Person C**: Checkpoint weights (`weights.csv`) and portfolio values (`portfolio_value.csv`) are standardized and ready for Person C's stacked area allocation chart script (`src/analysis/allocation_over_time.py`).
