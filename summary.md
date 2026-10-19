# Summary of GNN+RL Experiments

## Goal
Improve the GNN-based reinforcement learning system so that the final portfolio value exceeds the best performing baseline model (Buy & Hold at 17.35x).

## Experiments Conducted

### 1. Original GAT Hybrid Embeddings (5 features)
- **Description**: Used the original GAT Hybrid model to generate 32-dimensional node embeddings from the graph data (5 features: returns daily, volatility 20D, volume normalized, volatility annual, log return).
- **Reward Parameters Varied**:
  - Default: lambda_txn=1.0, lambda_vol=0.1, lambda_conc=0.01
  - Low transaction cost: lambda_txn=0.1, lambda_vol=0.01, lambda_conc=0.01
  - Very low penalties: lambda_txn=0.001, lambda_vol=0.001, lambda_conc=0.001
  - High volatility penalty: lambda_txn=0.1, lambda_vol=0.5, lambda_conc=0.01
- **Results**: 
  - Best final portfolio value observed: ~4.018x (seed 789 with default lambdas)
  - Most runs yielded portfolio values in the 3.9x - 4.0x range.
  - The reward function was often negative (around -36 per episode), indicating that the agent struggled to achieve positive risk-adjusted returns.

### 2. Sector-Enhanced Embeddings (9 features)
- **Description**: Modified the `TemporalGraphDataset` to include sector-relative features:
  - Sector-relative returns (stock return minus sector average return)
  - Sector-relative volatility (set to zero in implementation)
  - Sector ranking (within-sector return ranking, normalized to [0,1])
  - Normalized sector ID (sector index divided by max sector ID)
  - Combined with original 5 features to create 9-dimensional node features.
- **Reward Parameters Varied**: Same as above.
- **Results**:
  - Final portfolio values were generally lower than with original embeddings, typically in the 2.9x - 3.1x range.
  - Higher turnover was observed (average turnover ~0.16-0.17 vs. ~0.06 for original embeddings), suggesting the agent traded more frequently without improving returns.
  - The sector-enhanced features did not lead to improved predictive power in the proxy task.

### 3. Sector Rank Embeddings (from Proxy Task)
- **Description**: Trained a GNN to predict next-day sector rank (a proxy task) and used the resulting embeddings for RL.
- **Proxy Task Performance**: 
  - The sector rank prediction task showed extremely poor performance (R² = -0.002399, MAE = 0.301116), indicating that the GNN could not learn to predict next-day sector ranks from the graph data.
- **RL Results**:
  - Final portfolio value: 3.014x (seed 777 with very low penalties: lambda_txn=0.001, lambda_vol=0.001, lambda_conc=0.001).

## Proxy Task Evaluation (Next-Day Return Prediction)
- For both the original 5-feature model and the enhanced 9-feature model, the proxy task of predicting next-day returns performed very poorly:
  - R² scores were often negative and large in magnitude (e.g., -110 for earlier attempts, -0.002399 for sector rank prediction in our final evaluation).
  - This suggests that the graph constructs (stock correlation networks) and the node features used do not contain strong predictive signals for next-day individual stock returns or sector ranks.

## RL Environment and Reward Function
- The environment uses a risk-adjusted, transaction-cost-aware reward:
  ```
  R_t = ln(1 + r_port,t - lambda_txn * c_t) - lambda_vol * sigma_20,t^ann - lambda_conc * HHI_t
  ```
- The agent's observations consist of:
  - `gnn_embeddings`: (n_assets, 32) dimensional node embeddings from the GNN.
  - `weights`: (n_assets + 1,) dimensional vector of previous portfolio weights (including cash).
- The action space is continuous logits for each asset plus cash, transformed via softmax to obtain long-only weights that sum to 1.

## Key Findings
1. The GNN embeddings, regardless of feature set, do not appear to capture predictive information for next-day returns (as measured by the proxy task).
2. The RL agent is able to learn a policy that yields a modest positive portfolio value (around 4x) using the original embeddings, which is better than the GCN-based baseline (2.88x) but still far below the simple Buy & Hold baseline (17.35x).
3. Attempts to enhance the features with sector-relative signals did not improve performance and led to higher turnover.
4. The reward function parameters were varied extensively, but no combination yielded performance close to the Buy & Hold baseline.

## Possible Reasons for Limited Performance
- **Signal-to-Noise Ratio**: Predicting individual stock returns is extremely noisy, and the correlation-based graphs may not capture the relevant information for return prediction.
- **Graph Construction**: The graphs are based on correlation of returns over a window, which may not be informative for predicting future returns.
- **Feature Limitations**: The node features are largely historical returns and volatilities, which may not be predictive of future returns in this dataset.
- **Representational Limits**: The GNN may not be able to extrapolate from the graph structure to predict future returns effectively.

## Suggested Future Work
If the goal is to exceed the Buy & Hold baseline, consider:
1. **Alternative Graph Construction**: Use different graph structures (e.g., based on fundamental data, sector membership, or higher-order correlations) that might capture more predictive signals.
2. **Different Prediction Task**: Instead of predicting next-day returns, predict sector returns, market regimes, or other slower-moving signals that might be more predictable.
3. **Hierarchical GNN**: Implement a hierarchical GNN that first learns sector embeddings and then stock embeddings conditioned on sector, as originally proposed in the plan.
4. **Action Space Redesign**: Predict sector allocations first, then stock weights within sectors, to reduce the action space complexity and leverage sector-level signals.
5. **Longer Training**: Train the RL agent for more timesteps (e.g., 200k or 500k steps) to see if performance improves with more experience.
6. **Ensemble Methods**: Combine the GNN predictions with traditional statistical models or use the GNN features as input to a simpler, more interpretable model.

## Files Modified and Created
- `src/gnn/temporal_graph_dataset.py`: Modified to compute sector-relative features.
- `src/gnn/proxy_task_trainer.py`: Updated to handle variable input features from the enhanced dataset.
- `src/gnn/generate_embeddings_for_rl.py`: Used to generate sector-enhanced embeddings for RL.
- Various training and evaluation scripts were run to assess performance.

## Conclusion
After extensive experimentation, we were able to improve upon the GCN-based GNN+PPO baseline (2.88x) to achieve a final portfolio value of approximately 4.018x using the original GAT Hybrid embeddings with default reward parameters. However, we did not reach the target of exceeding the Buy & Hold baseline (17.35x). 

Our experiments with sector-enhanced features (9 features) and sector rank prediction task embeddings did not yield better performance, with final portfolio values typically in the 2.9x-3.1x range. The proxy task evaluations consistently showed poor predictive performance (R² scores near zero or negative), indicating that the graph constructs and node features used do not contain strong predictive signals for next-day returns or sector ranks.

The fundamental limitation appears to be the lack of strong predictive signals in the graph data and features for the proxy tasks we evaluated. Future work should focus on alternative graph constructions (e.g., based on fundamental data or higher-order correlations), different prediction tasks (e.g., sector returns or market regimes), or architectural changes (e.g., hierarchical GNN or action space redesign) that might capture more relevant signals for portfolio allocation.
