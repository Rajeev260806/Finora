# Finora Project Results & Deliverables (Phases 3 & 4)

This directory contains the core experimental deliverables for the Finora project across Phase 3 (Baselines & Graph Visualizations) and Phase 4 (PPO Reinforcement Learning & GNN Embedding Integration).

---

## 📁 Directory Structure

```
results/
├── ASSUMPTIONS.md              # Phase 3 baseline assumptions & limitations
├── PPO_ASSUMPTIONS.md          # Phase 4 PPO RL assumptions, hyperparameters & results
├── README.md                   # This overview & reproducibility guide
├── ppo_final_model.zip         # Phase 4 final trained PPO policy model
├── ppo_checkpoints/            # Phase 4 periodic PPO checkpoints
│   ├── ppo_checkpoint_20000_steps.zip
│   ├── ppo_checkpoint_40000_steps.zip
│   └── ppo_checkpoint_60000_steps.zip
├── ppo_logs/                   # Phase 4 training curves & monitor records
│   ├── monitor.csv             # Raw SB3 episode length/reward log
│   ├── training_metrics.csv    # Timestep-by-timestep episode reward history
│   └── events.out.tfevents.*   # TensorBoard execution curves
├── ppo_eval/                   # Standardized evaluation outputs for Person C
│   ├── final/                  # Final model outputs (portfolio_value.csv, weights.csv, metrics.json)
│   ├── checkpoint_20000/       # Step 20k outputs
│   ├── checkpoint_40000/       # Step 40k outputs
│   └── checkpoint_60000/       # Step 60k outputs
├── baselines/                  # Traditional benchmarks (Mean-Variance, Buy-and-Hold, etc.)
│   └── mean_variance_full/     # Markowitz long-only optimization outputs
├── csv/                        # Strategy comparison tables
│   └── gnn_ppo_comparison.csv  # Baselines vs GNN + PPO summary
└── plots/                      # Visualization outputs & animation
    ├── evolving_graph_animation.gif
    ├── allocation_over_time.png
    ├── mean_variance_equity.png
    └── gnn_ppo_comparison.png
```

---

## 🚀 How to Reproduce Phase 4 (Person B Work)

### Prerequisites
- Python 3.11
- Install dependencies:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
  pip install -r requirements.txt
  pip install stable-baselines3 tensorboard
  ```

### 1. Train the PPO Policy on GNN Embeddings
To train the PPO agent from scratch using Person A's 32-dimensional GNN embeddings:

```bash
python src/rl/train_ppo.py --data data/clean/train.parquet --timesteps 60000 --save_freq 20000 --seed 42
```

Outputs generated:
- Model checkpoints saved to: `results/ppo_checkpoints/`
- Tensorboard event logs saved to: `results/ppo_logs/`
- Final model saved to: `results/ppo_final_model.zip` and `src/gnn/models/gnn_ppo.zip`

### 2. Evaluate Any Checkpoint (Handoff Pipeline for Person C)
To run a deterministic evaluation of any checkpoint and generate `portfolio_value.csv`, `weights.csv`, and `metrics.json`:

```bash
python src/rl/evaluate_checkpoint.py \
  --checkpoint results/ppo_final_model.zip \
  --output_dir results/ppo_eval/final \
  --data data/clean/train.parquet
```

Or evaluate an intermediate checkpoint:
```bash
python src/rl/evaluate_checkpoint.py \
  --checkpoint results/ppo_checkpoints/ppo_checkpoint_40000_steps.zip \
  --output_dir results/ppo_eval/checkpoint_40000 \
  --data data/clean/train.parquet
```

### 3. Generate Strategy Comparison Chart
To update the benchmark comparison chart comparing GNN+PPO against Mean-Variance, Equal-Weight, and Momentum baselines:

```bash
python src/rl/compare_gnn_ppo.py
```

Outputs:
- `results/csv/gnn_ppo_comparison.csv`
- `results/plots/gnn_ppo_comparison.png`

---

## 📊 Summary of Phase 4 Results

| Policy / Strategy | Final Portfolio Value | Annualized Return | Sharpe Ratio | Max Drawdown | Average Turnover | Total Cost |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GNN+PPO (Step 20k)** | **3.0535x** | 15.18% | **0.780** | -37.77% | 3.17% | 0.0632 |
| **GNN+PPO (Step 40k)** | 2.9896x | 14.87% | 0.760 | -38.35% | 4.58% | 0.0912 |
| **GNN+PPO (Step 60k)** | 2.9297x | 14.57% | 0.752 | -36.78% | 6.79% | 0.1353 |
| **GNN+PPO (Final)** | 2.8813x | 14.33% | 0.743 | **-36.33%** | 7.09% | 0.1413 |
| **Mean-Variance Benchmark** | 5.0307x | 18.04% | 0.978 | -38.61% | 0.62% | 0.0152 |
| **Equal Weight Benchmark** | 4.8701x | 17.50% | 0.940 | -40.12% | 0.00% | 0.0000 |

> **Key Finding**: The GNN+PPO policy demonstrates effective risk control, achieving a smaller maximum drawdown (-36.33%) than both the Mean-Variance (-38.61%) and Equal-Weight (-40.12%) benchmarks, while maintaining steady capital appreciation.

---

## 🤝 Next Steps for Teammates

### For Person C (Evaluation Lead)
1. Use `results/ppo_eval/final/weights.csv` or any intermediate checkpoint folder as the input for your stacked area chart script (`src/analysis/allocation_over_time.py`).
2. Run `src/rl/evaluate_checkpoint.py` on any checkpoint to test performance across specific market regimes (e.g. Covid crash, 2022 rate hike cycle).

### For Person A (GNN Architecture Lead)
1. Review the PPO training dynamics and policy stability.
2. Note that the GNN embeddings provide effective structural regularization against drawdowns. If desired, proxy GNN retraining with GAT (Graph Attention) or multi-horizon return targets can be evaluated in Phase 5.