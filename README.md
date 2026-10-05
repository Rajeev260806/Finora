# ML_Package

A machine learning package for portfolio optimization using Graph Neural Networks (GNN) and Reinforcement Learning (RL).

## Project Structure

```
ML_Package/
├── data/                    # Raw and processed datasets
│   ├── clean/               # Cleaned datasets (parquet format)
│   │   ├── full_dataset.parquet
│   │   ├── train.parquet
│   │   └── test.parquet
│   └── raw/                 # Raw data files
├── docs/                    # Documentation and guides
│   ├── README.md            # This file (docs overview)
│   ├── *_GUIDE.md           # Phase-specific guides for each team member
│   ├── phase_wise_split.md  # High-level project plan
│   ├── project_ideas.md     # Initial project concepts
│   └── project_notes.md     # Ongoing project notes
├── graphs/                  # Graph construction and analysis
│   ├── gnn_data/            # Processed graph data for GNN (.npz files)
│   ├── graph_construction_config.json  # Graph construction parameters
│   ├── graph_features.csv   # Features used for graph construction
│   └── saved_graphs/        # Raw NetworkX graph pickles (ignored by git)
├── models/                  # Saved machine learning models
│   └── gnn_ppo.zip          # Saved PPO model with GNN embeddings
├── results/                 # Output files and visualizations
│   ├── baselines/           # Baseline model results
│   ├── csv/                 # CSV output files
│   ├── graphs/              # Graph visualizations
│   └── plots/               # Plots and charts
├── src/                     # Source code
│   ├── analysis/            # Data analysis and utilities
│   ├── baselines/           # Baseline portfolio strategies
│   ├── environment/         # Trading environment (PortfolioEnv)
│   ├── gnn/                 # Graph Neural Network implementation
│   │   ├── gnn_model.py     # GNN model definitions
│   │   ├── prepare_gnn_data.py  # Data preparation for GNN
│   │   ├── generate_embeddings.py # Embedding generation
│   │   ├── validate_embeddings.py # Embedding validation
│   │   ├── models/          # Saved GNN models (.pth files)
│   │   └── embeddings/      # Saved node embeddings (.npy files)
│   ├── rl/                  # Reinforcement Learning components
│   │   ├── train_gnn_ppo.py # PPO training script
│   │   ├── evaluate_gnn_ppo.py # PPO evaluation script
│   │   └── compare_gnn_ppo.py # Model comparison script
│   ├── visualization/       # Visualization utilities
│   └── __init__.py
├── tests/                   # Unit tests (to be added)
├── .gitignore               # Git ignore rules
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## Getting Started

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. The project is organized into phases as documented in the docs/ directory.

## Documentation

See the `docs/` directory for detailed guides:
- Phase-wise implementation plan: `docs/phase_wise_split.md`
- Individual responsibilities: `docs/PHASE_X_PERSON_Y_GUIDE.md`

## License

[Specify license here]
