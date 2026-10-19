import os
import numpy as np
import torch
import pandas as pd
from torch_geometric.loader import DataLoader
from temporal_graph_dataset import TemporalGraphDataset
from improved_gnn_model import ImprovedStockGNN
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
RETURNS_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "src", "gnn", "models", "best_proxy_gnn.pth")

# Load returns data
returns_df = pd.read_parquet(RETURNS_DATA_PATH)
returns_wide = returns_df["Returns Daily"].unstack(level="Ticker")
returns_wide = returns_wide.fillna(0.0)
returns_wide = returns_wide.sort_index(axis=1)

# Parameters
INPUT_FEATURES = 9
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32
DROPOUT_RATE = 0.2
GAT_HEADS = 4
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the same model class as used for training
class GNNForProxyTask(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout_rate=0.2, gat_heads=4):
        super().__init__()
        self.gnn = ImprovedStockGNN(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            dropout_rate=dropout_rate,
            gat_heads=gat_heads,
            use_gat=True
        )
        self.predictor = nn.Linear(output_features, 1)

    def forward(self, x, edge_index, edge_weight):
        embeddings = self.gnn(x, edge_index, edge_weight)
        out = self.predictor(embeddings)
        return out.squeeze(-1)

# Load the trained model
model = GNNForProxyTask(
    input_features=INPUT_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    output_features=OUTPUT_FEATURES,
    dropout_rate=DROPOUT_RATE,
    gat_heads=GAT_HEADS
).to(device)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Dataset and DataLoader for evaluation
dataset = TemporalGraphDataset(GRAPH_DIR)
loader = DataLoader(dataset, batch_size=16, shuffle=False, follow_batch=['x'])

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index, batch.edge_weight)

        # Get target returns
        node_graph_indices = batch.batch
        unique_graph_indices = torch.unique(node_graph_indices)

        batch_targets = []
        for graph_idx in unique_graph_indices:
            graph_idx_item = graph_idx.item()
            graph_data = dataset[graph_idx_item]
            date_str = graph_data.date_str[0]
            if date_str in returns_wide.index:
                date_returns = returns_wide.loc[date_str]
                tickers_graph = graph_data.tickers
                next_returns = date_returns[tickers_graph].values
            else:
                next_returns = np.zeros(len(tickers_graph))
            batch_targets.append(next_returns)

        target_returns = np.concatenate(batch_targets, axis=0)
        target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)

        all_preds.append(out.detach().cpu().numpy())
        all_targets.append(target_tensor.cpu().numpy())

preds = np.concatenate(all_preds, axis=0)
targets = np.concatenate(all_targets, axis=0)

from sklearn.metrics import r2_score, mean_absolute_error
r2 = r2_score(targets, preds)
mae = mean_absolute_error(targets, preds)

print(f"Proxy Task Evaluation:")
print(f"  R2 Score: {r2:.6f}")
print(f"  MAE: {mae:.6f}")