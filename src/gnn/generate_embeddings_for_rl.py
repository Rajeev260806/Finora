import os
import numpy as np
import torch
import pandas as pd
from temporal_graph_dataset import TemporalGraphDataset
from improved_gnn_model import ImprovedStockGNN
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
RETURNS_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")
MODEL_PATH = os.path.join(BASE_DIR, "src", "gnn", "models", "best_proxy_gnn.pth")
EMBEDDING_DIR_RL = os.path.join(BASE_DIR, "src", "gnn", "embeddings_rl")

# Ensure embedding directory exists
os.makedirs(EMBEDDING_DIR_RL, exist_ok=True)

# Load returns data (needed for the dataset to work)
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

# Dataset
dataset = TemporalGraphDataset(GRAPH_DIR)
print(f"Number of graphs: {len(dataset)}")

print("Generating embeddings for RL agent...")
with torch.no_grad():
    for idx in range(len(dataset)):
        graph_data = dataset[idx]
        graph_data = graph_data.to(device)
        # Get embeddings from the GNN part (before predictor)
        embeddings = model.gnn(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        embeddings_np = embeddings.cpu().numpy()
        embedding_filename = f"embeddings_rl_{idx:04d}.npy"
        embedding_path = os.path.join(EMBEDDING_DIR_RL, embedding_filename)
        np.save(embedding_path, embeddings_np)
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} embeddings")

print(f"Embeddings saved to: {EMBEDDING_DIR_RL}")