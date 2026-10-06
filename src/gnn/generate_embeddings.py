import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
from temporal_graph_dataset import TemporalGraphDataset

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
MODEL_DIR = os.path.join(BASE_DIR, "src", "gnn", "models")
EMBEDDING_DIR = os.path.join(BASE_DIR, "src", "gnn", "embeddings")

# Ensure embedding directory exists
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Model hyperparameters (must match the trained model)
INPUT_FEATURES = 5
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32
DROPOUT_RATE = 0.2

# ============================================================
# GNN MODEL DEFINITION (same as in proxy_task_trainer.py)
# ============================================================

class StockGNN(torch.nn.Module):
    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        dropout_rate=0.0
    ):
        super().__init__()
        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, output_features)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout_rate, train=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

# ============================================================
# MAIN EMBEDDING GENERATION
# ============================================================

def main():
    print("=" * 60)
    print("GENERATING NODE EMBEDDINGS FOR TRADING ENVIRONMENT")
    print("=" * 60)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the trained model
    model_path = os.path.join(MODEL_DIR, "proxy_task_gnn.pth")
    print(f"Loading trained model from: {model_path}")

    model = StockGNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        dropout_rate=DROPOUT_RATE
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # Set to evaluation mode
    print("Model loaded successfully.")

    # Load the dataset
    print(f"\nLoading graph data from: {GRAPH_DIR}")
    dataset = TemporalGraphDataset(GRAPH_DIR)
    print(f"Found {len(dataset)} graph snapshots.")

    # Generate embeddings for each graph
    print("\nGenerating embeddings...")
    for idx in range(len(dataset)):
        # Get graph data
        graph_data = dataset[idx]
        graph_data = graph_data.to(device)

        # Generate embeddings (no gradient)
        with torch.no_grad():
            embeddings = model(
                graph_data.x,
                graph_data.edge_index,
                graph_data.edge_weight
            )

        # Convert to numpy and save
        embeddings_np = embeddings.cpu().numpy()
        embedding_filename = f"embeddings_graph_{idx:04d}.npy"
        embedding_path = os.path.join(EMBEDDING_DIR, embedding_filename)
        np.save(embedding_path, embeddings_np)

        # Progress update
        if (idx + 1) % 100 == 0 or idx == len(dataset) - 1:
            print(f"  Processed {idx + 1}/{len(dataset)} graphs - Saved: {embedding_filename}")

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 60)
    print(f"Embeddings saved to: {EMBEDDING_DIR}")
    print(f"Number of embedding files: {len(dataset)}")
    print(f"Embedding shape per graph: [{dataset[0].num_nodes}, {OUTPUT_FEATURES}]")

if __name__ == "__main__":
    main()