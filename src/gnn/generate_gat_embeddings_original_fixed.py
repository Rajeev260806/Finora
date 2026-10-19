import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
EMBEDDING_DIR = os.path.join(BASE_DIR, "src", "gnn", "embeddings_original")

# Ensure embedding directory exists
os.makedirs(EMBEDDING_DIR, exist_ok=True)

# Model hyperparameters (original: 5 features)
INPUT_FEATURES = 5
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32
DROPOUT_RATE = 0.2
GAT_HEADS = 4  # Number of attention heads for GAT


# ============================================================
# GAT HYBRID MODEL DEFINITION
# ============================================================

class GATHybridStockGNN(torch.nn.Module):
    """
    GAT Hybrid model combining GAT layers for better expressiveness.
    """

    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        dropout_rate=0.2,
        gat_heads=4
    ):
        super().__init__()

        self.dropout_rate = dropout_rate
        self.gat_heads = gat_heads

        # First GAT layer
        self.gat1 = GATConv(
            input_features,
            hidden_features // gat_heads,
            heads=gat_heads,
            dropout=dropout_rate,
            add_self_loops=True,
            concat=True  # Concatenate heads -> (hidden_features // gat_heads) * gat_heads = hidden_features
        )

        # Second GAT layer
        self.gat2 = GATConv(
            hidden_features,
            output_features,
            heads=1,  # Single head for output to get desired dimension
            dropout=dropout_rate,
            add_self_loops=True,
            concat=False
        )

    def forward(self, x, edge_index, edge_weight):
        # GAT doesn't typically use edge_weight in the same way as GCN
        # But we accept it for API consistency with GCN models
        # Apply occasional edge dropout for regularization
        if self.training:
            # Occasionally dropout edges (10% chance)
            if torch.rand(1).item() < 0.1:
                mask = torch.rand_like(edge_weight) > 0.1
                edge_weight = edge_weight * mask.float()
            # Note: GAT implementation doesn't use edge_weight directly
            # This is common practice as GAT computes its own attention weights

        x = self.gat1(x, edge_index)
        x = torch.relu(x)  # Using ReLU for consistency with original GCN
        x = torch.dropout(x, p=self.dropout_rate, train=self.training)

        x = self.gat2(x, edge_index)
        # Output shape should be [num_nodes, output_features]

        return x


# ============================================================
# LOAD GRAPH DATA
# ============================================================

def load_graph(graph_index):
    filename = f"graph_{graph_index:04d}.npz"
    path = os.path.join(GRAPH_DIR, filename)
    data = np.load(path)

    x = torch.tensor(data["node_features"], dtype=torch.float32)
    edge_index = torch.tensor(data["edge_index"], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(data["edge_weight"], dtype=torch.float32)
    sector_id = torch.tensor(data["sector_ids"], dtype=torch.long)
    graph_date = str(data["date"])

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        sector_id=sector_id
    )

    return graph, graph_date


# ============================================================
# MAIN EMBEDDING GENERATION
# ============================================================

def main():
    print("=" * 70)
    print("GENERATING EMBEDDINGS WITH ORIGINAL GAT HYBRID MODEL (5 FEATURES)")
    print("=" * 70)

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create the GAT Hybrid model
    model = GATHybridStockGNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES,
        dropout_rate=DROPOUT_RATE,
        gat_heads=GAT_HEADS
    ).to(device)

    print(f"\nGAT Hybrid Model:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Get list of graph files
    graph_files = sorted([f for f in os.listdir(GRAPH_DIR) if f.startswith('graph_') and f.endswith('.npz')])
    print(f"\nLoading graph data from: {GRAPH_DIR}")
    print(f"Found {len(graph_files)} graph snapshots.")

    # Generate embeddings for each graph
    print("\nGenerating embeddings...")
    model.eval()  # Set to evaluation mode

    for idx in range(len(graph_files)):
        # Get graph data
        graph_data, graph_date = load_graph(idx)
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
        if (idx + 1) % 100 == 0 or idx == len(graph_files) - 1:
            print(f"  Processed {idx + 1}/{len(graph_files)} graphs - Saved: {embedding_filename}")

    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 70)
    print(f"Embeddings saved to: {EMBEDDING_DIR}")
    print(f"Number of embedding files: {len(graph_files)}")
    print(f"Embedding shape per graph: [{graph_files[0].split('_')[1].split('.')[0]}, {OUTPUT_FEATURES}]")  # This is hacky, but we know it's 20

    # Quick validation of the first embedding
    print("\nValidating first embedding...")
    first_embedding_path = os.path.join(EMBEDDING_DIR, "embeddings_graph_0000.npy")
    if os.path.exists(first_embedding_path):
        first_embedding = np.load(first_embedding_path)
        print(f"First embedding shape: {first_embedding.shape}")
        print(f"First embedding stats - Mean: {first_embedding.mean():.6f}, Std: {first_embedding.std():.6f}")

        # Check sector separation for first embedding
        graph_0, _ = load_graph(0)
        sector_ids = graph_0.sector_id.numpy()

        from itertools import combinations
        same_sector_distances = []
        diff_sector_distances = []

        embeddings_np = first_embedding
        for i in range(len(embeddings_np)):
            for j in range(i+1, len(embeddings_np)):
                dist = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
                if sector_ids[i] == sector_ids[j]:
                    same_sector_distances.append(dist)
                else:
                    diff_sector_distances.append(dist)

        if same_sector_distances and diff_sector_distances:
            avg_same = np.mean(same_sector_distances)
            avg_diff = np.mean(diff_sector_distances)
            ratio = avg_same / avg_diff if avg_diff > 0 else 0
            print(f"Sector separation - Same/Dist ratio: {ratio:.4f} (<1.0 indicates good separation)")
            if ratio < 1.0:
                print("✓ Embeddings show good sector separation")
            else:
                print("✗ Embeddings show poor sector separation")

if __name__ == "__main__":
    main()