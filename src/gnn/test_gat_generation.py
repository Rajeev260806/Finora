import os
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

# Test the GAT Hybrid model implementation
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

def main():
    print("Testing GAT Hybrid model implementation...")

    # Set up paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")

    # Load one graph for testing
    from temporal_graph_dataset import TemporalGraphDataset

    dataset = TemporalGraphDataset(GRAPH_DIR)
    filename = dataset.graph_files[0]
    filepath = os.path.join(GRAPH_DIR, filename)
    data = np.load(filepath)

    print(f"Testing on graph from date: {data['date'].item()}")

    # Create Data object for GNN processing
    x = torch.tensor(data['node_features'], dtype=torch.float32)
    edge_index = torch.tensor(data['edge_index'], dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(data['edge_weight'], dtype=torch.float32)

    # Create the GAT Hybrid model
    model = GATHybridStockGNN(
        input_features=5,
        hidden_features=64,
        output_features=32,
        dropout_rate=0.2,
        gat_heads=4
    )

    print(f"\nGAT Hybrid Model:")
    print(model)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")

    # Generate embeddings
    model.eval()
    with torch.no_grad():
        embeddings = model(
            x,
            edge_index,
            edge_weight
        )

    embeddings_np = embeddings.cpu().numpy()
    print(f"\nEmbeddings shape: {embeddings_np.shape}")
    print(f"Embeddings stats - Mean: {embeddings_np.mean():.6f}, Std: {embeddings_np.std():.6f}")

    # Check sector separation
    sector_ids = data['sector_ids']

    from itertools import combinations
    same_sector_dists = []
    diff_sector_dists = []

    for i in range(len(embeddings_np)):
        for j in range(i+1, len(embeddings_np)):
            dist = np.linalg.norm(embeddings_np[i] - embeddings_np[j])
            if sector_ids[i] == sector_ids[j]:
                same_sector_dists.append(dist)
            else:
                diff_sector_dists.append(dist)

    if same_sector_dists and diff_sector_dists:
        avg_same = np.mean(same_sector_dists)
        avg_diff = np.mean(diff_sector_dists)
        ratio = avg_same / avg_diff if avg_diff > 0 else 0
        print(f"\nSector separation - Same/Dist ratio: {ratio:.4f}")
        if ratio < 1.0:
            print("PASS: Embeddings show good sector separation")
        else:
            print("FAIL: Embeddings show poor sector separation")

    # Compare with original GCN embeddings if available
    try:
        original_emb = np.load('src/gnn/embeddings/embeddings_graph_0000.npy')
        print(f"\nOriginal GCN embeddings shape: {original_emb.shape}")
        print(f"Original GCN embeddings stats - Mean: {original_emb.mean():.6f}, Std: {original_emb.std():.6f}")

        # Calculate sector separation for original
        orig_same_dists = []
        orig_diff_dists = []

        for i in range(len(original_emb)):
            for j in range(i+1, len(original_emb)):
                dist = np.linalg.norm(original_emb[i] - original_emb[j])
                if sector_ids[i] == sector_ids[j]:
                    orig_same_dists.append(dist)
                else:
                    orig_diff_dists.append(dist)

        if orig_same_dists and orig_diff_dists:
            orig_avg_same = np.mean(orig_same_dists)
            orig_avg_diff = np.mean(orig_diff_dists)
            orig_ratio = orig_avg_same / orig_avg_diff if orig_avg_diff > 0 else 0
            print(f"Original GCN sector separation - Same/Dist ratio: {orig_ratio:.4f}")
            improvement = orig_ratio / ratio if ratio > 0 else 0
            print(f"Improvement factor: {improvement:.2f}x")
    except Exception as e:
        print(f"Could not load original embeddings for comparison: {e}")

if __name__ == "__main__":
    main()