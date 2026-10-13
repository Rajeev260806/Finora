import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GNN_DATA_DIR = os.path.join(
    BASE_DIR,
    "..",
    "..",
    "graphs",
    "gnn_data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

EMBEDDING_DIR = os.path.join(
    BASE_DIR,
    "embeddings"
)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_DIR, exist_ok=True)

INPUT_FEATURES = 5
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32
DROPOUT_RATE = 0.2
GAT_HEADS = 4  # Number of attention heads for GAT


# ============================================================
# IMPROVED GNN MODEL (GCN + GAT Hybrid)
# ============================================================

class ImprovedStockGNN(torch.nn.Module):
    """
    Improved GNN model combining GCN and GAT layers for better expressiveness.
    """

    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        dropout_rate=0.2,
        gat_heads=4,
        use_gat=True
    ):
        super().__init__()

        self.use_gat = use_gat
        self.dropout_rate = dropout_rate

        if use_gat:
            # GAT expects output features to be divisible by number of heads
            # We'll adjust hidden features accordingly
            gat_hidden = hidden_features // gat_heads
            if gat_hidden < 1:
                gat_hidden = 1

            self.conv1 = GATConv(
                input_features,
                gat_hidden,
                heads=gat_heads,
                dropout=dropout_rate,
                concat=True  # Concatenate heads -> gat_hidden * gat_heads = hidden_features
            )

            # Second GAT layer
            self.conv2 = GATConv(
                gat_hidden * gat_heads,  # Input from first GAT layer
                output_features,
                heads=1,  # Single head for output to get desired dimension
                dropout=dropout_rate,
                concat=False
            )
        else:
            # Standard GCN layers
            self.conv1 = GCNConv(
                input_features,
                hidden_features
            )

            self.conv2 = GCNConv(
                hidden_features,
                output_features
            )

    def forward(self, x, edge_index, edge_weight):
        # Apply dropout to edge weights for regularization (optional)
        if self.training and torch.rand(1).item() < 0.1:  # 10% chance to dropout edges
            # Randomly zero out some edge weights
            mask = torch.rand_like(edge_weight) > 0.1
            edge_weight = edge_weight * mask.float()

        if self.use_gat:
            x = self.conv1(x, edge_index, edge_weight)
            x = F.elu(x)  # GAT typically uses ELU activation
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            x = self.conv2(x, edge_index, None)  # GAT doesn't use edge_weight in same way
            # Note: GATv2 can use edge weights, but standard GAT doesn't
        else:
            x = self.conv1(
                x,
                edge_index,
                edge_weight
            )

            x = F.relu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            x = self.conv2(
                x,
                edge_index,
                edge_weight
            )

        return x


# ============================================================
# ALTERNATIVE: PURE GAT MODEL
# ============================================================

class PureGATStockGNN(torch.nn.Module):
    """
    Pure GAT model for comparison.
    """

    def __init__(
        self,
        input_features,
        hidden_features,
        output_features,
        dropout_rate=0.2,
        heads=4
    ):
        super().__init__()

        # GAT layer 1
        self.gat1 = GATConv(
            input_features,
            hidden_features // heads,
            heads=heads,
            dropout=dropout_rate,
            add_self_loops=True,
            concat=True
        )

        # GAT layer 2
        self.gat2 = GATConv(
            hidden_features,
            output_features // heads,
            heads=heads,
            dropout=dropout_rate,
            add_self_loops=True,
            concat=True
        )

        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight):
        # GAT doesn't typically use edge_weight in the same way as GCN
        # But we accept it for API consistency
        # We can still apply dropout to edges for regularization
        if self.training:
            # Occasionally dropout edges
            if torch.rand(1).item() < 0.1:
                mask = torch.rand_like(edge_weight) > 0.1
                edge_weight = edge_weight * mask.float()
            # Note: GAT implementation above doesn't use edge_weight
            # For simplicity, we'll ignore edge_weight in GAT (common practice)

        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.gat2(x, edge_index)
        # Output shape should be [num_nodes, output_features]

        return x


# ============================================================
# LOAD GRAPH DATA (same as before)
# ============================================================

def load_graph(graph_index):
    filename = f"graph_{graph_index:04d}.npz"
    path = os.path.join(GNN_DATA_DIR, filename)
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
# MAIN - COMPARE MODELS
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("COMPARING IMPROVED GNN MODELS")
    print("=" * 70)

    # Load first graph for testing
    graph, graph_date = load_graph(0)

    print(f"Testing on graph from date: {graph_date}")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    print(f"Node feature shape: {graph.x.shape}")

    # Test different model configurations
    models_to_test = [
        ("GCN (Original)", ImprovedStockGNN(5, 64, 32, 0.2, use_gat=False)),
        ("GAT Hybrid", ImprovedStockGNN(5, 64, 32, 0.2, use_gat=True, gat_heads=4)),
        ("Pure GAT", PureGATStockGNN(5, 64, 32, 0.2, heads=4))
    ]

    for name, model in models_to_test:
        print(f"\n{name}:")
        print(f"  Model architecture:\n{model}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  Total parameters: {total_params:,}")

        # Generate embeddings
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'conv1'):  # GCN or GAT hybrid
                embeddings = model(graph.x, graph.edge_index, graph.edge_weight)
            else:  # Pure GAT
                embeddings = model(graph.x, graph.edge_index, graph.edge_weight)  # Still pass edge_weight for consistency

        print(f"  Embeddings shape: {embeddings.shape}")
        print(f"  Embedding stats - Mean: {embeddings.mean().item():.6f}, Std: {embeddings.std().item():.6f}")

        # Check sector separation (quick test)
        sector_ids = graph.sector_id.numpy()
        embeddings_np = embeddings.numpy()

        # Calculate average intra-sector vs inter-sector distance
        from itertools import combinations
        same_sector_distances = []
        diff_sector_distances = []

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
            print(f"  Sector separation - Same/Dist ratio: {ratio:.4f} (<1.0 is good)")