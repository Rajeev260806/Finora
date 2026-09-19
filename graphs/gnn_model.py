import os
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

GNN_DATA_DIR = os.path.join(
    BASE_DIR,
    "gnn_data"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "gnn_models"
)

os.makedirs(MODEL_DIR, exist_ok=True)

INPUT_FEATURES = 5
HIDDEN_FEATURES = 32
OUTPUT_FEATURES = 16


# ============================================================
# GNN MODEL
# ============================================================

class StockGNN(torch.nn.Module):

    def __init__(
        self,
        input_features,
        hidden_features,
        output_features
    ):
        super().__init__()

        self.conv1 = GCNConv(
            input_features,
            hidden_features
        )

        self.conv2 = GCNConv(
            hidden_features,
            output_features
        )

    def forward(self, x, edge_index, edge_weight):

        x = self.conv1(
            x,
            edge_index,
            edge_weight
        )

        x = F.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x


# ============================================================
# LOAD GRAPH DATA
# ============================================================

def load_graph(graph_index):

    filename = f"graph_{graph_index:04d}.npz"

    path = os.path.join(
        GNN_DATA_DIR,
        filename
    )

    data = np.load(path)

    x = torch.tensor(
        data["node_features"],
        dtype=torch.float
    )

    edge_index = torch.tensor(
        data["edge_index"],
        dtype=torch.long
    ).t().contiguous()

    edge_weight = torch.tensor(
        data["edge_weight"],
        dtype=torch.float
    )

    sector_id = torch.tensor(
        data["sector_ids"],
        dtype=torch.long
    )

    graph_date = str(data["date"])

    graph = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
        sector_id=sector_id
    )

    return graph, graph_date


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STOCK GNN")
    print("=" * 60)

    # --------------------------------------------------------
    # Load first graph
    # --------------------------------------------------------

    graph, graph_date = load_graph(0)

    print("\nGraph loaded")
    print("Date:", graph_date)

    print(
        "Number of nodes:",
        graph.num_nodes
    )

    print(
        "Number of edges:",
        graph.num_edges
    )

    print(
        "Node feature shape:",
        graph.x.shape
    )

    print(
        "Edge index shape:",
        graph.edge_index.shape
    )

    print(
        "Edge weight shape:",
        graph.edge_weight.shape
    )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = StockGNN(
        input_features=INPUT_FEATURES,
        hidden_features=HIDDEN_FEATURES,
        output_features=OUTPUT_FEATURES
    )

    print("\nGNN Model:")
    print(model)

    # --------------------------------------------------------
    # Generate node embeddings
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        embeddings = model(
            graph.x,
            graph.edge_index,
            graph.edge_weight
        )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    print("\n===== GNN VALIDATION =====")

    print(
        "Input shape:",
        graph.x.shape
    )

    print(
        "Embedding shape:",
        embeddings.shape
    )

    print(
        "Embedding dimension:",
        embeddings.shape[1]
    )

    print(
        "First node embedding:",
        embeddings[0]
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    embedding_path = os.path.join(
        MODEL_DIR,
        "embeddings_graph_0000.npy"
    )

    np.save(
        embedding_path,
        embeddings.numpy()
    )

    print(
        "\nEmbeddings saved to:",
        embedding_path
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = os.path.join(
        MODEL_DIR,
        "stock_gnn.pth"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    print(
        "Model saved to:",
        model_path
    )

    print("\n===== COMPLETED =====")