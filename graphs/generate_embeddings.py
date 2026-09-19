import os
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


# ============================================================
# GNN MODEL
# ============================================================

class StockGNN(nn.Module):

    def __init__(self, input_dim=5, hidden_dim=32, embedding_dim=16):
        super().__init__()

        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, embedding_dim)

    def forward(self, x, edge_index, edge_weight):

        x = self.conv1(
            x,
            edge_index,
            edge_weight
        )

        x = torch.relu(x)

        x = self.conv2(
            x,
            edge_index,
            edge_weight
        )

        return x


# ============================================================
# LOAD GNN DATA
# ============================================================

def load_graph(index):

    path = os.path.join(
        "graphs",
        "gnn_data",
        f"graph_{index:04d}.npz"
    )

    data = np.load(path)

    x = torch.tensor(
        data["node_features"],
        dtype=torch.float32
    )

    edge_index = torch.tensor(
        data["edge_index"],
        dtype=torch.long
    ).t().contiguous()

    edge_weight = torch.tensor(
        data["edge_weight"],
        dtype=torch.float32
    )

    date = str(data["date"])

    return x, edge_index, edge_weight, date


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("GENERATING GNN EMBEDDINGS")
    print("=" * 60)

    input_dir = os.path.join(
        "graphs",
        "gnn_data"
    )

    output_dir = os.path.join(
        "graphs",
        "gnn_models",
        "embeddings"
    )

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Find available graph files
    # --------------------------------------------------------

    graph_files = sorted(
        [
            f for f in os.listdir(input_dir)
            if f.endswith(".npz")
        ]
    )

    total_graphs = len(graph_files)

    print("Total GNN graphs:", total_graphs)

    if total_graphs == 0:
        print("No GNN graph files found.")
        exit()

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = StockGNN(
        input_dim=5,
        hidden_dim=32,
        embedding_dim=16
    )

    model.eval()

    print("\nModel:")
    print(model)

    # --------------------------------------------------------
    # Generate embeddings
    # --------------------------------------------------------

    with torch.no_grad():

        for count, filename in enumerate(
            graph_files,
            start=1
        ):

            path = os.path.join(
                input_dir,
                filename
            )

            data = np.load(path)

            x = torch.tensor(
                data["node_features"],
                dtype=torch.float32
            )

            edge_index = torch.tensor(
                data["edge_index"],
                dtype=torch.long
            ).t().contiguous()

            edge_weight = torch.tensor(
                data["edge_weight"],
                dtype=torch.float32
            )

            embeddings = model(
                x,
                edge_index,
                edge_weight
            )

            embeddings = embeddings.numpy()

            index = filename.replace(
                "graph_",
                ""
            ).replace(
                ".npz",
                ""
            )

            output_path = os.path.join(
                output_dir,
                f"embedding_{index}.npy"
            )

            np.save(
                output_path,
                embeddings
            )

            if count % 100 == 0 or count == 1:
                print(
                    f"Processed {count}/{total_graphs}"
                )

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETED")
    print("=" * 60)

    print("Graphs processed:", total_graphs)

    print(
        "Embedding shape per graph:",
        "(20, 16)"
    )

    print(
        "Saved to:",
        output_dir
    )

    # --------------------------------------------------------
    # Validate first embedding
    # --------------------------------------------------------

    first_path = os.path.join(
        output_dir,
        "embedding_0000.npy"
    )

    first_embedding = np.load(first_path)

    print("\n===== VALIDATION =====")

    print(
        "First embedding shape:",
        first_embedding.shape
    )

    print(
        "First stock embedding:",
        first_embedding[0]
    )

    print("\n===== COMPLETED =====")