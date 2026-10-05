import os
import pickle
import numpy as np
import pandas as pd


# ==================================================
# Paths
# ==================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "clean",
    "full_dataset.parquet"
)

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "saved_graphs"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "gnn_data"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==================================================
# Configuration
# ==================================================

NODE_FEATURES = [
    "Returns Daily",
    "Volatility 20D",
    "Volume Normalized",
    "Volatility Annual",
    "Log Return"
]


# ==================================================
# Load Dataset
# ==================================================

print("=" * 60)
print("PREPARING GNN DATA")
print("=" * 60)

data = pd.read_parquet(DATA_PATH)

print("Dataset loaded.")
print("Rows:", len(data))


# ==================================================
# Load All Graph Files
# ==================================================

graph_files = sorted(
    [
        f
        for f in os.listdir(GRAPHS_DIR)
        if f.endswith(".pkl")
    ]
)

print("Graph files:", len(graph_files))

if len(graph_files) == 0:
    raise FileNotFoundError(
        "No graph files found."
    )


# ==================================================
# Process Every Graph
# ==================================================

total_processed = 0

for count, graph_file in enumerate(
    graph_files,
    start=1
):

    graph_path = os.path.join(
        GRAPHS_DIR,
        graph_file
    )

    # --------------------------------------------------
    # Load graph
    # --------------------------------------------------

    with open(graph_path, "rb") as file:
        G = pickle.load(file)

    tickers = list(G.nodes)

    # --------------------------------------------------
    # Prepare node features
    # --------------------------------------------------

    node_features = []

    for ticker in tickers:

        try:

            row = data.loc[
                (G.graph["date"], ticker)
            ]

            features = (
                row[NODE_FEATURES]
                .astype(float)
                .values
            )

        except KeyError:

            features = np.zeros(
                len(NODE_FEATURES)
            )

        node_features.append(features)

    node_features = np.array(
        node_features,
        dtype=np.float32
    )

    # --------------------------------------------------
    # Handle missing values
    # --------------------------------------------------

    node_features = np.nan_to_num(
        node_features,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # --------------------------------------------------
    # Create edge index
    # --------------------------------------------------

    edge_index = []
    edge_weight = []

    ticker_to_index = {
        ticker: i
        for i, ticker in enumerate(tickers)
    }

    for u, v, attributes in G.edges(data=True):

        u_index = ticker_to_index[u]
        v_index = ticker_to_index[v]

        # Forward edge
        edge_index.append(
            [u_index, v_index]
        )

        # Reverse edge
        edge_index.append(
            [v_index, u_index]
        )

        correlation = float(
            attributes["correlation"]
        )

        edge_weight.append(
            correlation
        )

        edge_weight.append(
            correlation
        )

    # --------------------------------------------------
    # Handle graph with no edges
    # --------------------------------------------------

    if len(edge_index) == 0:

        edge_index = np.empty(
            (0, 2),
            dtype=np.int64
        )

        edge_weight = np.empty(
            (0,),
            dtype=np.float32
        )

    else:

        edge_index = np.array(
            edge_index,
            dtype=np.int64
        )

        edge_weight = np.array(
            edge_weight,
            dtype=np.float32
        )

    # --------------------------------------------------
    # Sector encoding
    # --------------------------------------------------

    sectors = [
        G.nodes[ticker]["sector"]
        for ticker in tickers
    ]

    unique_sectors = sorted(
        set(sectors)
    )

    sector_to_id = {
        sector: i
        for i, sector in enumerate(
            unique_sectors
        )
    }

    sector_ids = np.array(
        [
            sector_to_id[sector]
            for sector in sectors
        ],
        dtype=np.int64
    )

    # --------------------------------------------------
    # Save GNN data
    # --------------------------------------------------

    graph_index = graph_file.replace(
        "graph_",
        ""
    ).replace(
        ".pkl",
        ""
    )

    output_path = os.path.join(
        OUTPUT_DIR,
        f"graph_{graph_index}.npz"
    )

    np.savez(
        output_path,
        node_features=node_features,
        edge_index=edge_index,
        edge_weight=edge_weight,
        sector_ids=sector_ids,
        tickers=np.array(tickers),
        date=str(G.graph["date"])
    )

    total_processed += 1

    # --------------------------------------------------
    # Progress
    # --------------------------------------------------

    if (
        count == 1
        or count % 100 == 0
        or count == len(graph_files)
    ):

        print(
            f"Processed {count}/{len(graph_files)}"
        )


# ==================================================
# Validation
# ==================================================

print("\n" + "=" * 60)
print("GNN DATA GENERATION COMPLETED")
print("=" * 60)

print(
    "Graphs processed:",
    total_processed
)

print(
    "Output directory:",
    OUTPUT_DIR
)


# ==================================================
# Validate First Graph
# ==================================================

first_output = os.path.join(
    OUTPUT_DIR,
    "graph_0000.npz"
)

test_data = np.load(first_output)

print("\n===== VALIDATION =====")

print(
    "Files stored:",
    len([
        f for f in os.listdir(OUTPUT_DIR)
        if f.endswith(".npz")
    ])
)

print(
    "Node feature shape:",
    test_data["node_features"].shape
)

print(
    "Edge index shape:",
    test_data["edge_index"].shape
)

print(
    "Edge weight shape:",
    test_data["edge_weight"].shape
)

print(
    "Sector ID shape:",
    test_data["sector_ids"].shape
)

print(
    "First graph date:",
    test_data["date"]
)

print("\n===== COMPLETED =====")