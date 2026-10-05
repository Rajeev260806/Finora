import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import pickle


# --------------------------------------------------
# Correlation Matrix
# --------------------------------------------------

def get_correlation_matrix(returns_df, window, idx):
    if idx < window:
        return None

    window_returns = returns_df.iloc[idx - window:idx]
    corr_matrix = window_returns.corr()
    corr_matrix = corr_matrix.fillna(0)

    return corr_matrix


# --------------------------------------------------
# Threshold Correlation Matrix
# --------------------------------------------------

def threshold_correlation(corr_matrix, threshold=0.6, use_absolute=True):
    if use_absolute:
        adj_mat = np.abs(corr_matrix.to_numpy()).copy()
    else:
        adj_mat = corr_matrix.to_numpy().copy()

    np.fill_diagonal(adj_mat, 0)

    adj_mat[adj_mat < threshold] = 0

    adj_mat = pd.DataFrame(
        adj_mat,
        index=corr_matrix.index,
        columns=corr_matrix.columns
    )

    return adj_mat


# --------------------------------------------------
# Convert Adjacency Matrix to NetworkX Graph
# --------------------------------------------------

def adjacency_to_graph(adj_matrix, tickers, date, corr_values=None):

    G = nx.Graph()

    ticker_to_sector = {
        "AAPL": "Technology",
        "MSFT": "Technology",
        "NVDA": "Technology",
        "ADBE": "Technology",

        "JPM": "Finance",
        "BAC": "Finance",
        "GS": "Finance",
        "V": "Finance",

        "XOM": "Energy",
        "CVX": "Energy",
        "COP": "Energy",
        "SLB": "Energy",

        "NEE": "Utilities",
        "DUK": "Utilities",
        "SO": "Utilities",
        "AEP": "Utilities",

        "PLD": "Real Estate",
        "AMT": "Real Estate",
        "O": "Real Estate",
        "SPG": "Real Estate"
    }

    # Add nodes
    for ticker in tickers:
        sector = ticker_to_sector.get(ticker, "Unknown")
        G.add_node(ticker, sector=sector)

    # Add edges
    for i in range(len(tickers)):
        for j in range(i + 1, len(tickers)):

            weight = adj_matrix.iloc[i, j]

            if weight > 0:

                if corr_values is not None:
                    correlation = corr_values.iloc[i, j]
                else:
                    correlation = weight

                G.add_edge(
                    tickers[i],
                    tickers[j],
                    weight=weight,
                    correlation=correlation
                )

    # Store date as graph attribute
    G.graph["date"] = date

    return G


# --------------------------------------------------
# Sector Homophily
# --------------------------------------------------

def sector_homophily(G):

    same = sum(
        1
        for u, v in G.edges()
        if G.nodes[u]["sector"] == G.nodes[v]["sector"]
    )

    return (
        same / G.number_of_edges()
        if G.number_of_edges()
        else 0.0
    )


# --------------------------------------------------
# Load Saved Graph
# --------------------------------------------------

def load_graph_by_index(idx: int) -> nx.Graph:

    base_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "saved_graphs"
    )

    fname = f"graph_{idx:04d}.pkl"
    path = os.path.join(base_dir, fname)

    with open(path, "rb") as file:
        return pickle.load(file)


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    # --------------------------------------------------
    # Paths
    # --------------------------------------------------

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    PROJECT_DIR = os.path.dirname(BASE_DIR)

    DATA_PATH = os.path.join(
        PROJECT_DIR,
        "data",
        "clean",
        "train.parquet"
    )

    GRAPHS_DIR = os.path.join(
        BASE_DIR,
        "saved_graphs"
    )

    PLOTS_DIR = os.path.join(
        BASE_DIR,
        "plots"
    )

    # Create directories
    os.makedirs(GRAPHS_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # --------------------------------------------------
    # Load Data
    # --------------------------------------------------

    data = pd.read_parquet(DATA_PATH)

    returns = data["Log Return"].unstack(level="Ticker")

    window = 60

    tickers = returns.columns.tolist()

    # --------------------------------------------------
    # Create Example Graph
    # --------------------------------------------------

    corr_mat = get_correlation_matrix(
        returns,
        window,
        60
    )

    print(corr_mat)

    adj_matrix = threshold_correlation(
        corr_mat,
        threshold=0.6,
        use_absolute=True
    )

    print(adj_matrix)

    G = adjacency_to_graph(
        adj_matrix,
        tickers,
        date=returns.index[60],
        corr_values=corr_mat
    )

    # --------------------------------------------------
    # Graph Information
    # --------------------------------------------------

    print("\nSector homophily:", sector_homophily(G))

    print(
        "Number of nodes:",
        G.number_of_nodes()
    )

    print(
        "Number of edges:",
        G.number_of_edges()
    )

    print("\nNodes:")
    print(G.nodes(data=True))

    print("\nEdges:")
    print(G.edges(data=True))

    print("\nGraph attributes:")
    print(G.graph)

    # --------------------------------------------------
    # Correlation Heatmap
    # --------------------------------------------------

    date = G.graph["date"]

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_mat,
        annot=False,
        cmap="coolwarm",
        center=0
    )

    plt.title(
        f"Correlation Matrix - {date}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"corr_heatmap_{date.date()}.png"
        ),
        dpi=300
    )

    plt.close()

    # --------------------------------------------------
    # Graph Diagram
    # --------------------------------------------------

    plt.figure(figsize=(12, 10))

    pos = nx.spring_layout(
        G,
        seed=42
    )

    sector_colors = {
        "Technology": "skyblue",
        "Finance": "lightgreen",
        "Energy": "salmon",
        "Utilities": "gold",
        "Real Estate": "plum"
    }

    node_colors = [
        sector_colors.get(
            G.nodes[ticker]["sector"],
            "gray"
        )
        for ticker in G.nodes
    ]

    nx.draw(
        G,
        pos,
        node_color=node_colors,
        with_labels=False,
        node_size=800,
        edge_color="gray"
    )

    nx.draw_networkx_labels(
        G,
        pos,
        labels={
            ticker: ticker
            for ticker in G.nodes
        },
        font_size=10
    )

    plt.title(
        f"Graph on {date} - Nodes Colored by Sector"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            f"graph_diagram_{date.date()}.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------
    # Graph Validation
    # --------------------------------------------------

    print("\n===== Graph Validation =====")

    print("Date:", date)

    print(
        "Number of nodes:",
        G.number_of_nodes()
    )

    print(
        "Number of edges:",
        G.number_of_edges()
    )

    print("\nNode attributes:")

    for node, attributes in G.nodes(data=True):
        print(
            node,
            attributes
        )

    print("\nEdge attributes:")

    for u, v, attributes in G.edges(data=True):
        print(
            u,
            "-",
            v,
            attributes
        )

    # --------------------------------------------------
    # Create Graph for Every Trading Day
    # --------------------------------------------------

    def create_every_graphs():

        print("\n===== Creating All Graphs =====")

        total_graphs = 0

        for i in range(
            window,
            len(returns)
        ):

            corr_mat = get_correlation_matrix(
                returns,
                window,
                i
            )

            adj_matrix = threshold_correlation(
                corr_mat,
                threshold=0.6,
                use_absolute=True
            )

            G = adjacency_to_graph(
                adj_matrix,
                tickers,
                date=returns.index[i],
                corr_values=corr_mat
            )

            graph_path = os.path.join(
                GRAPHS_DIR,
                f"graph_{i - window:04d}.pkl"
            )

            with open(
                graph_path,
                "wb"
            ) as file:

                pickle.dump(
                    G,
                    file
                )

            total_graphs += 1

        print(
            f"Created and saved {total_graphs} graphs."
        )

        print(
            "Graphs saved in:",
            GRAPHS_DIR
        )

    # --------------------------------------------------
    # Generate All Graphs
    # --------------------------------------------------

    create_every_graphs()

    # --------------------------------------------------
    # Test Loading Saved Graph
    # --------------------------------------------------

    print("\n===== Saved Graph Test =====")

    test_graph = load_graph_by_index(0)

    print(
        "Loaded graph date:",
        test_graph.graph["date"]
    )

    print(
        "Loaded graph nodes:",
        test_graph.number_of_nodes()
    )

    print(
        "Loaded graph edges:",
        test_graph.number_of_edges()
    )