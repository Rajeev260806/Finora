import os
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

GRAPHS_DIR = os.path.join(
    BASE_DIR,
    "graphs",
    "saved_graphs"
)

PLOTS_DIR = os.path.join(
    BASE_DIR,
    "results",
    "plots"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "graphs",
    "graph_features.csv"
)

os.makedirs(PLOTS_DIR, exist_ok=True)


# --------------------------------------------------
# Calculate Graph Features
# --------------------------------------------------

def calculate_features(G):

    nodes = G.number_of_nodes()
    edges = G.number_of_edges()

    # Average degree
    if nodes > 0:
        average_degree = (2 * edges) / nodes
    else:
        average_degree = 0

    # Graph density
    density = nx.density(G)

    # Connected components
    components = nx.number_connected_components(G)

    # Largest connected component
    if nodes > 0:
        largest_component = max(
            len(component)
            for component in nx.connected_components(G)
        )
    else:
        largest_component = 0

    # Average clustering coefficient
    clustering = nx.average_clustering(G)

    # Sector homophily
    if edges > 0:
        same_sector_edges = sum(
            1
            for u, v in G.edges()
            if G.nodes[u].get("sector") ==
               G.nodes[v].get("sector")
        )

        sector_homophily = (
            same_sector_edges / edges
        )
    else:
        sector_homophily = 0.0

    return {
        "date": G.graph["date"],
        "nodes": nodes,
        "edges": edges,
        "average_degree": average_degree,
        "density": density,
        "connected_components": components,
        "largest_component": largest_component,
        "clustering_coefficient": clustering,
        "sector_homophily": sector_homophily
    }


# --------------------------------------------------
# Load All Graphs
# --------------------------------------------------

def analyze_all_graphs():

    graph_files = sorted(
        file
        for file in os.listdir(GRAPHS_DIR)
        if file.endswith(".pkl")
    )

    print("=" * 60)
    print("GRAPH FEATURE ANALYSIS")
    print("=" * 60)

    print("Total graphs:", len(graph_files))

    results = []

    for count, filename in enumerate(graph_files, start=1):

        path = os.path.join(
            GRAPHS_DIR,
            filename
        )

        with open(path, "rb") as file:
            G = pickle.load(file)

        features = calculate_features(G)

        results.append(features)

        if count % 100 == 0:
            print(
                f"Processed {count}/{len(graph_files)} graphs"
            )

    # --------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------

    df = pd.DataFrame(results)

    df = df.sort_values("date")

    df = df.reset_index(drop=True)

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n===== Analysis Completed =====")

    print("Rows:", len(df))

    print("Columns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nSummary:")
    print(df.describe())

    print("\nSaved to:")
    print(OUTPUT_FILE)

    return df


# --------------------------------------------------
# Plot Number of Edges
# --------------------------------------------------

def plot_edges(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["edges"]
    )

    plt.xlabel("Date")
    plt.ylabel("Number of Edges")
    plt.title("Number of Correlation Network Edges Over Time")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "edges_over_time.png"
        ),
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# Plot Sector Homophily
# --------------------------------------------------

def plot_homophily(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["sector_homophily"]
    )

    plt.xlabel("Date")
    plt.ylabel("Sector Homophily")
    plt.title("Sector Homophily Over Time")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "sector_homophily_over_time.png"
        ),
        dpi=300
    )

    plt.close()


# --------------------------------------------------
# Plot Clustering Coefficient
# --------------------------------------------------

def plot_clustering(df):

    plt.figure(figsize=(12, 6))

    plt.plot(
        df["date"],
        df["clustering_coefficient"]
    )

    plt.xlabel("Date")
    plt.ylabel("Clustering Coefficient")
    plt.title("Network Clustering Coefficient Over Time")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "clustering_over_time.png"
        ),
        dpi=300
    )

    plt.close()


# ==================================================
# MAIN
# ==================================================

if __name__ == "__main__":

    df = analyze_all_graphs()

    plot_edges(df)

    plot_homophily(df)

    plot_clustering(df)

    print("\nPlots saved in:")
    print(PLOTS_DIR)