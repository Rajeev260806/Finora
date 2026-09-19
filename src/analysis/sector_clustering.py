import json
import os

import pandas as pd
import networkx as nx


CORRELATION_FILE = "results/csv/correlation_matrix.csv"
OUTPUT_CSV = "results/csv/sector_clustering.csv"
OUTPUT_JSON = "results/csv/sector_clustering_summary.json"

THRESHOLD = 0.60


SECTORS = {
    "AAPL": "Technology",
    "ADBE": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "V": "Finance",
    "AEP": "Utilities",
    "DUK": "Utilities",
    "NEE": "Utilities",
    "SO": "Utilities",
    "AMT": "Real Estate",
    "O": "Real Estate",
    "PLD": "Real Estate",
    "SPG": "Real Estate",
    "BAC": "Finance",
    "GS": "Finance",
    "JPM": "Finance",
    "COP": "Energy",
    "CVX": "Energy",
    "SLB": "Energy",
    "XOM": "Energy"
}


def create_network(correlation, threshold=THRESHOLD):
    graph = nx.Graph()

    for ticker in correlation.columns:
        graph.add_node(
            ticker,
            sector=SECTORS.get(ticker, "Unknown")
        )

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):
            stock1 = correlation.columns[i]
            stock2 = correlation.columns[j]

            corr_value = correlation.iloc[i, j]

            if corr_value >= threshold:
                graph.add_edge(
                    stock1,
                    stock2,
                    weight=float(corr_value)
                )

    return graph


def calculate_sector_homophily(graph):
    total_edges = graph.number_of_edges()

    if total_edges == 0:
        return 0.0

    same_sector_edges = 0

    for stock1, stock2 in graph.edges():
        if graph.nodes[stock1]["sector"] == graph.nodes[stock2]["sector"]:
            same_sector_edges += 1

    return same_sector_edges / total_edges


def save_cluster_results(clusters, graph):
    rows = []

    for cluster_id, cluster in enumerate(clusters, 1):
        for stock in sorted(cluster):
            rows.append({
                "cluster": cluster_id,
                "stock": stock,
                "sector": graph.nodes[stock]["sector"]
            })

    df = pd.DataFrame(rows)

    os.makedirs("results/csv", exist_ok=True)

    df.to_csv(
        OUTPUT_CSV,
        index=False
    )

    return df


if __name__ == "__main__":

    correlation = pd.read_csv(
        CORRELATION_FILE,
        index_col=0
    )

    correlation = correlation.loc[
        correlation.columns,
        correlation.columns
    ]

    graph = create_network(
        correlation,
        threshold=THRESHOLD
    )

    clusters = list(
        nx.connected_components(graph)
    )

    same_sector_edges = 0
    cross_sector_edges = 0

    for stock1, stock2 in graph.edges():
        sector1 = graph.nodes[stock1]["sector"]
        sector2 = graph.nodes[stock2]["sector"]

        if sector1 == sector2:
            same_sector_edges += 1
        else:
            cross_sector_edges += 1

    total_edges = graph.number_of_edges()

    homophily = calculate_sector_homophily(graph)

    cluster_df = save_cluster_results(
        clusters,
        graph
    )

    summary = {
        "correlation_threshold": THRESHOLD,
        "number_of_stocks": graph.number_of_nodes(),
        "number_of_edges": total_edges,
        "number_of_clusters": len(clusters),
        "same_sector_edges": same_sector_edges,
        "cross_sector_edges": cross_sector_edges,
        "sector_homophily": round(homophily, 4)
    }

    with open(OUTPUT_JSON, "w") as file:
        json.dump(
            summary,
            file,
            indent=4
        )

    print("\n" + "=" * 50)
    print("SECTOR CLUSTERING ANALYSIS")
    print("=" * 50)

    print(f"\nCorrelation threshold: {THRESHOLD}")
    print(f"Number of stocks: {graph.number_of_nodes()}")
    print(f"Number of edges: {total_edges}")

    print("\nStock Clusters")
    print("-" * 30)

    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {sorted(cluster)}")

    print(f"\nNumber of clusters: {len(clusters)}")

    print("\nSector Homophily")
    print("-" * 30)
    print(f"Same-sector edge ratio: {homophily:.4f}")
    print(f"Same-sector edges: {same_sector_edges}")
    print(f"Cross-sector edges: {cross_sector_edges}")

    print("\nFiles saved:")
    print(OUTPUT_CSV)
    print(OUTPUT_JSON)

    print("\n" + "=" * 50)
    print("Analysis completed.")
    print("=" * 50)