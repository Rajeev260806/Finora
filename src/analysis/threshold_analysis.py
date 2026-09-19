import os

import pandas as pd
import networkx as nx


CORRELATION_FILE = "results/csv/correlation_matrix.csv"
OUTPUT_FILE = "results/csv/threshold_analysis.csv"

THRESHOLDS = [0.50, 0.60, 0.70]

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


def create_network(correlation, threshold):
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


def calculate_homophily(graph):
    total_edges = graph.number_of_edges()

    if total_edges == 0:
        return 0.0, 0, 0

    same_sector_edges = 0
    cross_sector_edges = 0

    for stock1, stock2 in graph.edges():

        sector1 = graph.nodes[stock1]["sector"]
        sector2 = graph.nodes[stock2]["sector"]

        if sector1 == sector2:
            same_sector_edges += 1
        else:
            cross_sector_edges += 1

    homophily = same_sector_edges / total_edges

    return (
        homophily,
        same_sector_edges,
        cross_sector_edges
    )


if __name__ == "__main__":

    correlation = pd.read_csv(
        CORRELATION_FILE,
        index_col=0
    )

    correlation = correlation.loc[
        correlation.columns,
        correlation.columns
    ]

    results = []

    print("\n" + "=" * 60)
    print("CORRELATION THRESHOLD ANALYSIS")
    print("=" * 60)

    for threshold in THRESHOLDS:

        graph = create_network(
            correlation,
            threshold
        )

        clusters = list(
            nx.connected_components(graph)
        )

        (
            homophily,
            same_sector_edges,
            cross_sector_edges
        ) = calculate_homophily(graph)

        result = {
            "threshold": threshold,
            "stocks": graph.number_of_nodes(),
            "edges": graph.number_of_edges(),
            "clusters": len(clusters),
            "same_sector_edges": same_sector_edges,
            "cross_sector_edges": cross_sector_edges,
            "homophily": round(homophily, 4)
        }

        results.append(result)

        print(f"\nThreshold: {threshold}")
        print(f"Edges: {graph.number_of_edges()}")
        print(f"Clusters: {len(clusters)}")
        print(f"Same-sector edges: {same_sector_edges}")
        print(f"Cross-sector edges: {cross_sector_edges}")
        print(f"Homophily: {homophily:.4f}")

    df = pd.DataFrame(results)

    os.makedirs(
        "results/csv",
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print(df.to_string(index=False))

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 60)
    print("Threshold analysis completed.")
    print("=" * 60)