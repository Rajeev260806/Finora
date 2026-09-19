import os

import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CORRELATION_FILE = "results/csv/correlation_matrix.csv"
OUTPUT_FILE = "results/graphs/correlation_network.png"
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

    print("\n" + "=" * 50)
    print("CORRELATION NETWORK")
    print("=" * 50)

    print(f"\nCorrelation threshold: {THRESHOLD}")
    print("Number of stocks:", graph.number_of_nodes())
    print("Number of connections:", graph.number_of_edges())

    print("\nConnections")
    print("-" * 30)

    for stock1, stock2, data in graph.edges(data=True):
        print(
            f"{stock1} -- {stock2} : "
            f"{data['weight']:.2f}"
        )

    print("\nCreating graph layout...")

    pos = nx.spring_layout(
        graph,
        seed=42,
        iterations=50
    )

    sector_colors = {
        "Technology": "tab:blue",
        "Finance": "tab:green",
        "Utilities": "tab:orange",
        "Real Estate": "tab:red",
        "Energy": "tab:purple"
    }

    node_colors = [
        sector_colors.get(
            graph.nodes[node]["sector"],
            "gray"
        )
        for node in graph.nodes()
    ]

    plt.figure(figsize=(12, 9))

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=1200,
        node_color=node_colors
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        width=1.5,
        alpha=0.7
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9
    )

    plt.title(
        "Stock Correlation Network "
        "(Correlation ≥ 0.60)"
    )

    plt.axis("off")
    plt.tight_layout()

    os.makedirs(
        "results/graphs",
        exist_ok=True
    )

    plt.savefig(
        OUTPUT_FILE,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print("\nSaved:")
    print(OUTPUT_FILE)

    print("\n" + "=" * 50)
    print("Network analysis completed.")
    print("=" * 50)