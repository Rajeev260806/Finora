import pandas as pd
import networkx as nx
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def create_network(correlation, threshold=0.60):

    graph = nx.Graph()

    # Add all stocks as nodes
    for ticker in correlation.columns:
        graph.add_node(ticker)

    # Add edges for highly correlated stocks
    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):

            stock1 = correlation.columns[i]
            stock2 = correlation.columns[j]

            corr_value = correlation.iloc[i, j]

            if corr_value >= threshold:
                graph.add_edge(
                    stock1,
                    stock2,
                    weight=corr_value
                )

    return graph


if __name__ == "__main__":

    correlation = pd.read_csv(
        "correlation_matrix.csv",
        index_col=0
    )

    graph = create_network(correlation, threshold=0.60)

    print("\nCorrelation Network")
    print("-------------------")
    print("Number of stocks:", graph.number_of_nodes())
    print("Number of connections:", graph.number_of_edges())

    print("\nConnections:")

    for stock1, stock2, data in graph.edges(data=True):
        print(
            f"{stock1} -- {stock2} : "
            f"{data['weight']:.2f}"
        )

    plt.figure(figsize=(12, 9))

    pos = nx.spring_layout(
        graph,
        seed=42
    )

    nx.draw_networkx(
        graph,
        pos,
        with_labels=True,
        node_size=1200,
        font_size=9
    )

    plt.title(
        "Stock Correlation Network "
        "(Correlation ≥ 0.60)"
    )

    plt.tight_layout()

    plt.savefig(
        "correlation_network.png",
        dpi=150
    )

    print("\nSaved:")
    print("correlation_network.png")