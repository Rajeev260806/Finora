import pandas as pd
import networkx as nx


def create_network(correlation, threshold=0.60):

    graph = nx.Graph()

    for ticker in correlation.columns:
        graph.add_node(ticker)

    for i in range(len(correlation.columns)):
        for j in range(i + 1, len(correlation.columns)):

            stock1 = correlation.columns[i]
            stock2 = correlation.columns[j]

            corr_value = correlation.iloc[i, j]

            if corr_value >= threshold:
                graph.add_edge(stock1, stock2, weight=corr_value)

    return graph


if __name__ == "__main__":

    correlation = pd.read_csv(
        "correlation_matrix.csv",
        index_col=0
    )

    graph = create_network(correlation)

    clusters = list(nx.connected_components(graph))

    print("\nStock Clusters")
    print("-------------")

    for i, cluster in enumerate(clusters, 1):
        print(f"Cluster {i}: {sorted(cluster)}")

    print("\nNumber of clusters:", len(clusters))