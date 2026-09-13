import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from collections import Counter

def get_correlation_matrix(returns_df, window, idx):   
    if idx < window:
        return None

    window_returns = returns_df.iloc[idx-window:idx]
    corr_matrix = window_returns.corr()
    corr_matrix = corr_matrix.fillna(0)

    return corr_matrix

def threshold_correlation(corr_matrix,threshold=0.4,use_absolute=True):
    if use_absolute:
        adj_mat = np.abs(corr_matrix.to_numpy()).copy()
    else:
        adj_mat = corr_matrix.to_numpy().copy()
    np.fill_diagonal(adj_mat,0)
    adj_mat[adj_mat<threshold] = 0
    adj_mat = pd.DataFrame(
    adj_mat,
    index=corr_matrix.index,
    columns=corr_matrix.columns)
    return adj_mat

def adjacency_to_graph(adj_matrix, tickers, date, corr_values=None):
    G = nx.Graph()

    ticker_to_sector = {
        'AAPL': 'Technology',
        'MSFT': 'Technology',
        'NVDA': 'Technology',
        'ADBE': 'Technology',

        'JPM': 'Finance',
        'BAC': 'Finance',
        'GS': 'Finance',
        'V': 'Finance',

        'XOM': 'Energy',
        'CVX': 'Energy',
        'COP': 'Energy',
        'SLB': 'Energy',

        'NEE': 'Utilities',
        'DUK': 'Utilities',
        'SO': 'Utilities',
        'AEP': 'Utilities',

        'PLD': 'Real Estate',
        'AMT': 'Real Estate',
        'O': 'Real Estate',
        'SPG': 'Real Estate'
    }

    for i in tickers:
        sector = ticker_to_sector.get(i, "Unknown")
        G.add_node(i, sector=sector)
    
    for i in range(len(tickers)):
        for j in range(i+1,len(tickers)):
            weight = adj_matrix.iloc[i,j]
            if weight>0:
                if corr_values is not None:
                    correlation = corr_values.iloc[i,j]
                else:
                    correlation = weight
                G.add_edge(tickers[i],tickers[j],weight=weight,correlation=correlation)
    G.graph["date"] = date
    return G

def sector_homophily(G):
    same = sum(
        1 for u, v in G.edges()
        if G.nodes[u]["sector"] == G.nodes[v]["sector"]
    )
    return same / G.number_of_edges() if G.number_of_edges() else 0.0

def load_graph_by_index(idx: int, base_dir: str = "saved_graphs") -> nx.Graph:
    fname = f"graph_{idx:04d}.pkl"
    path = os.path.join(base_dir, fname)

    with open(path, "rb") as file:
        return pickle.load(file)

if __name__ == "__main__":

    data = pd.read_parquet("data\\clean\\train.parquet")
    returns = data["Log Return"].unstack(level="Ticker")
    window = 60

    corr_mat = get_correlation_matrix(returns,window,60)
    print(corr_mat)
    adj_matrix=threshold_correlation(corr_mat)
    print(adj_matrix)

    tickers = returns.columns.tolist()

    G = adjacency_to_graph(
        adj_matrix,
        tickers,
        date=returns.index[60],
        corr_values=corr_mat
    )

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    GRAPHS_DIR = os.path.join(BASE_DIR, "saved_graphs")
    os.makedirs(GRAPHS_DIR, exist_ok=True)

    def create_every_graphs():
        for i in range(60,len(returns)):
            corr_mat = get_correlation_matrix(returns,window,i)
            adj_matrix=threshold_correlation(corr_mat)
            G = adjacency_to_graph(
            adj_matrix,
            tickers,
            date=returns.index[i],
            corr_values=corr_mat
            )
            graph_path = os.path.join(
                GRAPHS_DIR,
                f"graph_{i-60:04d}.pkl"
            )
            with open(graph_path,"wb") as file:
                pickle.dump(G,file)

    print("\nSector homophily:", sector_homophily(G))

    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    print("\nNodes:")
    print(G.nodes(data=True))

    print("\nEdges:")
    print(G.edges(data=True))

    print("\nGraph attributes:")
    print(G.graph)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PLOTS_DIR = os.path.join(BASE_DIR, "plots")
    os.makedirs(PLOTS_DIR, exist_ok=True)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        corr_mat,
        annot=False,
        cmap="coolwarm",
        center=0
    )
    date = G.graph["date"]
    plt.title(f"Correlation Matrix - {date}")
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, f"corr_heatmap_{date.date()}.png"),
        dpi=300
    )

    plt.close()

    plt.figure(figsize=(12, 10))
    pos = nx.spring_layout(G, seed=42)

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
        labels={ticker: ticker for ticker in G.nodes},
        font_size=10
    )

    plt.title(f"Graph on {date} - Nodes Colored by Sector")
    plt.tight_layout()

    plt.savefig(
        os.path.join(PLOTS_DIR, f"graph_diagram_{date.date()}.png"),
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\n===== Graph Validation =====")
    print("Date:", date)
    print("Number of nodes:", G.number_of_nodes())
    print("Number of edges:", G.number_of_edges())

    print("\nNode attributes:")
    for node, attributes in G.nodes(data=True):
        print(node, attributes)

    print("\nEdge attributes:")
    for u, v, attributes in G.edges(data=True):
        print(u, "-", v, attributes)

    create_every_graphs()

    test_graph = load_graph_by_index(62)

    print("Loaded graph date:", test_graph.graph["date"])
    print("Loaded graph nodes:", test_graph.number_of_nodes())
    print("Loaded graph edges:", test_graph.number_of_edges())