"""
Build All Rolling Graphs for Finora
==================================
Constructs 60-day rolling correlation graphs for the entire dataset (2015 to 2024).
Crucially ensures NO DATA LEAKAGE:
- Graph at date t uses ONLY historical returns in [t-60, t).
- Node feature scaler is fitted strictly on the TRAIN period (dates <= 2021-12-31).
- Generates all graph snapshots so validation and test periods have full rolling coverage.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from sklearn.preprocessing import StandardScaler

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")
GRAPHS_DIR = os.path.join(BASE_DIR, "graphs", "saved_graphs")
GNN_DATA_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
CSV_PATH = os.path.join(BASE_DIR, "graphs", "graph_features.csv")
SCALER_PATH = os.path.join(BASE_DIR, "graphs", "scaler_train.pkl")

os.makedirs(GRAPHS_DIR, exist_ok=True)
os.makedirs(GNN_DATA_DIR, exist_ok=True)

WINDOW = 60
CORR_THRESHOLD = 0.6

NODE_FEATURES = [
    "Returns Daily",
    "Volatility 20D",
    "Volume Normalized",
    "Volatility Annual",
    "Log Return"
]

TICKER_TO_SECTOR = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", "ADBE": "Technology",
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance", "V": "Finance",
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities", "AEP": "Utilities",
    "PLD": "Real Estate", "AMT": "Real Estate", "O": "Real Estate", "SPG": "Real Estate"
}

SECTOR_TO_ID = {
    "Technology": 0,
    "Finance": 1,
    "Energy": 2,
    "Utilities": 3,
    "Real Estate": 4,
}


def sector_homophily(G: nx.Graph) -> float:
    same_sector_edges = 0
    total_edges = G.number_of_edges()
    if total_edges == 0:
        return 0.0
    for u, v in G.edges():
        if G.nodes[u].get("sector") == G.nodes[v].get("sector"):
            same_sector_edges += 1
    return same_sector_edges / total_edges


def main():
    print("=" * 65)
    print("  BUILDING ROLLING CORRELATION GRAPHS (NO DATA LEAKAGE)")
    print("=" * 65)

    data = pd.read_parquet(DATA_PATH)
    data = data.sort_index()
    tickers = sorted(data.index.get_level_values("Ticker").unique().tolist())
    n_assets = len(tickers)

    wide_returns = data["Returns Daily"].unstack(level="Ticker")[tickers].fillna(0.0)
    all_dates = wide_returns.index.tolist()
    total_dates = len(all_dates)

    print(f"Total dates in dataset: {total_dates} (Tickers: {n_assets})")
    print(f"Rolling window: {WINDOW} days, Threshold: {CORR_THRESHOLD}")

    # 1. Fit scaler on TRAIN PERIOD ONLY (2015-01-02 to 2021-12-31)
    train_dates = [d for d in all_dates if str(d)[:10] <= "2021-12-31"]
    train_sub = data.loc[data.index.get_level_values("Date").isin(train_dates)]
    raw_train_features = train_sub[NODE_FEATURES].fillna(0.0).values

    scaler = StandardScaler()
    scaler.fit(raw_train_features)
    with open(SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Fitted StandardScaler strictly on train period ({len(train_dates)} dates, {len(train_sub)} rows)")
    print(f"Saved train scaler to: {SCALER_PATH}")

    # 2. Iterate through all dates starting at WINDOW
    graph_records = []
    total_graphs = total_dates - WINDOW
    print(f"\nConstructing {total_graphs} rolling graph snapshots...")

    for idx, date_idx in enumerate(range(WINDOW, total_dates)):
        curr_date = all_dates[date_idx]
        date_str = str(curr_date)[:10]

        # Historical window strictly in [date_idx - WINDOW, date_idx)
        window_ret = wide_returns.iloc[date_idx - WINDOW: date_idx]
        corr_matrix = window_ret.corr().fillna(0.0).values  # (n_assets, n_assets)

        # Adjacency thresholding
        abs_corr = np.abs(corr_matrix)
        np.fill_diagonal(abs_corr, 0.0)

        G = nx.Graph(date=curr_date)
        for t in tickers:
            sec = TICKER_TO_SECTOR.get(t, "Unknown")
            sec_id = SECTOR_TO_ID.get(sec, 0)
            G.add_node(t, sector=sec, sector_id=sec_id)

        edge_index = []
        edge_weight = []

        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                if abs_corr[i, j] >= CORR_THRESHOLD:
                    w = float(corr_matrix[i, j])
                    G.add_edge(tickers[i], tickers[j], weight=abs(w), correlation=w)
                    # Undirected: forward and reverse
                    edge_index.append([i, j])
                    edge_index.append([j, i])
                    edge_weight.append(w)
                    edge_weight.append(w)

        # Graph stats for CSV
        n_edges = G.number_of_edges()
        density = nx.density(G)
        avg_degree = 2.0 * n_edges / n_assets if n_assets > 0 else 0.0
        n_components = nx.number_connected_components(G)
        largest_comp = max(len(c) for c in nx.connected_components(G)) if n_edges > 0 else 1
        clustering = nx.average_clustering(G)
        homophily = sector_homophily(G)

        graph_records.append({
            "date": date_str,
            "nodes": n_assets,
            "edges": n_edges,
            "average_degree": avg_degree,
            "density": density,
            "connected_components": n_components,
            "largest_component": largest_comp,
            "clustering_coefficient": clustering,
            "sector_homophily": homophily,
        })

        # Save NetworkX pickle
        pkl_path = os.path.join(GRAPHS_DIR, f"graph_{idx:04d}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(G, f)

        # Node features for curr_date normalized using train scaler
        node_features_raw = []
        sector_ids = []
        for t in tickers:
            try:
                row = data.loc[(curr_date, t)]
                feat = row[NODE_FEATURES].astype(float).values
            except KeyError:
                feat = np.zeros(len(NODE_FEATURES), dtype=float)
            node_features_raw.append(feat)
            sector_ids.append(SECTOR_TO_ID.get(TICKER_TO_SECTOR.get(t, "Unknown"), 0))

        node_features_raw = np.array(node_features_raw, dtype=np.float32)
        node_features_scaled = scaler.transform(node_features_raw).astype(np.float32)

        edge_index_np = np.array(edge_index, dtype=np.int64) if edge_index else np.empty((0, 2), dtype=np.int64)
        edge_weight_np = np.array(edge_weight, dtype=np.float32) if edge_weight else np.empty((0,), dtype=np.float32)

        npz_path = os.path.join(GNN_DATA_DIR, f"graph_{idx:04d}.npz")
        np.savez_compressed(
            npz_path,
            node_features=node_features_scaled,
            edge_index=edge_index_np,
            edge_weight=edge_weight_np,
            sector_ids=np.array(sector_ids, dtype=np.int64),
            tickers=np.array(tickers),
            date=str(curr_date),
        )

        if (idx + 1) % 500 == 0 or idx == total_graphs - 1:
            print(f"  Processed {idx + 1}/{total_graphs} graphs (latest date: {date_str})")

    # Save graph_features.csv
    features_df = pd.DataFrame(graph_records)
    features_df.to_csv(CSV_PATH, index=False)
    print(f"\n[OK] Saved {len(features_df)} graph metadata entries to: {CSV_PATH}")
    print(f"Date range in features: {features_df['date'].min()} to {features_df['date'].max()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
