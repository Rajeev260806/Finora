"""
Proxy Task Training and Validation for GNN Stock Embeddings
===========================================================
Addresses Phase 1.4 of PROJECT_ACTION_PLAN.md:
- Standardizes targets cross-sectionally per day: z_i = (r_i - mean(r)) / std(r)
- End-to-end joint training of GNN backbone and linear predictor
- Reports Spearman Rank IC (Information Coefficient), IC IR, and R^2 on Validation set
- Trains strictly on TRAIN period (2015 to 2021) without leakage
- Saves trained models and generates embeddings for RL
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import r2_score, mean_squared_error
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")
GNN_DATA_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
CSV_PATH = os.path.join(BASE_DIR, "graphs", "graph_features.csv")
MODEL_DIR = os.path.join(BASE_DIR, "src", "gnn", "models")
EMBEDDING_GCN_DIR = os.path.join(BASE_DIR, "src", "gnn", "embeddings_gcn")
EMBEDDING_GAT_DIR = os.path.join(BASE_DIR, "src", "gnn", "embeddings_gat")
EMBEDDING_RANDOM_DIR = os.path.join(BASE_DIR, "src", "gnn", "embeddings_random")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EMBEDDING_GCN_DIR, exist_ok=True)
os.makedirs(EMBEDDING_GAT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_RANDOM_DIR, exist_ok=True)

INPUT_FEATURES = 5
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32
DROPOUT_RATE = 0.2
GAT_HEADS = 4


# -------------------------------------------------------------
# Models with Joint Prediction Head
# -------------------------------------------------------------

class GCNProxyModel(nn.Module):
    """2-layer GCN with joint prediction head"""
    def __init__(self, in_features=INPUT_FEATURES, hidden=HIDDEN_FEATURES, out_features=OUTPUT_FEATURES, dropout=DROPOUT_RATE):
        super().__init__()
        self.conv1 = GCNConv(in_features, hidden)
        self.conv2 = GCNConv(hidden, out_features)
        self.dropout = dropout
        self.predictor = nn.Linear(out_features, 1)

    def get_embeddings(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x

    def forward(self, x, edge_index, edge_weight):
        emb = self.get_embeddings(x, edge_index, edge_weight)
        preds = self.predictor(emb).squeeze(-1)
        return preds, emb


class GATProxyModel(nn.Module):
    """2-layer GAT with joint prediction head"""
    def __init__(self, in_features=INPUT_FEATURES, hidden=HIDDEN_FEATURES, out_features=OUTPUT_FEATURES, heads=GAT_HEADS, dropout=DROPOUT_RATE):
        super().__init__()
        gat_hidden = hidden // heads
        self.conv1 = GATConv(in_features, gat_hidden, heads=heads, dropout=dropout, concat=True)
        self.conv2 = GATConv(gat_hidden * heads, out_features, heads=1, dropout=dropout, concat=False)
        self.dropout = dropout
        self.predictor = nn.Linear(out_features, 1)

    def get_embeddings(self, x, edge_index, edge_weight):
        # GAT learns attention weights directly
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return x

    def forward(self, x, edge_index, edge_weight):
        emb = self.get_embeddings(x, edge_index, edge_weight)
        preds = self.predictor(emb).squeeze(-1)
        return preds, emb


# -------------------------------------------------------------
# In-Memory Graph Dataset for Fast Training
# -------------------------------------------------------------

class FinoraGraphDataset(Dataset):
    def __init__(self, graph_items):
        super().__init__()
        self.items = graph_items

    def len(self):
        return len(self.items)

    def get(self, idx):
        return self.items[idx]


def load_dataset_with_targets(data_path, gnn_data_dir, csv_path):
    """
    Loads all graphs, dates, and matches next-day cross-sectional standardized targets.
    """
    feat_df = pd.read_csv(csv_path)
    full_data = pd.read_parquet(data_path)
    wide_returns = full_data["Returns Daily"].unstack(level="Ticker").fillna(0.0)
    all_dates = wide_returns.index.tolist()
    date_to_loc = {str(d)[:10]: i for i, d in enumerate(all_dates)}

    graph_items = []
    skipped = 0

    for idx, row in feat_df.iterrows():
        d_str = row["date"]
        loc = date_to_loc.get(d_str, -1)
        if loc == -1 or loc >= len(all_dates) - 1:
            skipped += 1
            continue

        next_date = all_dates[loc + 1]
        next_ret_series = wide_returns.loc[next_date]

        npz_file = os.path.join(gnn_data_dir, f"graph_{idx:04d}.npz")
        if not os.path.exists(npz_file):
            continue

        d = np.load(npz_file)
        tickers = d["tickers"]
        next_ret = np.array([next_ret_series[t] for t in tickers], dtype=np.float32)

        # Cross-sectional z-score standardization
        mean_r = np.mean(next_ret)
        std_r = np.std(next_ret)
        if std_r > 1e-8:
            target_z = (next_ret - mean_r) / std_r
        else:
            target_z = np.zeros_like(next_ret)

        x = torch.tensor(d["node_features"], dtype=torch.float32)
        edge_index = torch.tensor(d["edge_index"], dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(d["edge_weight"], dtype=torch.float32)
        y = torch.tensor(target_z, dtype=torch.float32)
        raw_ret = torch.tensor(next_ret, dtype=torch.float32)

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            y=y,
            raw_ret=raw_ret,
            date=d_str,
            graph_idx=idx,
        )
        graph_items.append(data)

    print(f"Loaded {len(graph_items)} graph samples with valid next-day targets.")
    return graph_items


def evaluate_proxy(model, loader, device):
    """
    Evaluates proxy model on validation set:
    Computes MSE, R^2, and daily Spearman Rank Information Coefficient (IC).
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_raw_returns = []
    daily_ics = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds, _ = model(batch.x, batch.edge_index, batch.edge_weight)

            # Separate by graph within batch
            num_graphs = batch.num_graphs
            num_nodes = batch.x.size(0) // num_graphs

            preds_matrix = preds.view(num_graphs, num_nodes).cpu().numpy()
            targets_matrix = batch.y.view(num_graphs, num_nodes).cpu().numpy()
            raw_matrix = batch.raw_ret.view(num_graphs, num_nodes).cpu().numpy()

            for i in range(num_graphs):
                p_i = preds_matrix[i]
                t_i = targets_matrix[i]
                raw_i = raw_matrix[i]

                all_preds.extend(p_i)
                all_targets.extend(t_i)
                all_raw_returns.extend(raw_i)

                # Spearman Rank IC against actual next-day returns
                if np.std(p_i) > 1e-8 and np.std(raw_i) > 1e-8:
                    corr, _ = spearmanr(p_i, raw_i)
                    if np.isfinite(corr):
                        daily_ics.append(corr)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    mse = float(mean_squared_error(all_targets, all_preds))
    r2 = float(r2_score(all_targets, all_preds))
    mean_ic = float(np.mean(daily_ics)) if daily_ics else 0.0
    std_ic = float(np.std(daily_ics, ddof=1)) if len(daily_ics) > 1 else 1.0
    ic_ir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    pct_positive_ic = float(np.mean(np.array(daily_ics) > 0.0)) * 100.0 if daily_ics else 0.0

    return {
        "mse": mse,
        "r2": r2,
        "mean_ic": mean_ic,
        "std_ic": std_ic,
        "ic_ir": ic_ir,
        "pct_positive_ic": pct_positive_ic,
        "n_days": len(daily_ics)
    }


def train_model(model_type, train_items, val_items, epochs=40, lr=0.001, batch_size=32, device="cpu"):
    """
    Trains GCN or GAT proxy model.
    """
    if model_type == "GCN":
        model = GCNProxyModel().to(device)
    elif model_type == "GAT":
        model = GATProxyModel().to(device)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_dataset = FinoraGraphDataset(train_items)
    val_dataset = FinoraGraphDataset(val_items)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.MSELoss()

    best_val_ic = -float("inf")
    best_state = None
    best_metrics = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            preds, _ = model(batch.x, batch.edge_index, batch.edge_weight)
            loss = criterion(preds, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * batch.num_graphs

        train_loss /= len(train_items)

        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_metrics = evaluate_proxy(model, val_loader, device)
            # Track best model by Mean Rank IC
            if val_metrics["mean_ic"] > best_val_ic:
                best_val_ic = val_metrics["mean_ic"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                best_metrics = val_metrics

            print(f"  Epoch {epoch+1:02d}/{epochs:02d} | Train MSE: {train_loss:.4f} | Val MSE: {val_metrics['mse']:.4f} | Val R^2: {val_metrics['r2']:.4f} | Mean Rank IC: {val_metrics['mean_ic']:+.4f} (IR: {val_metrics['ic_ir']:+.3f}, {val_metrics['pct_positive_ic']:.1f}% pos)")

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_metrics


def generate_all_embeddings(model, all_items, output_dir, device="cpu"):
    """
    Generates and saves node embeddings for all graphs.
    """
    model.eval()
    os.makedirs(output_dir, exist_ok=True)
    loader = DataLoader(FinoraGraphDataset(all_items), batch_size=32, shuffle=False)

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            if hasattr(model, "get_embeddings"):
                emb = model.get_embeddings(batch.x, batch.edge_index, batch.edge_weight)
            else:
                # Random model
                emb = model(batch.x, batch.edge_index, batch.edge_weight)

            num_graphs = batch.num_graphs
            num_nodes = batch.x.size(0) // num_graphs
            emb_matrix = emb.view(num_graphs, num_nodes, -1).cpu().numpy()

            for i in range(num_graphs):
                g_idx = batch.graph_idx[i].item()
                out_path = os.path.join(output_dir, f"embeddings_graph_{g_idx:04d}.npy")
                np.save(out_path, emb_matrix[i].astype(np.float32))

    print(f"  Saved {len(all_items)} embeddings to: {output_dir}")


def main():
    print("=" * 65)
    print("  FINORA PHASE 1.4: PROXY TASK TRAINING & IC EVALUATION")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_items = load_dataset_with_targets(DATA_PATH, GNN_DATA_DIR, CSV_PATH)

    # Split:
    # Train: dates <= 2021-12-31
    # Val:   2022-01-01 <= dates <= 2022-12-28
    # Test:  dates >= 2022-12-29
    train_items = [item for item in all_items if item.date <= "2021-12-31"]
    val_items = [item for item in all_items if "2022-01-01" <= item.date <= "2022-12-28"]
    test_items = [item for item in all_items if item.date >= "2022-12-29"]

    print(f"Train graphs: {len(train_items)} (2015 to 2021)")
    print(f"Val graphs:   {len(val_items)} (2022)")
    print(f"Test graphs:  {len(test_items)} (2023 to 2024 - holdout)")

    # 1. Train GCN Proxy Model
    print("\n--- Training GCN Proxy Model ---")
    gcn_model, gcn_val_metrics = train_model("GCN", train_items, val_items, epochs=35, lr=0.002, device=device)
    gcn_path = os.path.join(MODEL_DIR, "gcn_proxy_model.pth")
    torch.save(gcn_model.state_dict(), gcn_path)
    print(f"[OK] Saved GCN Proxy Model to: {gcn_path}")
    print(f"GCN Validation Performance: Mean Rank IC = {gcn_val_metrics['mean_ic']:+.4f}, IC IR = {gcn_val_metrics['ic_ir']:+.3f}, R^2 = {gcn_val_metrics['r2']:.4f}")

    # 2. Train GAT Proxy Model
    print("\n--- Training GAT Proxy Model ---")
    gat_model, gat_val_metrics = train_model("GAT", train_items, val_items, epochs=35, lr=0.001, device=device)
    gat_path = os.path.join(MODEL_DIR, "gat_proxy_model.pth")
    torch.save(gat_model.state_dict(), gat_path)
    print(f"[OK] Saved GAT Proxy Model to: {gat_path}")
    print(f"GAT Validation Performance: Mean Rank IC = {gat_val_metrics['mean_ic']:+.4f}, IC IR = {gat_val_metrics['ic_ir']:+.3f}, R^2 = {gat_val_metrics['r2']:.4f}")

    # 3. Random Baseline (Frozen Untrained GAT) for ablation
    print("\n--- Initializing Random Embeddings (Frozen Untrained Model) ---")
    torch.manual_seed(42)
    random_model = GATProxyModel().to(device)
    random_model.eval()

    # 4. Generate Embeddings across all snapshots
    print("\n--- Generating Embeddings for RL Training & Evaluation ---")
    print("1. Generating GCN Embeddings...")
    generate_all_embeddings(gcn_model, all_items, EMBEDDING_GCN_DIR, device)
    
    print("2. Generating GAT Embeddings...")
    generate_all_embeddings(gat_model, all_items, EMBEDDING_GAT_DIR, device)
    
    print("3. Generating Random Embeddings (Ablation)...")
    generate_all_embeddings(random_model, all_items, EMBEDDING_RANDOM_DIR, device)

    # Also update src/gnn/embeddings to point to GAT embeddings by default
    default_emb_dir = os.path.join(BASE_DIR, "src", "gnn", "embeddings")
    generate_all_embeddings(gat_model, all_items, default_emb_dir, device)

    print("\n" + "=" * 65)
    print("  SUMMARY: PROXY TASK EVALUATION (VALIDATION SET 2022)")
    print("=" * 65)
    print(f"| Model | Mean Rank IC | IC Std | IC IR | % Positive IC Days | R^2 Score | Val MSE |")
    print(f"|-------|--------------|--------|-------|--------------------|-----------|---------|")
    print(f"| GCN   | {gcn_val_metrics['mean_ic']:+.4f}       | {gcn_val_metrics['std_ic']:.4f} | {gcn_val_metrics['ic_ir']:+.3f} | {gcn_val_metrics['pct_positive_ic']:.1f}%              | {gcn_val_metrics['r2']:.4f}    | {gcn_val_metrics['mse']:.4f}  |")
    print(f"| GAT   | {gat_val_metrics['mean_ic']:+.4f}       | {gat_val_metrics['std_ic']:.4f} | {gat_val_metrics['ic_ir']:+.3f} | {gat_val_metrics['pct_positive_ic']:.1f}%              | {gat_val_metrics['r2']:.4f}    | {gat_val_metrics['mse']:.4f}  |")
    print("=" * 65)


if __name__ == "__main__":
    main()
