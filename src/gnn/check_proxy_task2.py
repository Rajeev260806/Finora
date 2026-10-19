import os
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from temporal_graph_dataset import TemporalGraphDataset
import sys
sys.path.append('.')
from improved_gnn_model import ImprovedStockGNN

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
EMBEDDING_DIR = os.path.join(BASE_DIR, "src", "gnn", "embeddings")
RETURNS_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")

# Load returns data
returns_df = pd.read_parquet(RETURNS_DATA_PATH)
returns_wide = returns_df["Returns Daily"].unstack(level="Ticker")
returns_wide = returns_wide.fillna(0.0)
returns_wide = returns_wide.sort_index(axis=1)
tickers = returns_wide.columns.tolist()

# Load dataset to get graph dates and sector mappings
dataset = TemporalGraphDataset(GRAPH_DIR)
print(f"Number of graphs: {len(dataset)}")

# Define the model parameters (must match the ones used to generate embeddings)
INPUT_FEATURES = 9
HIDDEN_FEATURES = 64
OUTPUT_FEATURES = 32
DROPOUT_RATE = 0.2
GAT_HEADS = 4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Define GNN model for proxy task
class GNNForProxyTask(torch.nn.Module):
    def __init__(self, input_features, hidden_features, output_features, dropout_rate=0.2, gat_heads=4):
        super().__init__()
        self.gnn = ImprovedStockGNN(
            input_features=input_features,
            hidden_features=hidden_features,
            output_features=output_features,
            dropout_rate=dropout_rate,
            gat_heads=gat_heads,
            use_gat=True
        )
        self.predictor = torch.nn.Linear(output_features, 1)

    def forward(self, x, edge_index, edge_weight):
        embeddings = self.gnn(x, edge_index, edge_weight)
        out = self.predictor(embeddings)
        return out.squeeze(-1)

model = GNNForProxyTask(
    input_features=INPUT_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    output_features=OUTPUT_FEATURES,
    dropout_rate=DROPOUT_RATE,
    gat_heads=GAT_HEADS
).to(device)

print(f"Model architecture:\n{model}")

# Loss and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# We'll create a DataLoader for training
from torch_geometric.loader import DataLoader

# We'll split the dataset into train and validation (80-20)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    follow_batch=['x']
)
val_loader = DataLoader(
    val_dataset,
    batch_size=16,
    shuffle=False,
    follow_batch=['x']
)

# Training loop
num_epochs = 20
best_val_r2 = -float('inf')

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    train_preds = []
    train_targets = []

    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        # Forward pass
        out = model(batch.x, batch.edge_index, batch.edge_weight)

        # Get target returns (next-day returns)
        node_graph_indices = batch.batch  # shape: [total_nodes_in_batch]
        unique_graph_indices = torch.unique(node_graph_indices)

        batch_targets = []
        for graph_idx in unique_graph_indices:
            graph_idx_item = graph_idx.item()
            graph_data = dataset[graph_idx_item]
            date_str = graph_data.date_str[0]
            if date_str in returns_wide.index:
                date_returns = returns_wide.loc[date_str]
                tickers_graph = graph_data.tickers
                next_returns = date_returns[tickers_graph].values
            else:
                next_returns = np.zeros(len(tickers_graph))
            batch_targets.append(next_returns)

        target_returns = np.concatenate(batch_targets, axis=0)
        target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)

        loss = criterion(out, target_tensor)
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        train_preds.append(out.detach().cpu().numpy())
        train_targets.append(target_tensor.cpu().numpy())

    # Compute training R2
    train_preds = np.concatenate(train_preds, axis=0)
    train_targets = np.concatenate(train_targets, axis=0)
    train_r2 = r2_score(train_targets, train_preds)

    # Validation
    model.eval()
    val_losses = []
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_weight)

            node_graph_indices = batch.batch
            unique_graph_indices = torch.unique(node_graph_indices)

            batch_targets = []
            for graph_idx in unique_graph_indices:
                graph_idx_item = graph_idx.item()
                graph_data = dataset[graph_idx_item]
                date_str = graph_data.date_str[0]
                if date_str in returns_wide.index:
                    date_returns = returns_wide.loc[date_str]
                    tickers_graph = graph_data.tickers
                    next_returns = date_returns[tickers_graph].values
                else:
                    next_returns = np.zeros(len(tickers_graph))
                batch_targets.append(next_returns)

            target_returns = np.concatenate(batch_targets, axis=0)
            target_tensor = torch.tensor(target_returns, dtype=torch.float32).to(device)

            loss = criterion(out, target_tensor)
            val_losses.append(loss.item())
            val_preds.append(out.detach().cpu().numpy())
            val_targets.append(target_tensor.cpu().numpy())

    val_preds = np.concatenate(val_preds, axis=0)
    val_targets = np.concatenate(val_targets, axis=0)
    val_r2 = r2_score(val_targets, val_preds)
    val_loss = np.mean(val_losses) if val_losses else 0

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {np.mean(train_losses):.6f}, Train R2: {train_r2:.6f} | Val Loss: {val_loss:.6f}, Val R2: {val_r2:.6f}")

    # Save best model
    if val_r2 > best_val_r2:
        best_val_r2 = val_r2
        torch.save(model.state_dict(), os.path.join(BASE_DIR, "src", "gnn", "models", "best_proxy_gnn.pth"))
        print(f"  -> New best model saved with Val R2: {val_r2:.6f}")

print(f"\nBest validation R2: {best_val_r2:.6f}")

# Now, we can use the best model to generate embeddings for the RL agent.
# But note: the RL agent expects embeddings of dimension 32, which is the output of the GNN before the predictor.
# We can save the embeddings from the GNN part of the best model.

# Let's load the best model and generate embeddings for all graphs.
best_model = GNNForProxyTask(
    input_features=INPUT_FEATURES,
    hidden_features=HIDDEN_FEATURES,
    output_features=OUTPUT_FEATURES,
    dropout_rate=DROPOUT_RATE,
    gat_heads=GAT_HEADS
).to(device)
best_model.load_state_dict(torch.load(os.path.join(BASE_DIR, "src", "gnn", "models", "best_proxy_gnn.pth")))
best_model.eval()

# Directory to save embeddings for RL
rl_embedding_dir = os.path.join(BASE_DIR, "src", "gnn", "embeddings_rl")
os.makedirs(rl_embedding_dir, exist_ok=True)

print("\nGenerating embeddings for RL agent from best proxy model...")
with torch.no_grad():
    for idx in range(len(dataset)):
        graph_data = dataset[idx]
        graph_data = graph_data.to(device)
        # Get embeddings from the GNN part
        embeddings = best_model.gnn(graph_data.x, graph_data.edge_index, graph_data.edge_weight)
        embeddings_np = embeddings.cpu().numpy()
        embedding_filename = f"embeddings_rl_{idx:04d}.npy"
        embedding_path = os.path.join(rl_embedding_dir, embedding_filename)
        np.save(embedding_path, embeddings_np)
        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} embeddings")

print(f"Embeddings saved to: {rl_embedding_dir}")

# Now, we can evaluate the proxy task performance of the best model (already done above)
print(f"\nBest validation R2 from proxy task: {best_val_r2:.6f}")