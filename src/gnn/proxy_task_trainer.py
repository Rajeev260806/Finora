import os
import sys
# Add the project root to sys.path to allow imports like src.gnn.temporal_graph_dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Tuple, List

# Import our temporal dataset
from src.gnn.temporal_graph_dataset import TemporalGraphDataset


class StockGNN(torch.nn.Module):
    """GNN model for generating node embeddings from financial graphs."""

    def __init__(
        self,
        input_features: int = 5,
        hidden_features: int = 64,
        output_features: int = 32,
        dropout_rate: float = 0.2
    ):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, output_features)
        self.dropout_rate = dropout_rate

    def forward(self, x, edge_index, edge_weight):
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = torch.dropout(x, p=self.dropout_rate, train=self.training)
        x = self.conv2(x, edge_index, edge_weight)
        return x


def load_returns_data(data_path: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load returns data for proxy task.

    Args:
        data_path: Path to the parquet file containing returns data

    Returns:
        Tuple of (returns_dataframe, ticker_list)
    """
    data = pd.read_parquet(data_path)
    returns = data["Returns Daily"].unstack(level="Ticker")
    returns = returns.fillna(0.0)
    returns = returns.sort_index(axis=1)
    tickers = returns.columns.tolist()
    return returns, tickers


def get_next_day_returns(returns_df: pd.DataFrame, date_str: str, tickers: List[str]) -> np.ndarray:
    """
    Get next-day returns for given date and tickers.

    Args:
        returns_df: DataFrame with returns (index=date, columns=ticker)
        date_str: Date string in format 'YYYY-MM-DD HH:MM:SS'
        tickers: List of ticker symbols

    Returns:
        Array of next-day returns for each ticker
    """
    # Convert date string to pandas Timestamp
    target_date = pd.to_datetime(date_str)

    # Get the next trading day
    try:
        date_idx = returns_df.index.get_loc(target_date)
        if date_idx >= len(returns_df) - 1:
            # If we're at the last date, return zeros (no next day data)
            return np.zeros(len(tickers))
        next_date = returns_df.index[date_idx + 1]
        next_day_returns = returns_df.loc[next_date, tickers].values
    except KeyError:
        # If date not found, return zeros
        return np.zeros(len(tickers))

    return next_day_returns


def train_proxy_task(
    graph_dir: str,
    returns_data_path: str,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> StockGNN:
    """
    Train GNN using proxy task of predicting next-day returns.

    Args:
        graph_dir: Directory containing graph_*.npz files
        returns_data_path: Path to returns data parquet file
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training

    Returns:
        Trained StockGNN model
    """
    print("=" * 60)
    print("PROXY TASK TRAINING: Next-Day Return Prediction")
    print("=" * 60)

    # Load returns data
    print(f"Loading returns data from: {returns_data_path}")
    returns_df, _ = load_returns_data(returns_data_path)
    print(f"Returns data shape: {returns_df.shape}")

    # Create dataset and dataloaders
    print(f"\nLoading graph data from: {graph_dir}")
    dataset = TemporalGraphDataset(graph_dir)

    # We'll use all data for training in this simple example
    # In practice, you'd want to split into train/val to avoid overfitting
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        follow_batch=['x', 'graph_index']
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = StockGNN(
        input_features=5,
        hidden_features=64,
        output_features=32,
        dropout_rate=0.2
    ).to(device)

    print(f"Model architecture:\n{model}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        train_batch_count = 0

        for batch in dataloader:
            batch = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass to get embeddings
            embeddings = model(batch.x, batch.edge_index, batch.edge_weight)

            # For simplicity, we'll use a linear predictor on embeddings
            # In practice, you might want to use a separate predictor network
            predictor = nn.Linear(embeddings.size(1), 1).to(device)
            predictions = predictor(embeddings).squeeze(-1)

            # Reshape predictions to match batch format: (batch_size, num_nodes)
            # embeddings shape: (batch_size * num_nodes, embedding_dim)
            # predictions shape: (batch_size * num_nodes,)
            # We need: (batch_size, num_nodes)
            num_nodes = batch.x.size(0) // batch.num_graphs  # nodes per graph
            predictions = predictions.view(batch.num_graphs, num_nodes)

            # Get next-day returns for this batch
            # Use the built-in batch attribute which indicates graph membership for each node
            # Get the graph index for each node
            node_graph_indices = batch.batch  # This should be a tensor of shape [num_nodes]

            # Get unique graph indices in this batch
            unique_graph_indices = torch.unique(node_graph_indices)

            # Debug: print what we got
            if train_batch_count == 0:  # Only print for the first batch of each epoch
                print(f"Epoch {epoch+1}, Batch {train_batch_count}:")
                print(f"  Batch has {batch.num_graphs} graphs")
                print(f"  Unique graph indices from batch.batch: {unique_graph_indices.tolist()}")
                print(f"  Dataset length: {len(dataset)}")
                print(f"  Min index: {unique_graph_indices.min().item()}, Max index: {unique_graph_indices.max().item()}")

            batch_returns = []
            for graph_idx in unique_graph_indices:
                graph_idx_item = graph_idx.item()
                # Check if index is valid
                if graph_idx_item < 0 or graph_idx_item >= len(dataset):
                    print(f"ERROR: Invalid graph index {graph_idx_item} for dataset of length {len(dataset)}")
                    continue
                # Get the date from the dataset using the original graph index
                # We need to access the dataset directly since date_str is stored as list in the Data object
                graph_data = dataset[graph_idx_item]
                date_str = graph_data.date_str[0]  # All nodes have the same date string

                # Get next-day returns using graph-specific tickers
                next_returns = get_next_day_returns(returns_df, date_str, graph_data.tickers)
                batch_returns.append(next_returns)

            # Convert to tensor
            target_returns = torch.tensor(
                np.array(batch_returns),
                dtype=torch.float32
            ).to(device)

            # Compute loss
            loss = criterion(predictions, target_returns)

            # Backward pass
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            train_batch_count += 1

        avg_loss = total_loss / train_batch_count if train_batch_count > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

    print("\n" + "=" * 60)
    print("PROXY TASK TRAINING COMPLETED")
    print("=" * 60)

    return model


def evaluate_proxy_task(
    model: StockGNN,
    graph_dir: str,
    returns_data_path: str,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluate the proxy task performance.

    Args:
        model: Trained StockGNN model
        graph_dir: Directory containing graph_*.npz files
        returns_data_path: Path to returns data parquet file
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (R2 score, MAE)
    """
    print("\nEvaluating proxy task performance...")

    # Load returns data
    returns_df, _ = load_returns_data(returns_data_path)

    # Create dataset
    dataset = TemporalGraphDataset(graph_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=['x']
    )

    # Set model to evaluation mode
    model.eval()
    device = next(model.parameters()).device

    all_predictions = []
    all_targets = []
    eval_batch_count = 0

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Get embeddings
            embeddings = model(batch.x, batch.edge_index, batch.edge_weight)

            # Simple linear predictor (in practice, you'd train this separately)
            predictor = nn.Linear(embeddings.size(1), 1).to(device)
            # Initialize predictor with model's final state or load a trained one
            # For simplicity, we'll use random initialization here
            predictions = predictor(embeddings).squeeze(-1)

            # Reshape predictions to match batch format: (batch_size, num_nodes)
            num_nodes = batch.x.size(0) // batch.num_graphs  # nodes per graph
            predictions = predictions.view(batch.num_graphs, num_nodes)

            # Get target returns
            # Use the built-in batch attribute which indicates graph membership for each node
            # Get the graph index for each node
            node_graph_indices = batch.batch  # This should be a tensor of shape [num_nodes]

            # Get unique graph indices in this batch
            unique_graph_indices = torch.unique(node_graph_indices)

            # Debug: print what we got
            if eval_batch_count == 0:  # Only print for the first batch
                print(f"Evaluation Batch {eval_batch_count}:")
                print(f"  Batch has {batch.num_graphs} graphs")
                print(f"  Unique graph indices from batch.batch: {unique_graph_indices.tolist()}")
                print(f"  Dataset length: {len(dataset)}")
                print(f"  Min index: {unique_graph_indices.min().item()}, Max index: {unique_graph_indices.max().item()}")

            batch_returns = []
            for graph_idx in unique_graph_indices:
                graph_idx_item = graph_idx.item()
                # Check if index is valid
                if graph_idx_item < 0 or graph_idx_item >= len(dataset):
                    print(f"ERROR: Invalid graph index {graph_idx_item} for dataset of length {len(dataset)}")
                    continue
                # Get the date from the dataset using the original graph index
                # We need to access the dataset directly since date_str is stored as list in the Data object
                graph_data = dataset[graph_idx_item]
                date_str = graph_data.date_str[0]  # All nodes have the same date string

                # Get next-day returns using graph-specific tickers
                next_returns = get_next_day_returns(returns_df, date_str, graph_data.tickers)
                batch_returns.append(next_returns)

            target_returns = torch.tensor(
                np.array(batch_returns),
                dtype=torch.float32
            ).to(device)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target_returns.cpu().numpy())

            eval_batch_count += 1

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Calculate metrics
    r2 = r2_score(targets, predictions)
    mae = mean_absolute_error(targets, predictions)

    print(f"Proxy Task Performance:")
    print(f"  R2 Score: {r2:.6f}")
    print(f"  MAE: {mae:.6f}")

    return r2, mae


if __name__ == "__main__":
    # Set up paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
    RETURNS_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")

    print(f"Graph directory: {GRAPH_DIR}")
    print(f"Returns data path: {RETURNS_DATA_PATH}")

    # Train the model
    trained_model = train_proxy_task(
        graph_dir=GRAPH_DIR,
        returns_data_path=RETURNS_DATA_PATH,
        num_epochs=20,  # Reduced for demo
        learning_rate=0.001,
        batch_size=16
    )

    # Evaluate the model
    r2_score, mae = evaluate_proxy_task(
        model=trained_model,
        graph_dir=GRAPH_DIR,
        returns_data_path=RETURNS_DATA_PATH,
        batch_size=16
    )

    # Save the trained model
    model_dir = os.path.join(BASE_DIR, "src", "gnn", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "proxy_task_gnn.pth")
    torch.save(trained_model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")