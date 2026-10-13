import os
import sys
# Add the project root to sys.path to allow imports like src.gnn.temporal_graph_dataset
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error
from typing import Tuple, List

# Import our temporal dataset and improved models
from src.gnn.temporal_graph_dataset import TemporalGraphDataset
from src.gnn.improved_gnn_model import ImprovedStockGNN, PureGATStockGNN


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
    model: torch.nn.Module,
    graph_dir: str,
    returns_data_path: str,
    num_epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32
) -> Tuple[torch.nn.Module, float, float]:
    """
    Train GNN using proxy task of predicting next-day returns.

    Args:
        model: GNN model to train
        graph_dir: Directory containing graph_*.npz files
        returns_data_path: Path to returns data parquet file
        num_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        batch_size: Batch size for training

    Returns:
        Tuple of (trained_model, final_r2, final_mae)
    """
    print("=" * 60)
    print(f"PROXY TASK TRAINING: {model.__class__.__name__}")
    print("=" * 60)

    # Load returns data
    print(f"Loading returns data from: {returns_data_path}")
    returns_df, _ = load_returns_data(returns_data_path)
    print(f"Returns data shape: {returns_df.shape}")

    # Create dataset and dataloaders
    print(f"\nLoading graph data from: {graph_dir}")
    dataset = TemporalGraphDataset(graph_dir)

    # Use 80% for training, 20% for validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        follow_batch=['x']
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        follow_batch=['x']
    )

    print(f"Created DataLoaders:")
    print(f"  Train batches: {len(train_loader)} ({train_size} graphs)")
    print(f"  Val batches: {len(val_loader)} ({val_size} graphs)")
    print(f"  Batch size: {batch_size}")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    model = model.to(device)
    print(f"Model architecture:\n{model}")

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    best_val_r2 = -float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            batch = batch.to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass to get embeddings
            embeddings = model(batch.x, batch.edge_index, batch.edge_weight)

            # Simple linear predictor on embeddings
            predictor = nn.Linear(embeddings.size(1), 1).to(device)
            predictions = predictor(embeddings).squeeze(-1)

            # Reshape predictions to match batch format: (batch_size, num_nodes)
            num_nodes = batch.x.size(0) // batch.num_graphs  # nodes per graph
            predictions = predictions.view(batch.num_graphs, num_nodes)

            # Get next-day returns for this batch
            batch_returns = []
            unique_graph_indices = torch.unique(batch.batch)

            for graph_idx in unique_graph_indices:
                graph_idx_item = graph_idx.item()
                # Get the actual dataset index from the split
                if isinstance(dataset, torch.utils.data.Subset):
                    actual_idx = dataset.indices[graph_idx_item]
                else:
                    actual_idx = graph_idx_item

                if actual_idx < 0 or actual_idx >= len(dataset):
                    continue

                # Get the date from the dataset
                graph_data = dataset[actual_idx]
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

            train_loss += loss.item()
            train_batches += 1

        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0

        # Validation phase
        model.eval()
        val_predictions = []
        val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)

                # Get embeddings
                embeddings = model(batch.x, batch.edge_index, batch.edge_weight)

                # Simple linear predictor
                predictor = nn.Linear(embeddings.size(1), 1).to(device)
                predictions = predictor(embeddings).squeeze(-1)

                # Reshape predictions
                num_nodes = batch.x.size(0) // batch.num_graphs
                predictions = predictions.view(batch.num_graphs, num_nodes)

                # Get target returns
                batch_returns = []
                unique_graph_indices = torch.unique(batch.batch)

                for graph_idx in unique_graph_indices:
                    graph_idx_item = graph_idx.item()
                    if isinstance(dataset, torch.utils.data.Subset):
                        actual_idx = dataset.indices[graph_idx_item]
                    else:
                        actual_idx = graph_idx_item

                    if actual_idx < 0 or actual_idx >= len(dataset):
                        continue

                    graph_data = dataset[actual_idx]
                    date_str = graph_data.date_str[0]

                    next_returns = get_next_day_returns(returns_df, date_str, graph_data.tickers)
                    batch_returns.append(next_returns)

                target_returns = torch.tensor(
                    np.array(batch_returns),
                    dtype=torch.float32
                ).to(device)

                val_predictions.append(predictions.cpu().numpy())
                val_targets.append(target_returns.cpu().numpy())

        # Calculate validation metrics
        if val_predictions and val_targets:
            val_predictions = np.concatenate(val_predictions, axis=0)
            val_targets = np.concatenate(val_targets, axis=0)
            val_r2 = r2_score(val_targets, val_predictions)
            val_mae = mean_absolute_error(val_targets, val_predictions)

            # Save best model
            if val_r2 > best_val_r2:
                best_val_r2 = val_r2
                best_model_state = model.state_dict().copy()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], "
                      f"Train Loss: {avg_train_loss:.6f}, "
                      f"Val R2: {val_r2:.6f}, Val MAE: {val_mae:.6f}")

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on validation set
    model.eval()
    val_predictions = []
    val_targets = []

    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)

            embeddings = model(batch.x, batch.edge_index, batch.edge_weight)
            predictor = nn.Linear(embeddings.size(1), 1).to(device)
            predictions = predictor(embeddings).squeeze(-1)

            num_nodes = batch.x.size(0) // batch.num_graphs
            predictions = predictions.view(batch.num_graphs, num_nodes)

            batch_returns = []
            unique_graph_indices = torch.unique(batch.batch)

            for graph_idx in unique_graph_indices:
                graph_idx_item = graph_idx.item()
                if isinstance(dataset, torch.utils.data.Subset):
                    actual_idx = dataset.indices[graph_idx_item]
                else:
                    actual_idx = graph_idx_item

                if actual_idx < 0 or actual_idx >= len(dataset):
                    continue

                graph_data = dataset[actual_idx]
                date_str = graph_data.date_str[0]

                next_returns = get_next_day_returns(returns_df, date_str, graph_data.tickers)
                batch_returns.append(next_returns)

            target_returns = torch.tensor(
                np.array(batch_returns),
                dtype=torch.float32
            ).to(device)

            val_predictions.append(predictions.cpu().numpy())
            val_targets.append(target_returns.cpu().numpy())

    if val_predictions and val_targets:
        val_predictions = np.concatenate(val_predictions, axis=0)
        val_targets = np.concatenate(val_targets, axis=0)
        final_r2 = r2_score(val_targets, val_predictions)
        final_mae = mean_absolute_error(val_targets, val_predictions)
    else:
        final_r2 = 0.0
        final_mae = float('inf')

    print("\n" + "=" * 60)
    print(f"PROXY TASK TRAINING COMPLETED: {model.__class__.__name__}")
    print(f"Final Validation R2: {final_r2:.6f}")
    print(f"Final Validation MAE: {final_mae:.6f}")
    print("=" * 60)

    return model, final_r2, final_mae


def evaluate_proxy_task(
    model: torch.nn.Module,
    graph_dir: str,
    returns_data_path: str,
    batch_size: int = 32
) -> Tuple[float, float]:
    """
    Evaluate the proxy task performance on full dataset.

    Args:
        model: Trained GNN model
        graph_dir: Directory containing graph_*.npz files
        returns_data_path: Path to returns data parquet file
        batch_size: Batch size for evaluation

    Returns:
        Tuple of (R2 score, MAE)
    """
    print(f"\nEvaluating {model.__class__.__name__} on full dataset...")

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

    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)

            # Get embeddings
            embeddings = model(batch.x, batch.edge_index, batch.edge_weight)

            # Simple linear predictor
            predictor = nn.Linear(embeddings.size(1), 1).to(device)
            predictions = predictor(embeddings).squeeze(-1)

            # Reshape predictions
            num_nodes = batch.x.size(0) // batch.num_graphs
            predictions = predictions.view(batch.num_graphs, num_nodes)

            # Get target returns
            batch_returns = []
            unique_graph_indices = torch.unique(batch.batch)

            for graph_idx in unique_graph_indices:
                graph_idx_item = graph_idx.item()
                if isinstance(dataset, torch.utils.data.Subset):
                    actual_idx = dataset.indices[graph_idx_item]
                else:
                    actual_idx = graph_idx_item

                if actual_idx < 0 or actual_idx >= len(dataset):
                    continue

                graph_data = dataset[actual_idx]
                date_str = graph_data.date_str[0]

                next_returns = get_next_day_returns(returns_df, date_str, graph_data.tickers)
                batch_returns.append(next_returns)

            target_returns = torch.tensor(
                np.array(batch_returns),
                dtype=torch.float32
            ).to(device)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(target_returns.cpu().numpy())

    # Concatenate all batches
    if all_predictions and all_targets:
        predictions = np.concatenate(all_predictions, axis=0)
        targets = np.concatenate(all_targets, axis=0)

        # Calculate metrics
        r2 = r2_score(targets, predictions)
        mae = mean_absolute_error(targets, predictions)

        print(f"Proxy Task Performance on Full Dataset:")
        print(f"  R2 Score: {r2:.6f}")
        print(f"  MAE: {mae:.6f}")

        return r2, mae
    else:
        print("No predictions generated")
        return 0.0, float('inf')


if __name__ == "__main__":
    # Set up paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
    RETURNS_DATA_PATH = os.path.join(BASE_DIR, "data", "clean", "full_dataset.parquet")

    print(f"Graph directory: {GRAPH_DIR}")
    print(f"Returns data path: {RETURNS_DATA_PATH}")

    # Define models to test
    models_to_test = [
        ("GCN (Original)", ImprovedStockGNN(5, 64, 32, 0.2, use_gat=False)),
        ("GAT Hybrid", ImprovedStockGNN(5, 64, 32, 0.2, use_gat=True, gat_heads=4)),
        ("Pure GAT", PureGATStockGNN(5, 64, 32, 0.2, heads=4))
    ]

    results = {}

    # Train and evaluate each model
    for name, model in models_to_test:
        print(f"\n{'='*80}")
        print(f"TRAINING AND EVALUATING: {name}")
        print(f"{'='*80}")

        # Train the model
        trained_model, train_r2, train_mae = train_proxy_task(
            model=model,
            graph_dir=GRAPH_DIR,
            returns_data_path=RETURNS_DATA_PATH,
            num_epochs=30,  # Reduced for faster testing
            learning_rate=0.001,
            batch_size=16
        )

        # Evaluate on full dataset
        final_r2, final_mae = evaluate_proxy_task(
            model=trained_model,
            graph_dir=GRAPH_DIR,
            returns_data_path=RETURNS_DATA_PATH,
            batch_size=16
        )

        results[name] = {
            'model': trained_model,
            'train_r2': train_r2,
            'train_mae': train_mae,
            'final_r2': final_r2,
            'final_mae': final_mae
        }

        # Save the trained model
        model_dir = os.path.join(BASE_DIR, "src", "gnn", "models")
        os.makedirs(model_dir, exist_ok=True)
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        model_path = os.path.join(model_dir, f"improved_gnn_{safe_name}.pth")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to: {model_path}")

    # Print summary
    print(f"\n{'='*80}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'Train R2':<10} {'Train MAE':<10} {'Final R2':<10} {'Final MAE':<10}")
    print("-" * 80)
    for name, result in results.items():
        print(f"{name:<20} {result['train_r2']:<10.6f} {result['train_mae']:<10.6f} "
              f"{result['final_r2']:<10.6f} {result['final_mae']:<10.6f}")

    # Find best model based on final R2
    best_model_name = max(results.keys(), key=lambda k: results[k]['final_r2'])
    best_result = results[best_model_name]
    print(f"\nBest model based on final R2: {best_model_name}")
    print(f"  Final R2: {best_result['final_r2']:.6f}")
    print(f"  Final MAE: {best_result['final_mae']:.6f}")