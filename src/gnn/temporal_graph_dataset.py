import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader


class TemporalGraphDataset(Dataset):
    """
    Dataset for temporal graph snapshots.
    Each graph snapshot is treated as a separate graph in the dataset.
    """

    def __init__(self, graph_dir, transform=None, pre_transform=None):
        """
        Args:
            graph_dir (str): Directory containing graph_*.npz files
            transform (callable, optional): Transform to apply to each graph
            pre_transform (callable, optional): Transform to apply before saving
        """
        super().__init__(None, transform, pre_transform)

        self.graph_dir = graph_dir
        print(f"Looking for graph files in: {graph_dir}")
        self.graph_files = sorted([
            f for f in os.listdir(graph_dir)
            if f.startswith('graph_') and f.endswith('.npz')
        ])
        print(f"Found {len(self.graph_files)} graph files matching pattern")

        if len(self.graph_files) == 0:
            raise FileNotFoundError(f"No graph files found in {graph_dir}")

        print(f"Found {len(self.graph_files)} graph snapshots in {graph_dir}")
        # Print first few files to verify
        if len(self.graph_files) > 0:
            print(f"First 5 graph files: {self.graph_files[:5]}")
            print(f"Last 5 graph files: {self.graph_files[-5:]}")

    def len(self):
        return len(self.graph_files)

    def __getitem__(self, idx):
        # Check if idx is out of bounds for __getitem__
        if idx < 0 or idx >= len(self.graph_files):
            # Clamp to valid range
            idx = max(0, min(idx, len(self.graph_files) - 1))
        return self.get(idx)

    def get(self, idx):
        """
        Load and return a graph at index idx.

        Returns:
            torch_geometric.data.Data: Graph data object
        """
        filename = self.graph_files[idx]
        filepath = os.path.join(self.graph_dir, filename)
        # print(f"Loading graph from: {filepath}")  # Commented out to reduce noise

        data = np.load(filepath)

        x = torch.tensor(data['node_features'], dtype=torch.float32)
        edge_index = torch.tensor(data['edge_index'], dtype=torch.long).t().contiguous()
        edge_weight = torch.tensor(data['edge_weight'], dtype=torch.float32)

        graph_data = Data(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight
        )

        # Add metadata as node-level attributes (so they get copied correctly during batching)
        # Store original graph index as node-level data
        graph_index_tensor = torch.full((x.size(0),), idx, dtype=torch.long)
        graph_data.graph_index = graph_index_tensor
        # Store date as string - we'll need to handle this specially for batching
        # For now, let's store it as node-level data too
        date_list = [str(data['date'])] * x.size(0)
        graph_data.date_str = date_list

        # Store global attributes for graph-level access (used in proxy task)
        graph_data.graph_date = str(data['date'])
        graph_data.tickers = data['tickers']

        return graph_data


def create_temporal_dataloaders(graph_dir, batch_size=32, train_split=0.8, shuffle=True):
    """
    Create train and validation DataLoaders for temporal graph data.

    Args:
        graph_dir (str): Directory containing graph_*.npz files
        batch_size (int): Batch size for DataLoader
        train_split (float): Fraction of data to use for training
        shuffle (bool): Whether to shuffle the data

    Returns:
        tuple: (train_loader, val_loader)
    """
    # Create full dataset
    full_dataset = TemporalGraphDataset(graph_dir)

    # Calculate split sizes
    train_size = int(train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size

    # Split dataset
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        follow_batch=['x']  # Important for batching node features
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

    return train_loader, val_loader


def get_graph_for_proxy_task(graph_dir, graph_index):
    """
    Get a single graph for use in proxy task (next-day return prediction).

    Args:
        graph_dir (str): Directory containing graph_*.npz files
        graph_index (int): Index of the graph to retrieve

    Returns:
        tuple: (graph_data, date, tickers)
    """
    dataset = TemporalGraphDataset(graph_dir)
    graph_data = dataset[graph_index]

    return graph_data, graph_data.graph_date, graph_data.tickers


# Example usage and validation
if __name__ == "__main__":
    print("=" * 60)
    print("TEMPORAL GRAPH DATASET VALIDATION")
    print("=" * 60)

    # Set up paths - go up to project root, then to graphs/gnn_data
    # Current file: src/gnn/temporal_graph_dataset.py
    # Project root: ../../ (from src/gnn)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    print(f"BASE_DIR (three levels up): {BASE_DIR}")
    GRAPH_DIR = os.path.join(BASE_DIR, "graphs", "gnn_data")
    print(f"Constructed GRAPH_DIR: {GRAPH_DIR}")

    print(f"Loading graphs from: {GRAPH_DIR}")

    # Create dataset
    dataset = TemporalGraphDataset(GRAPH_DIR)

    # Get first graph
    first_graph = dataset[0]
    print(f"\nFirst graph:")
    print(f"  Date: {first_graph.graph_date}")
    print(f"  Node features shape: {first_graph.x.shape}")
    print(f"  Edge index shape: {first_graph.edge_index.shape}")
    print(f"  Edge weight shape: {first_graph.edge_weight.shape}")
    print(f"  Number of nodes: {first_graph.num_nodes}")
    print(f"  Number of edges: {first_graph.num_edges}")

    # Create DataLoaders
    train_loader, val_loader = create_temporal_dataloaders(
        GRAPH_DIR,
        batch_size=16,
        train_split=0.8
    )

    # Test one batch
    for batch in train_loader:
        print(f"\nBatch from DataLoader:")
        print(f"  Batch node features shape: {batch.x.shape}")
        print(f"  Batch edge index shape: {batch.edge_index.shape}")
        print(f"  Batch edge weight shape: {batch.edge_weight.shape}")
        print(f"  Batch vector (graph indices): {batch.batch.shape}")
        break

    print("\n" + "=" * 60)
    print("DATASET VALIDATION COMPLETED")
    print("=" * 60)