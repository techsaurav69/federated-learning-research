"""
data_utils.py — Dataset loading and Non-IID federated partitioning.

Provides:
  - load_dataset(): Loads MNIST or Credit Card Fraud dataset
  - partition_data_dirichlet(): Splits data across N clients using Dir(α)
  - get_client_loaders(): Returns per-client DataLoaders + global test loader
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from typing import List, Tuple, Dict


def load_mnist(data_dir: str = "./data") -> Tuple[TensorDataset, TensorDataset]:
    """
    Load MNIST dataset via torchvision.

    Returns:
        (train_dataset, test_dataset) as TensorDatasets with normalized images.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_data = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    return train_data, test_data


def load_fashionmnist(data_dir: str = "./data") -> Tuple[TensorDataset, TensorDataset]:
    """
    Load Fashion-MNIST dataset via torchvision.

    Same 28x28 grayscale format as MNIST — 10 clothing categories.
    Uses the standard normalization for Fashion-MNIST.

    Returns:
        (train_dataset, test_dataset) as torchvision datasets.
    """
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))   # Fashion-MNIST mean/std
    ])

    train_data = datasets.FashionMNIST(
        root=data_dir, train=True, download=True, transform=transform
    )
    test_data = datasets.FashionMNIST(
        root=data_dir, train=False, download=True, transform=transform
    )

    return train_data, test_data


def load_creditcard(csv_path: str = "./data/creditcard.csv") -> Tuple[TensorDataset, TensorDataset]:
    """
    Load Credit Card Fraud dataset from CSV.

    The dataset is expected to have columns V1–V28, Amount, and Class.
    Features are standardized; Amount is log-transformed.

    Returns:
        (train_dataset, test_dataset) as TensorDatasets.

    Raises:
        FileNotFoundError: If the CSV file is not found at the specified path.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Credit Card Fraud dataset not found at: {csv_path}\n"
            "Please download it from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud\n"
            "and place 'creditcard.csv' in the ./data/ directory."
        )

    df = pd.read_csv(csv_path)

    # Drop 'Time' column, log-transform 'Amount'
    df = df.drop(columns=["Time"])
    df["Amount"] = np.log1p(df["Amount"])

    # Separate features and labels
    X = df.drop(columns=["Class"]).values.astype(np.float32)
    y = df["Class"].values.astype(np.int64)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X).astype(np.float32)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    return train_dataset, test_dataset


def partition_data_dirichlet(
    labels: np.ndarray,
    num_clients: int,
    alpha: float,
    seed: int = 42
) -> Dict[int, np.ndarray]:
    """
    Partition dataset indices across clients using Dirichlet distribution.

    Creates Non-IID splits where lower α → more skewed label distributions.

    Args:
        labels: Array of integer labels for the entire training set.
        num_clients: Number of federated clients (N).
        alpha: Dirichlet concentration parameter.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping client_id → array of sample indices.
    """
    rng = np.random.default_rng(seed)
    num_classes = len(np.unique(labels))
    client_indices = {i: [] for i in range(num_clients)}

    for cls in range(num_classes):
        # Get all indices for this class
        cls_indices = np.where(labels == cls)[0]
        rng.shuffle(cls_indices)

        # Sample Dirichlet proportions for this class
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))

        # Ensure minimum 1 sample per client per class (if possible)
        proportions = proportions / proportions.sum()

        # Split indices according to proportions
        splits = (proportions * len(cls_indices)).astype(int)

        # Distribute remainder to random clients
        remainder = len(cls_indices) - splits.sum()
        for j in range(remainder):
            splits[j % num_clients] += 1

        # Assign indices to clients
        start = 0
        for client_id in range(num_clients):
            end = start + splits[client_id]
            client_indices[client_id].append(cls_indices[start:end])
            start = end

    # Concatenate all class indices for each client
    for client_id in range(num_clients):
        client_indices[client_id] = np.concatenate(client_indices[client_id])
        rng.shuffle(client_indices[client_id])

    return client_indices


def get_labels(dataset) -> np.ndarray:
    """Extract labels from a dataset (handles both torchvision datasets and TensorDatasets)."""
    if isinstance(dataset, TensorDataset):
        return dataset.tensors[1].numpy()
    elif hasattr(dataset, "targets"):
        targets = dataset.targets
        if isinstance(targets, torch.Tensor):
            return targets.numpy()
        return np.array(targets)
    else:
        # Fallback: iterate through dataset
        labels = []
        for _, label in dataset:
            labels.append(label if isinstance(label, int) else label.item())
        return np.array(labels)


def get_client_loaders(
    dataset_name: str,
    num_clients: int,
    alpha: float,
    batch_size: int,
    data_dir: str = "./data",
    creditcard_path: str = "./data/creditcard.csv",
    seed: int = 42
) -> Tuple[List[DataLoader], DataLoader, int]:
    """
    Main entry point: load dataset, partition across clients, return DataLoaders.

    Args:
        dataset_name: "mnist", "fashionmnist", or "creditcard"
        num_clients: Number of federated clients.
        alpha: Dirichlet α for Non-IID partitioning.
        batch_size: Batch size for DataLoaders.
        data_dir: Directory for datasets.
        creditcard_path: Path to Credit Card CSV.
        seed: Random seed.

    Returns:
        (client_loaders, test_loader, num_features)
        - client_loaders: List of DataLoaders, one per client
        - test_loader: Global test DataLoader
        - num_features: Number of input features (for SHAP computation)
    """
    # Load dataset
    if dataset_name == "mnist":
        train_dataset, test_dataset = load_mnist(data_dir)
        num_features = 784  # 28 * 28 flattened for SHAP
    elif dataset_name == "fashionmnist":
        train_dataset, test_dataset = load_fashionmnist(data_dir)
        num_features = 784  # same 28 * 28 as MNIST
    elif dataset_name == "creditcard":
        train_dataset, test_dataset = load_creditcard(creditcard_path)
        num_features = 29   # V1-V28 + log(Amount)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist', 'fashionmnist', or 'creditcard'.")

    # Extract labels and partition
    labels = get_labels(train_dataset)
    client_indices = partition_data_dirichlet(labels, num_clients, alpha, seed)

    # Create per-client DataLoaders
    client_loaders = []
    for client_id in range(num_clients):
        indices = client_indices[client_id]
        subset = Subset(train_dataset, indices)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=True, drop_last=False)
        client_loaders.append(loader)

    # Global test loader
    test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

    # Print partition statistics
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name.upper()} | Clients: {num_clients} | alpha: {alpha}")
    print(f"{'='*60}")
    for cid in range(num_clients):
        client_labels = labels[client_indices[cid]]
        unique, counts = np.unique(client_labels, return_counts=True)
        dist_str = ", ".join([f"{u}:{c}" for u, c in zip(unique, counts)])
        print(f"  Client {cid:2d}: {len(client_indices[cid]):5d} samples | Labels: {dist_str}")
    print(f"  Test set: {len(test_dataset)} samples")
    print(f"{'='*60}\n")

    return client_loaders, test_loader, num_features


def get_dataset_sizes(client_loaders: List[DataLoader]) -> List[int]:
    """Get the number of samples per client from their DataLoaders."""
    return [len(loader.dataset) for loader in client_loaders]
