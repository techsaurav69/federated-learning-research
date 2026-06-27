"""
models.py — Neural network architectures for FedL-SHAP experiments.

Provides:
  - TabularMLP: Multi-layer perceptron for tabular datasets (Credit Card Fraud)
  - SimpleCNN: Lightweight CNN for image datasets (MNIST)
  - get_model(): Factory function to instantiate by dataset name
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TabularMLP(nn.Module):
    """
    2-hidden-layer MLP for binary classification on tabular data.

    Architecture: input → 128 → 64 → 2
    Uses ReLU activations and dropout for regularization.
    """

    def __init__(self, input_dim: int = 29, num_classes: int = 2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


class SimpleCNN(nn.Module):
    """
    Lightweight CNN for MNIST digit classification.

    Architecture: 2 conv layers (32, 64 filters) → 2 FC layers (128 → 10)
    Uses max pooling and dropout.
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, 1, 28, 28)
        x = self.pool(F.relu(self.conv1(x)))   # → (batch, 32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))   # → (batch, 64, 7, 7)
        x = x.view(x.size(0), -1)             # → (batch, 64*7*7)
        x = self.dropout(F.relu(self.fc1(x)))  # → (batch, 128)
        x = self.fc2(x)                        # → (batch, num_classes)
        return x


def get_model(dataset_name: str, device: str = "cpu") -> nn.Module:
    """
    Factory function to create the appropriate model for a given dataset.

    Args:
        dataset_name: "mnist" or "creditcard"
        device: Target device ("cpu" or "cuda")

    Returns:
        Initialized model on the specified device.
    """
    if dataset_name == "mnist":
        model = SimpleCNN(num_classes=10)
    elif dataset_name == "creditcard":
        model = TabularMLP(input_dim=29, num_classes=2)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Use 'mnist' or 'creditcard'.")

    return model.to(device)
