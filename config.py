"""
config.py — Centralized hyperparameters and experiment configurations for FedL-SHAP.

All experiment parameters are defined here as dataclasses for type safety and clarity.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class FederatedConfig:
    """Core federated learning hyperparameters."""
    num_clients: int = 10                # N: total number of clients
    clients_per_round: int = 10          # Clients sampled per round (set = num_clients for full participation)
    local_epochs: int = 5                # E: local SGD epochs per round
    communication_rounds: int = 100      # T: total communication rounds
    batch_size: int = 64                 # Local training batch size
    learning_rate: float = 0.01          # Local SGD learning rate
    momentum: float = 0.9               # SGD momentum
    weight_decay: float = 1e-4           # L2 regularization


@dataclass
class PrivacyConfig:
    """Differential privacy and FedL-SHAP specific parameters."""
    epsilon: float = 50.0                # Per-round privacy budget (central DP model)
    delta: float = 1e-5                  # Privacy failure probability
    clip_weight: float = 5.0             # C_w: L2 clipping threshold for model updates
    clip_shap: float = 1.0              # C_s: L2 clipping threshold for SHAP vectors
    lambda_decay: float = 2.0           # Decay coefficient for dynamic epsilon-decoupling
    # Epsilon values to sweep for Pareto frontier experiments
    epsilon_values: List[float] = field(default_factory=lambda: [5.0, 10.0, 20.0, 50.0, 100.0])
    # Lambda values for sensitivity analysis
    lambda_values: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0, 5.0])


@dataclass
class DataConfig:
    """Dataset and partitioning configuration."""
    dataset: str = "mnist"               # "mnist" or "creditcard"
    data_dir: str = "./data"             # Directory for downloaded/stored datasets
    creditcard_path: str = "./data/creditcard.csv"  # Path to Kaggle Credit Card CSV
    dirichlet_alpha: float = 0.5         # α: Dirichlet concentration for Non-IID split
    test_fraction: float = 0.2           # Fraction of data reserved for global test set
    num_shap_samples: int = 100          # Number of background samples for SHAP computation
    # Alpha values to sweep for Non-IID experiments
    alpha_values: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0, 10.0])


@dataclass
class ByzantineConfig:
    """Byzantine attack simulation parameters."""
    attack_type: str = "inflate"         # "inflate", "random", "sign_flip"
    malicious_fraction: float = 0.3      # Fraction of clients that are malicious
    inflation_factor: float = 10.0       # Multiplier for feature inflation attack
    # Fractions to sweep for Byzantine experiments
    malicious_fractions: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4])


@dataclass
class ExperimentConfig:
    """Master experiment configuration."""
    federated: FederatedConfig = field(default_factory=FederatedConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    data: DataConfig = field(default_factory=DataConfig)
    byzantine: ByzantineConfig = field(default_factory=ByzantineConfig)

    # Output paths
    results_dir: str = "./results"
    plots_dir: str = "./results/plots"
    csv_dir: str = "./results/csv"

    # Reproducibility
    seed: int = 42

    # Device
    device: str = "cuda"  # Will be overridden to "cpu" if CUDA unavailable

    # Methods to compare
    methods: List[str] = field(default_factory=lambda: [
        "fedavg",               # FedAvg (No Privacy) — upper bound
        "fedavg_uniform_dp",    # FedAvg + Uniform DP on weights only
        "fedavg_global_shap",   # FedAvg + SHAP computed on global model post-aggregation
        "fedavg_fixed_split",   # FedAvg + Local SHAP with fixed 50/50 ε split
        "fedl_shap",            # FedL-SHAP (Proposed) — dynamic ε-decoupling
    ])

    def __post_init__(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.csv_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)


# ─── Preset Configurations ───────────────────────────────────────────────────

def get_quick_test_config() -> ExperimentConfig:
    """Minimal config for fast smoke testing."""
    cfg = ExperimentConfig()
    cfg.federated.num_clients = 5
    cfg.federated.clients_per_round = 5
    cfg.federated.communication_rounds = 10
    cfg.federated.local_epochs = 1             # Single epoch for speed
    cfg.federated.batch_size = 128             # Larger batches = fewer iterations
    cfg.data.num_shap_samples = 50
    cfg.privacy.epsilon_values = [2.0, 5.0, 8.0, 20.0]       # richer Pareto sweep (was [2.0, 8.0])
    cfg.privacy.lambda_values = [1.0, 2.0, 3.0]               # includes default λ=2.0 → fixes Fig5 blank panel
    cfg.byzantine.malicious_fractions = [0.0, 0.1, 0.2, 0.3]  # more fractions → fixes Fig8 (was [0.0, 0.2])
    return cfg


def get_mnist_config() -> ExperimentConfig:
    """Full experiment config for MNIST."""
    cfg = ExperimentConfig()
    cfg.data.dataset = "mnist"
    cfg.federated.communication_rounds = 100
    cfg.federated.num_clients = 10
    cfg.federated.clients_per_round = 10
    return cfg


def get_creditcard_config() -> ExperimentConfig:
    """Full experiment config for Credit Card Fraud."""
    cfg = ExperimentConfig()
    cfg.data.dataset = "creditcard"
    cfg.federated.communication_rounds = 100
    cfg.federated.num_clients = 10
    cfg.federated.clients_per_round = 10
    cfg.data.dirichlet_alpha = 1.0  # Less skew for tabular
    return cfg
