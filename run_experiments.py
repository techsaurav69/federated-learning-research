"""
run_experiments.py — Main orchestrator for all FedL-SHAP experiments.

Runs:
  - Experiment A: Classification accuracy vs. communication rounds
  - Experiment B: SHAP fidelity metrics (cosine similarity, top-K, MSE, Spearman)
  - Experiment C: Pareto frontier (privacy vs. explainability tradeoff)
  - Experiment D: Dynamic budget visualization + λ sensitivity analysis
  - Experiment E: Byzantine resilience under SHAP poisoning attacks

Usage:
  python run_experiments.py                        # Full run (MNIST)
  python run_experiments.py --dataset creditcard   # Credit Card Fraud
  python run_experiments.py --quick                # Quick smoke test
"""

import argparse
import copy
import os
import json
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from config import ExperimentConfig, get_quick_test_config, get_mnist_config, get_creditcard_config
from data_utils import get_client_loaders, get_dataset_sizes
from models import get_model
from client import FederatedClient
from server import FederatedServer, inject_byzantine_shap
from privacy import (dynamic_epsilon_split, fixed_epsilon_split,
                     get_privacy_schedule, compute_sigma, add_gaussian_noise)


# ─── Ground Truth SHAP Computation ───────────────────────────────────────────

def compute_ground_truth_shap(
    dataset_name: str,
    test_loader,
    device: str,
    data_dir: str = "./data",
    num_background: int = 100,
) -> torch.Tensor:
    """
    Train a centralized (non-private) model and compute ground-truth SHAP values.
    This serves as the reference for evaluating explanation fidelity.
    """
    import shap
    from models import get_model

    print("\n--- Computing Ground Truth SHAP (centralized, non-private model) ---")

    # Train centralized model on full training data
    if dataset_name == "mnist":
        from torchvision import datasets, transforms
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        full_train = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(full_train, batch_size=128, shuffle=True)
    else:
        from data_utils import load_creditcard
        train_ds, _ = load_creditcard(os.path.join(data_dir, "creditcard.csv"))
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=128, shuffle=True)

    model = get_model(dataset_name, device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(10):
        for batch_data in train_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            if dataset_name == "mnist" and inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Evaluate centralized model
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for batch_data in test_loader:
            inputs, labels = batch_data[0].to(device), batch_data[1].to(device)
            if dataset_name == "mnist" and inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            _, pred = torch.max(model(inputs), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f"  Centralized model accuracy: {100.0 * correct / total:.2f}%")

    # Compute feature attributions
    all_inputs = []
    all_labels = []
    for batch_data in test_loader:
        all_inputs.append(batch_data[0])
        all_labels.append(batch_data[1])
        if sum(x.shape[0] for x in all_inputs) >= num_background + 20:
            break
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if dataset_name == "mnist" and all_inputs.dim() == 3:
        all_inputs = all_inputs.unsqueeze(1)

    if dataset_name == "creditcard":
        # Tabular: use full SHAP library
        background = all_inputs[:num_background].to(device)
        explain_data = all_inputs[num_background:num_background + 20].to(device)
        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(explain_data)
            if isinstance(shap_values, list):
                shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_array = np.abs(shap_values)
            gt_shap = torch.tensor(shap_array.mean(axis=0).flatten(), dtype=torch.float32)
        except Exception as e:
            print(f"  Ground truth SHAP failed: {e}. Using IG fallback.")
            gt_shap = _integrated_gradients_standalone(model, explain_data, None, device, steps=30)
    else:
        # Image: use fast Integrated Gradients
        num_explain = min(20, len(all_inputs))
        explain_data = all_inputs[:num_explain].to(device)
        explain_labels = all_labels[:num_explain].to(device)
        gt_shap = _integrated_gradients_standalone(model, explain_data, explain_labels, device, steps=30)

    print(f"  Ground truth SHAP vector: dim={gt_shap.shape[0]}, "
          f"min={gt_shap.min():.6f}, max={gt_shap.max():.6f}")
    return gt_shap


def _integrated_gradients_standalone(model, data, labels, device, steps=30):
    """Standalone Integrated Gradients for ground truth computation."""
    model.eval()
    baseline = torch.zeros_like(data).to(device)
    data = data.to(device)
    all_grads = []
    for alpha in np.linspace(0, 1, steps):
        interp = baseline + alpha * (data - baseline)
        interp = interp.clone().requires_grad_(True)
        outputs = model(interp)
        if labels is not None:
            target_scores = outputs.gather(1, labels.unsqueeze(1)).sum()
        else:
            target_scores = outputs.max(dim=1).values.sum()
        model.zero_grad()
        target_scores.backward()
        all_grads.append(interp.grad.detach().clone())
    avg_grads = torch.stack(all_grads).mean(dim=0)
    attributions = (data - baseline).detach() * avg_grads
    return attributions.abs().mean(dim=0).flatten().cpu()


# ─── SHAP Fidelity Metrics ───────────────────────────────────────────────────

def shap_cosine_similarity(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Cosine similarity between predicted and ground-truth SHAP vectors."""
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim].float(), ground_truth[:min_dim].float()
    if torch.norm(p) < 1e-10 or torch.norm(g) < 1e-10:
        return 0.0
    return torch.nn.functional.cosine_similarity(p.unsqueeze(0), g.unsqueeze(0)).item()


def shap_top_k_agreement(predicted: torch.Tensor, ground_truth: torch.Tensor, k: int = 5) -> float:
    """Fraction of top-K features that overlap between predicted and ground truth."""
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim], ground_truth[:min_dim]
    k = min(k, min_dim)
    top_k_pred = set(torch.topk(p.abs(), k).indices.tolist())
    top_k_gt = set(torch.topk(g.abs(), k).indices.tolist())
    return len(top_k_pred & top_k_gt) / k


def shap_mse(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Mean Squared Error between predicted and ground-truth SHAP vectors."""
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim].float(), ground_truth[:min_dim].float()
    return torch.nn.functional.mse_loss(p, g).item()


def shap_spearman_rank(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    """Spearman rank correlation between SHAP feature importances."""
    from scipy.stats import spearmanr
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim].numpy(), ground_truth[:min_dim].numpy()
    if np.std(p) < 1e-10 or np.std(g) < 1e-10:
        return 0.0
    corr, _ = spearmanr(p, g)
    return float(corr) if not np.isnan(corr) else 0.0


# ─── Core Federated Training Loop ────────────────────────────────────────────

def run_federated(
    method: str,
    cfg: ExperimentConfig,
    client_loaders: List,
    test_loader,
    ground_truth_shap: torch.Tensor,
    epsilon_override: Optional[float] = None,
    lambda_override: Optional[float] = None,
    byzantine_config: Optional[Dict] = None,
) -> Dict:
    """
    Run one complete federated training experiment for a given method.

    Args:
        method: One of the 5 methods.
        cfg: Experiment configuration.
        client_loaders: Per-client DataLoaders.
        test_loader: Global test DataLoader.
        ground_truth_shap: Reference SHAP vector.
        epsilon_override: Override ε for sweep experiments.
        lambda_override: Override λ for sensitivity experiments.
        byzantine_config: Dict with attack settings (or None).

    Returns:
        Dictionary with per-round metrics.
    """
    epsilon = epsilon_override if epsilon_override is not None else cfg.privacy.epsilon
    lambda_decay = lambda_override if lambda_override is not None else cfg.privacy.lambda_decay
    T = cfg.federated.communication_rounds
    dataset_name = cfg.data.dataset

    # Initialize
    global_model = get_model(dataset_name, cfg.device)
    server = FederatedServer(global_model, cfg.device)
    dataset_sizes = get_dataset_sizes(client_loaders)

    clients = [
        FederatedClient(i, loader, cfg.device)
        for i, loader in enumerate(client_loaders)
    ]

    # Tracking metrics
    results = {
        "method": method,
        "epsilon": epsilon,
        "lambda": lambda_decay,
        "rounds": [],
        "accuracy": [],
        "loss": [],
        "shap_cosine": [],
        "shap_topk": [],
        "shap_mse": [],
        "shap_spearman": [],
        "epsilon_w": [],
        "epsilon_s": [],
    }

    compute_shap = method in ["fedavg_fixed_split", "fedl_shap", "fedavg_global_shap"]

    print(f"\n{'='*60}")
    print(f"  Method: {method} | eps={epsilon} | lambda={lambda_decay} | T={T}")
    print(f"{'='*60}")

    for t in tqdm(range(T), desc=f"  {method}", leave=True):
        # ── Phase 1: Local training + SHAP ──
        client_updates = []
        client_shap_vectors = []

        for client in clients:
            model_update, local_model = client.local_train(
                server.global_model,
                cfg.federated.local_epochs,
                cfg.federated.learning_rate,
                cfg.federated.momentum,
                cfg.federated.weight_decay,
            )

            # Compute local SHAP if method requires it
            shap_vec = None
            if method in ["fedavg_fixed_split", "fedl_shap"]:
                shap_vec = client.compute_local_shap(
                    local_model, dataset_name, cfg.data.num_shap_samples
                )

            # ── Phase 2: Client-side clipping only ──
            clipped_update, clipped_shap = client.prepare_upload(
                model_update, shap_vec, method, t, T,
                epsilon, cfg.privacy.delta,
                cfg.privacy.clip_weight, cfg.privacy.clip_shap,
                lambda_decay,
            )

            client_updates.append(clipped_update)
            if clipped_shap is not None:
                client_shap_vectors.append(clipped_shap)

        # ── Byzantine attack simulation ──
        if byzantine_config and client_shap_vectors:
            client_shap_vectors, _ = inject_byzantine_shap(
                client_shap_vectors, dataset_sizes,
                attack_type=byzantine_config.get("attack_type", "inflate"),
                malicious_fraction=byzantine_config.get("malicious_fraction", 0.3),
                inflation_factor=byzantine_config.get("inflation_factor", 10.0),
                seed=cfg.seed + t,
            )

        # ── Phase 3+4: Server aggregation with central DP noise ──
        # Compute sigma for weight path based on method
        dp_sigma_w = 0.0
        if method in ["fedavg_uniform_dp", "fedavg_fixed_split", "fedl_shap"]:
            if method == "fedl_shap":
                eps_w, eps_s = dynamic_epsilon_split(epsilon, t, T, lambda_decay)
            elif method == "fedavg_fixed_split":
                eps_w, eps_s = fixed_epsilon_split(epsilon, ratio=0.5)
            else:
                eps_w, eps_s = epsilon, 0.0
            dp_sigma_w = compute_sigma(cfg.privacy.clip_weight, eps_w, cfg.privacy.delta)
        else:
            eps_w, eps_s = epsilon, 0.0

        server.aggregate_weights(client_updates, dataset_sizes, dp_sigma=dp_sigma_w)

        # Aggregate SHAP then add noise to SHAP path
        global_shap = None
        if client_shap_vectors:
            global_shap = server.aggregate_shap(client_shap_vectors, dataset_sizes)
            if method in ["fedavg_fixed_split", "fedl_shap"]:
                if method == "fedl_shap":
                    _, eps_s = dynamic_epsilon_split(epsilon, t, T, lambda_decay)
                else:
                    _, eps_s = fixed_epsilon_split(epsilon, ratio=0.5)
                sigma_s = compute_sigma(cfg.privacy.clip_shap, eps_s, cfg.privacy.delta)
                global_shap = add_gaussian_noise(global_shap, sigma_s)
        elif method == "fedavg_global_shap":
            global_shap = server.compute_global_shap(
                test_loader, dataset_name, cfg.data.num_shap_samples
            )

        # ── Evaluate ──
        accuracy, loss = server.evaluate(test_loader)

        # ── SHAP fidelity metrics ──
        cos_sim = topk = mse = spearman = 0.0
        if global_shap is not None and ground_truth_shap is not None:
            cos_sim = shap_cosine_similarity(global_shap, ground_truth_shap)
            # k=10 for high-dim MNIST (784-dim); k=5 for low-dim creditcard (29-dim)
            k_topk = 10 if dataset_name == "mnist" else 5
            topk = shap_top_k_agreement(global_shap, ground_truth_shap, k=k_topk)
            mse = shap_mse(global_shap, ground_truth_shap)
            spearman = shap_spearman_rank(global_shap, ground_truth_shap)

        # Track epsilon budget splits
        if method == "fedl_shap":
            ew, es = dynamic_epsilon_split(epsilon, t, T, lambda_decay)
        elif method == "fedavg_fixed_split":
            ew, es = fixed_epsilon_split(epsilon)
        else:
            ew, es = epsilon, 0.0

        results["rounds"].append(t)
        results["accuracy"].append(accuracy)
        results["loss"].append(loss)
        results["shap_cosine"].append(cos_sim)
        results["shap_topk"].append(topk)
        results["shap_mse"].append(mse)
        results["shap_spearman"].append(spearman)
        results["epsilon_w"].append(ew)
        results["epsilon_s"].append(es)

        if (t + 1) % max(1, T // 10) == 0:
            tqdm.write(
                f"    Round {t+1}/{T} | Acc: {accuracy:.2f}% | Loss: {loss:.4f}"
                f" | SHAP cos: {cos_sim:.4f} | Top-5: {topk:.2f}"
            )

    return results


# ─── Experiment Runners ──────────────────────────────────────────────────────

def experiment_A_accuracy(cfg: ExperimentConfig, client_loaders, test_loader, gt_shap):
    """Experiment A: Accuracy vs. Communication Rounds for all methods."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT A: Classification Performance vs. Rounds")
    print("=" * 60)

    all_results = {}
    for method in cfg.methods:
        results = run_federated(method, cfg, client_loaders, test_loader, gt_shap)
        all_results[method] = results

    # Save results
    save_results(all_results, cfg.csv_dir, "experiment_A")
    return all_results


def experiment_B_shap_fidelity(cfg: ExperimentConfig, client_loaders, test_loader, gt_shap):
    """Experiment B: SHAP fidelity metrics across methods."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT B: SHAP Fidelity Metrics")
    print("=" * 60)

    # Methods that produce SHAP values
    shap_methods = ["fedavg_global_shap", "fedavg_fixed_split", "fedl_shap"]
    all_results = {}
    for method in shap_methods:
        results = run_federated(method, cfg, client_loaders, test_loader, gt_shap)
        all_results[method] = results

    save_results(all_results, cfg.csv_dir, "experiment_B")
    return all_results


def experiment_C_pareto(cfg: ExperimentConfig, client_loaders, test_loader, gt_shap):
    """Experiment C: Pareto frontier — sweep ε for all methods."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT C: Privacy-Explainability Pareto Frontier")
    print("=" * 60)

    all_results = {}
    dp_methods = ["fedavg_uniform_dp", "fedavg_fixed_split", "fedl_shap"]

    for eps in cfg.privacy.epsilon_values:
        for method in dp_methods:
            key = f"{method}_eps{eps}"
            results = run_federated(
                method, cfg, client_loaders, test_loader, gt_shap,
                epsilon_override=eps,
            )
            all_results[key] = results

    save_results(all_results, cfg.csv_dir, "experiment_C")
    return all_results


def experiment_D_budget_viz(cfg: ExperimentConfig, client_loaders, test_loader, gt_shap):
    """Experiment D: Dynamic budget visualization + λ sensitivity."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT D: Dynamic Budget & Lambda Sensitivity")
    print("=" * 60)

    all_results = {}
    for lam in cfg.privacy.lambda_values:
        key = f"fedl_shap_lambda{lam}"
        results = run_federated(
            "fedl_shap", cfg, client_loaders, test_loader, gt_shap,
            lambda_override=lam,
        )
        all_results[key] = results

    # Also save the ε schedule for plotting
    schedule_data = {}
    for lam in cfg.privacy.lambda_values:
        eps_w, eps_s = get_privacy_schedule(
            cfg.privacy.epsilon, cfg.federated.communication_rounds, lam
        )
        schedule_data[f"lambda_{lam}"] = {
            "epsilon_w": eps_w.tolist(),
            "epsilon_s": eps_s.tolist(),
        }

    schedule_path = os.path.join(cfg.csv_dir, "experiment_D_schedules.json")
    with open(schedule_path, "w") as f:
        json.dump(schedule_data, f)

    save_results(all_results, cfg.csv_dir, "experiment_D")
    return all_results


def experiment_E_byzantine(cfg: ExperimentConfig, client_loaders, test_loader, gt_shap):
    """Experiment E: Byzantine resilience under SHAP poisoning."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT E: Byzantine Resilience")
    print("=" * 60)

    all_results = {}
    attack_types = ["inflate", "random", "sign_flip"]

    for attack in attack_types:
        for frac in cfg.byzantine.malicious_fractions:
            key = f"fedl_shap_{attack}_frac{frac}"
            byz_config = {
                "attack_type": attack,
                "malicious_fraction": frac,
                "inflation_factor": cfg.byzantine.inflation_factor,
            }
            results = run_federated(
                "fedl_shap", cfg, client_loaders, test_loader, gt_shap,
                byzantine_config=byz_config if frac > 0 else None,
            )
            all_results[key] = results

    save_results(all_results, cfg.csv_dir, "experiment_E")
    return all_results


# ─── Utility Functions ───────────────────────────────────────────────────────

def save_results(all_results: Dict, csv_dir: str, experiment_name: str):
    """Save experiment results as CSV files."""
    for key, results in all_results.items():
        df = pd.DataFrame({
            "round": results["rounds"],
            "accuracy": results["accuracy"],
            "loss": results["loss"],
            "shap_cosine": results["shap_cosine"],
            "shap_topk": results["shap_topk"],
            "shap_mse": results["shap_mse"],
            "shap_spearman": results["shap_spearman"],
            "epsilon_w": results["epsilon_w"],
            "epsilon_s": results["epsilon_s"],
        })
        filepath = os.path.join(csv_dir, f"{experiment_name}_{key}.csv")
        df.to_csv(filepath, index=False)

    print(f"  [OK] Results saved to {csv_dir}/{experiment_name}_*.csv")


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── Main Entry Point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FedL-SHAP Experiment Runner")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "creditcard"],
                        help="Dataset to use (default: mnist)")
    parser.add_argument("--quick", action="store_true",
                        help="Run a quick smoke test with reduced parameters")
    parser.add_argument("--experiments", type=str, nargs="+",
                        default=["A", "B", "C", "D", "E"],
                        help="Which experiments to run (A B C D E)")
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of communication rounds")
    parser.add_argument("--clients", type=int, default=None,
                        help="Override number of clients")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override base epsilon")
    args = parser.parse_args()

    # Configuration
    if args.quick:
        cfg = get_quick_test_config()
    elif args.dataset == "creditcard":
        cfg = get_creditcard_config()
    else:
        cfg = get_mnist_config()

    # Apply overrides
    cfg.data.dataset = args.dataset
    if args.rounds:
        cfg.federated.communication_rounds = args.rounds
    if args.clients:
        cfg.federated.num_clients = args.clients
        cfg.federated.clients_per_round = args.clients
    if args.epsilon:
        cfg.privacy.epsilon = args.epsilon

    # Device setup
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[*] Device: {cfg.device}")
    print(f"[*] Dataset: {cfg.data.dataset}")
    print(f"[*] Clients: {cfg.federated.num_clients} | Rounds: {cfg.federated.communication_rounds}")
    print(f"[*] Epsilon: {cfg.privacy.epsilon} | Delta: {cfg.privacy.delta}")
    print(f"[*] C_w: {cfg.privacy.clip_weight} | C_s: {cfg.privacy.clip_shap}")
    print(f"[*] Lambda: {cfg.privacy.lambda_decay}")

    # Set seed
    set_seed(cfg.seed)

    # Load data and partition
    client_loaders, test_loader, num_features = get_client_loaders(
        cfg.data.dataset,
        cfg.federated.num_clients,
        cfg.data.dirichlet_alpha,
        cfg.federated.batch_size,
        cfg.data.data_dir,
        cfg.data.creditcard_path,
        cfg.seed,
    )

    # Compute ground truth SHAP
    gt_shap = compute_ground_truth_shap(
        cfg.data.dataset, test_loader, cfg.device,
        cfg.data.data_dir, cfg.data.num_shap_samples,
    )

    # Run experiments
    start_time = time.time()
    all_experiment_results = {}

    if "A" in args.experiments:
        all_experiment_results["A"] = experiment_A_accuracy(
            cfg, client_loaders, test_loader, gt_shap
        )

    if "B" in args.experiments:
        all_experiment_results["B"] = experiment_B_shap_fidelity(
            cfg, client_loaders, test_loader, gt_shap
        )

    if "C" in args.experiments:
        all_experiment_results["C"] = experiment_C_pareto(
            cfg, client_loaders, test_loader, gt_shap
        )

    if "D" in args.experiments:
        all_experiment_results["D"] = experiment_D_budget_viz(
            cfg, client_loaders, test_loader, gt_shap
        )

    if "E" in args.experiments:
        all_experiment_results["E"] = experiment_E_byzantine(
            cfg, client_loaders, test_loader, gt_shap
        )

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE -- Total time: {elapsed/60:.1f} minutes")
    print(f"  Results saved to: {cfg.csv_dir}")
    print(f"{'='*60}")

    # Generate plots
    print("\n--- Generating Plots ---")
    from visualization import generate_all_plots
    generate_all_plots(cfg)
    print(f"  Plots saved to: {cfg.plots_dir}")


if __name__ == "__main__":
    main()
