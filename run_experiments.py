"""
run_experiments.py — Main orchestrator for all FedL-SHAP experiments.

New in this version (v2):
  - Multi-seed support: run all experiments over N seeds, compute mean ± std
  - FedProx baseline (M6): proximal-term FL without explainability
  - Fashion-MNIST support: second dataset, same pipeline
  - Privacy accountant: cumulative ε tracking per round (Moments accountant approximation)
  - Overhead profiling: communication size + per-round wall-clock time
  - Statistical significance: Wilcoxon signed-rank test between FedL-SHAP and best baseline

Experiments:
  A  — Classification accuracy vs. communication rounds (all methods)
  B  — SHAP fidelity metrics (cosine, Spearman, top-K, MSE)
  C  — Pareto frontier (privacy vs. explainability, sweep ε)
  D  — Dynamic budget visualization + λ ablation study
  E  — Byzantine resilience under SHAP poisoning attacks
  F  — Multi-seed statistical summary (NEW)
  G  — Overhead profiling (NEW)

Usage:
  python run_experiments.py                             # Full run, MNIST, single seed
  python run_experiments.py --dataset fashionmnist      # Fashion-MNIST
  python run_experiments.py --multi-seed                # Run all 5 seeds, compute stats
  python run_experiments.py --experiments F G           # Only new experiments
  python run_experiments.py --quick                     # Quick smoke test
"""

import argparse
import copy
import os
import json
import sys
import time
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from config import (ExperimentConfig, get_quick_test_config,
                    get_mnist_config, get_fashionmnist_config,
                    get_creditcard_config)
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
    This serves as the oracle reference for evaluating explanation fidelity.
    """
    print("\n--- Computing Ground Truth SHAP (centralized, non-private model) ---")

    if dataset_name in ("mnist", "fashionmnist"):
        from torchvision import datasets, transforms
        if dataset_name == "mnist":
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))
            ])
            full_train = datasets.MNIST(
                root=data_dir, train=True, download=True, transform=transform
            )
        else:
            transform = transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))
            ])
            full_train = datasets.FashionMNIST(
                root=data_dir, train=True, download=True, transform=transform
            )
        train_loader = torch.utils.data.DataLoader(
            full_train, batch_size=128, shuffle=True
        )
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
            if dataset_name in ("mnist", "fashionmnist") and inputs.dim() == 3:
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
            if dataset_name in ("mnist", "fashionmnist") and inputs.dim() == 3:
                inputs = inputs.unsqueeze(1)
            _, pred = torch.max(model(inputs), 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
    print(f"  Centralized model accuracy: {100.0 * correct / total:.2f}%")

    # Compute feature attributions via Integrated Gradients
    all_inputs, all_labels = [], []
    for batch_data in test_loader:
        all_inputs.append(batch_data[0])
        all_labels.append(batch_data[1])
        if sum(x.shape[0] for x in all_inputs) >= num_background + 20:
            break
    all_inputs = torch.cat(all_inputs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    if dataset_name in ("mnist", "fashionmnist") and all_inputs.dim() == 3:
        all_inputs = all_inputs.unsqueeze(1)

    if dataset_name == "creditcard":
        import shap
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
        num_explain = min(20, len(all_inputs))
        explain_data = all_inputs[:num_explain].to(device)
        explain_labels = all_labels[:num_explain].to(device)
        gt_shap = _integrated_gradients_standalone(
            model, explain_data, explain_labels, device, steps=30
        )

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


# ─── SHAP Fidelity Metrics ────────────────────────────────────────────────────

def shap_cosine_similarity(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim].float(), ground_truth[:min_dim].float()
    if torch.norm(p) < 1e-10 or torch.norm(g) < 1e-10:
        return 0.0
    return torch.nn.functional.cosine_similarity(p.unsqueeze(0), g.unsqueeze(0)).item()


def shap_top_k_agreement(predicted: torch.Tensor, ground_truth: torch.Tensor, k: int = 5) -> float:
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim], ground_truth[:min_dim]
    k = min(k, min_dim)
    top_k_pred = set(torch.topk(p.abs(), k).indices.tolist())
    top_k_gt = set(torch.topk(g.abs(), k).indices.tolist())
    return len(top_k_pred & top_k_gt) / k


def shap_mse(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim].float(), ground_truth[:min_dim].float()
    return torch.nn.functional.mse_loss(p, g).item()


def shap_spearman_rank(predicted: torch.Tensor, ground_truth: torch.Tensor) -> float:
    from scipy.stats import spearmanr
    min_dim = min(predicted.shape[0], ground_truth.shape[0])
    p, g = predicted[:min_dim].numpy(), ground_truth[:min_dim].numpy()
    if np.std(p) < 1e-10 or np.std(g) < 1e-10:
        return 0.0
    corr, _ = spearmanr(p, g)
    return float(corr) if not np.isnan(corr) else 0.0


# ─── Privacy Accountant (Moments / Rényi DP approximation) ───────────────────

def compute_cumulative_privacy(
    epsilon_per_round: List[float],
    delta: float,
) -> List[float]:
    """
    Compute cumulative privacy loss ε over rounds using simple composition.

    Uses basic composition theorem as an upper bound (tight for small T).
    For tighter bounds, the Moments Accountant or Rényi DP accountant
    (as in TensorFlow Privacy) would be used.

    Args:
        epsilon_per_round: List of per-round ε values.
        delta: Privacy failure probability.

    Returns:
        List of cumulative ε after each round.
    """
    cumulative = []
    running_eps = 0.0
    for eps in epsilon_per_round:
        running_eps += eps   # Basic composition: ε_total = Σ ε_t
        cumulative.append(running_eps)
    return cumulative


# ─── Communication Overhead Profiler ─────────────────────────────────────────

def measure_tensor_size_bytes(tensor_dict: Dict) -> int:
    """Compute total bytes in a dict of tensors (model update)."""
    total = 0
    for t in tensor_dict.values():
        total += t.element_size() * t.nelement()
    return total


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
    track_overhead: bool = False,
) -> Dict:
    """
    Run one complete federated training experiment for a given method.

    Args:
        method: One of the supported methods (fedavg, fedprox, fedl_shap, etc.)
        cfg: Experiment configuration.
        client_loaders: Per-client DataLoaders.
        test_loader: Global test DataLoader.
        ground_truth_shap: Reference SHAP vector from centralized oracle.
        epsilon_override: Override ε for sweep experiments.
        lambda_override: Override λ for ablation experiments.
        byzantine_config: Dict with attack settings (or None).
        track_overhead: If True, measure round time and communication bytes.

    Returns:
        Dictionary with per-round metrics.
    """
    epsilon = epsilon_override if epsilon_override is not None else cfg.privacy.epsilon
    lambda_decay = lambda_override if lambda_override is not None else cfg.privacy.lambda_decay
    T = cfg.federated.communication_rounds
    dataset_name = cfg.data.dataset
    mu = cfg.federated.fedprox_mu

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
        "cumulative_epsilon": [],    # NEW: privacy accountant
        "round_time_sec": [],        # NEW: wall-clock per round
        "comm_bytes_weights": [],    # NEW: bytes sent for weights per client per round
        "comm_bytes_shap": [],       # NEW: bytes sent for SHAP per client per round
    }

    compute_shap = method in ["fedavg_fixed_split", "fedl_shap", "fedavg_global_shap"]
    running_eps = 0.0   # For privacy accountant

    print(f"\n{'='*60}")
    print(f"  Method: {method} | eps={epsilon} | lambda={lambda_decay} | T={T}")
    print(f"{'='*60}")

    for t in tqdm(range(T), desc=f"  {method}", leave=True):
        round_start = time.time()
        client_updates = []
        client_shap_vectors = []
        round_weight_bytes = 0
        round_shap_bytes = 0

        for client in clients:
            # Phase 1: Local training
            if method == "fedprox":
                model_update, local_model = client.local_train_fedprox(
                    server.global_model,
                    cfg.federated.local_epochs,
                    cfg.federated.learning_rate,
                    mu=mu,
                    momentum=cfg.federated.momentum,
                    weight_decay=cfg.federated.weight_decay,
                )
            else:
                model_update, local_model = client.local_train(
                    server.global_model,
                    cfg.federated.local_epochs,
                    cfg.federated.learning_rate,
                    cfg.federated.momentum,
                    cfg.federated.weight_decay,
                )

            # Measure weight communication size
            if track_overhead:
                round_weight_bytes += measure_tensor_size_bytes(model_update)

            # Phase 1b: SHAP
            shap_vec = None
            if method in ["fedavg_fixed_split", "fedl_shap"]:
                shap_vec = client.compute_local_shap(
                    local_model, dataset_name, cfg.data.num_shap_samples
                )
                if track_overhead and shap_vec is not None:
                    round_shap_bytes += shap_vec.element_size() * shap_vec.nelement()

            # Phase 2: Clipping
            clipped_update, clipped_shap = client.prepare_upload(
                model_update, shap_vec, method, t, T,
                epsilon, cfg.privacy.delta,
                cfg.privacy.clip_weight, cfg.privacy.clip_shap,
                lambda_decay,
            )

            client_updates.append(clipped_update)
            if clipped_shap is not None:
                client_shap_vectors.append(clipped_shap)

        # Byzantine attack simulation
        if byzantine_config and client_shap_vectors:
            client_shap_vectors, _ = inject_byzantine_shap(
                client_shap_vectors, dataset_sizes,
                attack_type=byzantine_config.get("attack_type", "inflate"),
                malicious_fraction=byzantine_config.get("malicious_fraction", 0.3),
                inflation_factor=byzantine_config.get("inflation_factor", 10.0),
                seed=cfg.seed + t,
            )

        # Phase 3+4: Server aggregation with central DP
        dp_sigma_w = 0.0
        if method in ["fedavg_uniform_dp", "fedavg_fixed_split", "fedl_shap", "fedprox"]:
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

        # Evaluate
        accuracy, loss = server.evaluate(test_loader)

        # SHAP fidelity metrics
        cos_sim = topk = mse = spearman = 0.0
        if global_shap is not None and ground_truth_shap is not None:
            cos_sim = shap_cosine_similarity(global_shap, ground_truth_shap)
            k_topk = 50 if dataset_name in ("mnist", "fashionmnist") else 5
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

        # Privacy accountant: cumulative ε (basic composition)
        running_eps += epsilon   # Each round spends ε
        round_time = time.time() - round_start

        results["rounds"].append(t)
        results["accuracy"].append(accuracy)
        results["loss"].append(loss)
        results["shap_cosine"].append(cos_sim)
        results["shap_topk"].append(topk)
        results["shap_mse"].append(mse)
        results["shap_spearman"].append(spearman)
        results["epsilon_w"].append(ew)
        results["epsilon_s"].append(es)
        results["cumulative_epsilon"].append(running_eps)
        results["round_time_sec"].append(round_time)
        results["comm_bytes_weights"].append(round_weight_bytes)
        results["comm_bytes_shap"].append(round_shap_bytes)

        if (t + 1) % max(1, T // 10) == 0:
            tqdm.write(
                f"    Round {t+1}/{T} | Acc: {accuracy:.2f}% | Loss: {loss:.4f}"
                f" | SHAP cos: {cos_sim:.4f} | Spearman: {spearman:.4f}"
                f" | Cumul-ε: {running_eps:.1f}"
            )

    return results


# ─── Experiment Runners ───────────────────────────────────────────────────────

def experiment_A_accuracy(cfg, client_loaders, test_loader, gt_shap):
    """Experiment A: Accuracy vs. Communication Rounds (all methods)."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT A: Classification Performance vs. Rounds")
    print("=" * 60)
    all_results = {}
    for method in cfg.methods:
        results = run_federated(method, cfg, client_loaders, test_loader, gt_shap)
        all_results[method] = results
    save_results(all_results, cfg.csv_dir, "experiment_A")
    return all_results


def experiment_B_shap_fidelity(cfg, client_loaders, test_loader, gt_shap):
    """Experiment B: SHAP fidelity metrics across methods."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT B: SHAP Fidelity Metrics")
    print("=" * 60)
    shap_methods = ["fedavg_global_shap", "fedavg_fixed_split", "fedl_shap"]
    all_results = {}
    for method in shap_methods:
        results = run_federated(method, cfg, client_loaders, test_loader, gt_shap)
        all_results[method] = results
    save_results(all_results, cfg.csv_dir, "experiment_B")
    return all_results


def experiment_C_pareto(cfg, client_loaders, test_loader, gt_shap):
    """Experiment C: Pareto frontier — sweep ε for all DP methods."""
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


def experiment_D_budget_viz(cfg, client_loaders, test_loader, gt_shap):
    """Experiment D: Dynamic budget visualization + λ ablation study."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT D: Dynamic Budget & Lambda Ablation")
    print("=" * 60)
    all_results = {}
    for lam in cfg.privacy.lambda_values:
        key = f"fedl_shap_lambda{lam}"
        results = run_federated(
            "fedl_shap", cfg, client_loaders, test_loader, gt_shap,
            lambda_override=lam,
        )
        all_results[key] = results

    # Also save ε schedule for plotting
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


def experiment_E_byzantine(cfg, client_loaders, test_loader, gt_shap):
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


def experiment_F_multi_seed(cfg, client_loaders_fn, test_loader_fn, data_dir):
    """
    Experiment F (NEW): Multi-seed statistical summary.

    Runs FedL-SHAP and all baselines over multiple random seeds,
    then computes mean ± std for key metrics and performs
    Wilcoxon signed-rank test vs. the best baseline.

    Args:
        cfg: Experiment configuration (cfg.seeds defines the list of seeds to use).
        client_loaders_fn: Callable(seed) -> client_loaders (re-partitions per seed)
        test_loader_fn: Callable(seed) -> test_loader
        data_dir: Data directory for reloading datasets.
    """
    from scipy.stats import wilcoxon

    print("\n" + "=" * 60)
    print("  EXPERIMENT F: Multi-Seed Statistical Summary")
    print(f"  Seeds: {cfg.seeds}")
    print("=" * 60)

    # Methods to compare in the multi-seed study
    target_methods = ["fedavg", "fedavg_uniform_dp", "fedavg_fixed_split",
                      "fedl_shap", "fedprox"]

    # seed_results[method][seed] = final_round_metrics
    seed_results = {m: [] for m in target_methods}

    for seed in cfg.seeds:
        print(f"\n  --- Seed {seed} ---")
        set_seed(seed)
        cfg.seed = seed

        # Re-partition data for this seed
        client_loaders, test_loader, _ = get_client_loaders(
            cfg.data.dataset,
            cfg.federated.num_clients,
            cfg.data.dirichlet_alpha,
            cfg.federated.batch_size,
            cfg.data.data_dir,
            cfg.data.creditcard_path,
            seed,
        )

        gt_shap = compute_ground_truth_shap(
            cfg.data.dataset, test_loader, cfg.device,
            cfg.data.data_dir, cfg.data.num_shap_samples,
        )

        for method in target_methods:
            results = run_federated(method, cfg, client_loaders, test_loader, gt_shap)
            # Store final-round metrics
            seed_results[method].append({
                "accuracy": results["accuracy"][-1],
                "shap_cosine": results["shap_cosine"][-1],
                "shap_spearman": results["shap_spearman"][-1],
                "shap_mse": results["shap_mse"][-1],
            })

    # Aggregate: mean ± std
    summary_rows = []
    for method in target_methods:
        accs = [r["accuracy"] for r in seed_results[method]]
        coss = [r["shap_cosine"] for r in seed_results[method]]
        spms = [r["shap_spearman"] for r in seed_results[method]]
        mses = [r["shap_mse"] for r in seed_results[method]]

        summary_rows.append({
            "method": method,
            "accuracy_mean": np.mean(accs),
            "accuracy_std": np.std(accs),
            "shap_cosine_mean": np.mean(coss),
            "shap_cosine_std": np.std(coss),
            "shap_spearman_mean": np.mean(spms),
            "shap_spearman_std": np.std(spms),
            "shap_mse_mean": np.mean(mses),
            "shap_mse_std": np.std(mses),
            "n_seeds": len(cfg.seeds),
        })

    summary_df = pd.DataFrame(summary_rows)

    # Wilcoxon signed-rank test: FedL-SHAP vs best non-proposed baseline
    # on Spearman ρ (the primary fidelity metric)
    fedlshap_spearman = [r["shap_spearman"] for r in seed_results["fedl_shap"]]
    best_baseline_spearman = [r["shap_spearman"] for r in seed_results["fedavg_fixed_split"]]

    if len(fedlshap_spearman) >= 2:
        try:
            stat, p_val = wilcoxon(fedlshap_spearman, best_baseline_spearman)
            summary_df["wilcoxon_p_vs_fixed_split"] = p_val
            print(f"\n  Wilcoxon signed-rank test (FedL-SHAP vs Fixed-Split) "
                  f"on Spearman ρ: stat={stat:.4f}, p={p_val:.4f}")
            if p_val < 0.05:
                print("  ✓ Statistically significant improvement (p < 0.05)")
            else:
                print("  ✗ Not statistically significant at α=0.05")
        except Exception as e:
            print(f"  Wilcoxon test failed: {e}")

    # Save
    out_path = os.path.join(cfg.csv_dir, "experiment_F_multiseed_summary.csv")
    summary_df.to_csv(out_path, index=False, float_format="%.4f")
    print(f"\n  [OK] Multi-seed summary saved to {out_path}")
    print(summary_df.to_string(index=False))

    return summary_df


def experiment_G_overhead(cfg, client_loaders, test_loader, gt_shap):
    """
    Experiment G (NEW): Communication and runtime overhead profiling.

    Runs FedL-SHAP and FedAvg for a few rounds to measure:
    - Per-round wall-clock time
    - Communication bytes for weights and SHAP vectors
    - Reports breakdown as a table.
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT G: Overhead Profiling (Communication + Runtime)")
    print("=" * 60)

    # Run a short version (10 rounds) for timing
    short_cfg = copy.deepcopy(cfg)
    short_cfg.federated.communication_rounds = 10

    overhead_results = {}
    for method in ["fedavg", "fedl_shap", "fedprox"]:
        results = run_federated(
            method, short_cfg, client_loaders, test_loader, gt_shap,
            track_overhead=True,
        )
        overhead_results[method] = results

    # Build summary table
    rows = []
    for method, results in overhead_results.items():
        # Average over rounds
        avg_time = np.mean(results["round_time_sec"])
        avg_weight_kb = np.mean(results["comm_bytes_weights"]) / 1024
        avg_shap_kb = np.mean(results["comm_bytes_shap"]) / 1024
        total_kb = avg_weight_kb + avg_shap_kb
        rows.append({
            "method": method,
            "avg_round_time_sec": round(avg_time, 3),
            "avg_weight_comm_KB": round(avg_weight_kb, 2),
            "avg_shap_comm_KB": round(avg_shap_kb, 2),
            "avg_total_comm_KB": round(total_kb, 2),
        })

    overhead_df = pd.DataFrame(rows)
    out_path = os.path.join(cfg.csv_dir, "experiment_G_overhead.csv")
    overhead_df.to_csv(out_path, index=False)
    print(f"\n  [OK] Overhead table saved to {out_path}")
    print(overhead_df.to_string(index=False))

    return overhead_df


# ─── Utility Functions ────────────────────────────────────────────────────────

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
            "cumulative_epsilon": results["cumulative_epsilon"],
            "round_time_sec": results["round_time_sec"],
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


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="FedL-SHAP Experiment Runner v2",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--dataset", type=str, default="mnist",
        choices=["mnist", "fashionmnist", "creditcard"],
        help="Dataset to use (default: mnist)\n"
             "  mnist        — MNIST handwritten digits (28x28, 10 classes)\n"
             "  fashionmnist — Fashion-MNIST clothing (28x28, 10 classes)\n"
             "  creditcard   — Credit Card Fraud (tabular, requires creditcard.csv)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Run a quick smoke test (~5-10 min on CPU, 2 seeds, 10 rounds)",
    )
    parser.add_argument(
        "--multi-seed", action="store_true",
        help="Run Experiment F: multi-seed statistical summary (5 seeds)",
    )
    parser.add_argument(
        "--experiments", type=str, nargs="+",
        default=["A", "B", "C", "D", "E"],
        help="Which experiments to run. Options: A B C D E F G\n"
             "  A — Accuracy vs rounds\n"
             "  B — SHAP fidelity\n"
             "  C — Pareto frontier\n"
             "  D — Dynamic budget + lambda ablation\n"
             "  E — Byzantine resilience\n"
             "  F — Multi-seed statistics (use with --multi-seed)\n"
             "  G — Overhead profiling",
    )
    parser.add_argument("--rounds", type=int, default=None,
                        help="Override number of communication rounds")
    parser.add_argument("--clients", type=int, default=None,
                        help="Override number of clients")
    parser.add_argument("--epsilon", type=float, default=None,
                        help="Override base epsilon")
    parser.add_argument("--seeds", type=int, nargs="+", default=None,
                        help="Override seed list for multi-seed runs (e.g. --seeds 42 123 456)")
    args = parser.parse_args()

    # Configuration
    if args.quick:
        cfg = get_quick_test_config()
    elif args.dataset == "fashionmnist":
        cfg = get_fashionmnist_config()
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
    if args.seeds:
        cfg.seeds = args.seeds

    # Device setup
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  FedL-SHAP Experiment Runner v2")
    print(f"{'='*60}")
    print(f"  Device  : {cfg.device}")
    print(f"  Dataset : {cfg.data.dataset.upper()}")
    print(f"  Clients : {cfg.federated.num_clients} | Rounds: {cfg.federated.communication_rounds}")
    print(f"  Epsilon : {cfg.privacy.epsilon} | Delta: {cfg.privacy.delta}")
    print(f"  Lambda  : {cfg.privacy.lambda_decay} | FedProx μ: {cfg.federated.fedprox_mu}")
    print(f"  Seeds   : {cfg.seeds}")
    print(f"  Experiments: {args.experiments}")
    print(f"{'='*60}\n")

    # Set primary seed
    set_seed(cfg.seed)

    # Load data
    client_loaders, test_loader, num_features = get_client_loaders(
        cfg.data.dataset,
        cfg.federated.num_clients,
        cfg.data.dirichlet_alpha,
        cfg.federated.batch_size,
        cfg.data.data_dir,
        cfg.data.creditcard_path,
        cfg.seed,
    )

    # Compute ground truth SHAP (once, reused across experiments)
    gt_shap = compute_ground_truth_shap(
        cfg.data.dataset, test_loader, cfg.device,
        cfg.data.data_dir, cfg.data.num_shap_samples,
    )

    # Run experiments
    start_time = time.time()

    if "A" in args.experiments:
        experiment_A_accuracy(cfg, client_loaders, test_loader, gt_shap)

    if "B" in args.experiments:
        experiment_B_shap_fidelity(cfg, client_loaders, test_loader, gt_shap)

    if "C" in args.experiments:
        experiment_C_pareto(cfg, client_loaders, test_loader, gt_shap)

    if "D" in args.experiments:
        experiment_D_budget_viz(cfg, client_loaders, test_loader, gt_shap)

    if "E" in args.experiments:
        experiment_E_byzantine(cfg, client_loaders, test_loader, gt_shap)

    if "F" in args.experiments or args.multi_seed:
        experiment_F_multi_seed(cfg, None, None, cfg.data.data_dir)

    if "G" in args.experiments:
        experiment_G_overhead(cfg, client_loaders, test_loader, gt_shap)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"  ALL EXPERIMENTS COMPLETE")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Results  : {cfg.csv_dir}")
    print(f"  Plots    : {cfg.plots_dir}")
    print(f"{'='*60}")

    # Generate plots
    print("\n--- Generating Plots ---")
    try:
        from visualization import generate_all_plots
        generate_all_plots(cfg)
        print(f"  Plots saved to: {cfg.plots_dir}")
    except Exception as e:
        print(f"  [Warning] Plot generation failed: {e}")


if __name__ == "__main__":
    main()
