"""
visualization.py — Publication-quality plots for FedL-SHAP experiments.

Generates IEEE-style figures:
  - Figure 2: Accuracy vs. Communication Rounds
  - Figure 3: SHAP Cosine Similarity vs. Communication Rounds
  - Figure 4: Privacy-Explainability Pareto Frontier
  - Figure 5: Dynamic ε-Decoupling Visualization
  - Figure 6: λ Sensitivity Analysis
  - Figure 7: SHAP Feature Importance Comparison (Bar Plot)
  - Figure 8: Byzantine Resilience
  - Figure 9: Final Accuracy vs. ε
"""

import os
import json
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Optional

from config import ExperimentConfig


# ─── Style Configuration ─────────────────────────────────────────────────────

def setup_plot_style():
    """Configure matplotlib for IEEE publication-quality figures."""
    plt.rcParams.update({
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "font.size": 12,
        "font.family": "serif",
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1,
    })
    sns.set_palette("deep")


# Method display names and colors
METHOD_NAMES = {
    "fedavg": "FedAvg (No Privacy)",
    "fedavg_uniform_dp": "FedAvg + Uniform DP",
    "fedavg_global_shap": "FedAvg + Global SHAP",
    "fedavg_fixed_split": "FedAvg + Fixed Split",
    "fedl_shap": "FedL-SHAP (Proposed)",
}

METHOD_COLORS = {
    "fedavg": "#2196F3",            # Blue
    "fedavg_uniform_dp": "#FF9800", # Orange
    "fedavg_global_shap": "#9C27B0",# Purple
    "fedavg_fixed_split": "#4CAF50",# Green
    "fedl_shap": "#F44336",         # Red (highlight proposed)
}

METHOD_MARKERS = {
    "fedavg": "o",
    "fedavg_uniform_dp": "s",
    "fedavg_global_shap": "^",
    "fedavg_fixed_split": "D",
    "fedl_shap": "*",
}

METHOD_LINESTYLES = {
    "fedavg": "--",
    "fedavg_uniform_dp": "-.",
    "fedavg_global_shap": ":",
    "fedavg_fixed_split": "-.",
    "fedl_shap": "-",
}


def load_csv(csv_dir: str, experiment: str, key: str) -> Optional[pd.DataFrame]:
    """Load a CSV result file."""
    filepath = os.path.join(csv_dir, f"{experiment}_{key}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return None


# ─── Figure 2: Accuracy vs. Communication Rounds ─────────────────────────────

def plot_accuracy_vs_rounds(csv_dir: str, plots_dir: str):
    """Figure 2: Test accuracy over communication rounds for all methods."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    has_data = False
    for method in METHOD_NAMES:
        df = load_csv(csv_dir, "experiment_A", method)
        if df is None:
            continue
        has_data = True

        # Subsample markers for clarity
        marker_every = max(1, len(df) // 10)

        ax.plot(
            df["round"], df["accuracy"],
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markevery=marker_every,
            markersize=7,
            linewidth=2.5 if method == "fedl_shap" else 1.8,
            zorder=5 if method == "fedl_shap" else 3,
        )

    if not has_data:
        print("  [WARN] No data found for Experiment A. Skipping Figure 2.")
        plt.close()
        return

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Classification Performance vs. Communication Rounds")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True)

    filepath = os.path.join(plots_dir, "fig2_accuracy_vs_rounds.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 3: SHAP Cosine Similarity vs. Rounds ─────────────────────────────

def plot_shap_similarity_vs_rounds(csv_dir: str, plots_dir: str):
    """Figure 3: SHAP cosine similarity over communication rounds."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    shap_methods = ["fedavg_global_shap", "fedavg_fixed_split", "fedl_shap"]
    has_data = False

    for method in shap_methods:
        df = load_csv(csv_dir, "experiment_B", method)
        if df is None:
            continue
        has_data = True
        marker_every = max(1, len(df) // 10)

        ax.plot(
            df["round"], df["shap_cosine"],
            label=METHOD_NAMES[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            marker=METHOD_MARKERS[method],
            markevery=marker_every,
            linewidth=2.5 if method == "fedl_shap" else 1.8,
        )

    if not has_data:
        print("  [WARN] No data found for Experiment B. Skipping Figure 3.")
        plt.close()
        return

    ax.set_xlabel("Communication Round")
    ax.set_ylabel("SHAP Cosine Similarity")
    ax.set_title("Explanation Fidelity vs. Communication Rounds")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True)

    filepath = os.path.join(plots_dir, "fig3_shap_similarity_vs_rounds.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 4: Pareto Frontier ───────────────────────────────────────────────

def plot_pareto_frontier(csv_dir: str, plots_dir: str):
    """Figure 4: Privacy budget vs. SHAP fidelity with accuracy as dot size."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 6))

    dp_methods = ["fedavg_uniform_dp", "fedavg_fixed_split", "fedl_shap"]
    has_data = False

    for method in dp_methods:
        epsilons = []
        shap_scores = []
        accuracies = []

        for file in sorted(glob.glob(os.path.join(csv_dir, f"experiment_C_{method}_eps*.csv"))):
            df = pd.read_csv(file)
            # Extract epsilon from filename
            eps = float(file.split("eps")[-1].replace(".csv", ""))
            epsilons.append(eps)
            shap_scores.append(df["shap_cosine"].iloc[-1])  # Final round SHAP
            accuracies.append(df["accuracy"].iloc[-1])        # Final accuracy

        if epsilons:
            has_data = True
            # Normalize accuracy for dot size
            sizes = [(a / 100.0) * 200 + 50 for a in accuracies]

            ax.scatter(
                epsilons, shap_scores,
                s=sizes,
                c=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                label=METHOD_NAMES[method],
                edgecolors="black",
                linewidths=0.5,
                alpha=0.85,
                zorder=5 if method == "fedl_shap" else 3,
            )
            # Connect points with a line
            sorted_idx = np.argsort(epsilons)
            ax.plot(
                [epsilons[i] for i in sorted_idx],
                [shap_scores[i] for i in sorted_idx],
                color=METHOD_COLORS[method],
                linestyle=METHOD_LINESTYLES[method],
                alpha=0.5,
                linewidth=1.5,
            )

    if not has_data:
        print("  [WARN] No data found for Experiment C. Skipping Figure 4.")
        plt.close()
        return

    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Final SHAP Cosine Similarity")
    ax.set_title("Privacy–Explainability Pareto Frontier\n(dot size ∝ accuracy)")
    ax.legend(loc="lower right", framealpha=0.9)
    ax.grid(True)

    filepath = os.path.join(plots_dir, "fig4_pareto_frontier.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 5: Dynamic ε-Decoupling ──────────────────────────────────────────

def plot_dynamic_budget(csv_dir: str, plots_dir: str, cfg: ExperimentConfig):
    """Figure 5: ε_w(t) and ε_s(t) over communication rounds."""
    setup_plot_style()

    schedule_path = os.path.join(csv_dir, "experiment_D_schedules.json")
    if not os.path.exists(schedule_path):
        print("  [WARN] No schedule data found. Skipping Figure 5.")
        return

    with open(schedule_path, "r") as f:
        schedule_data = json.load(f)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Left: ε_w and ε_s for the default λ
    default_key = f"lambda_{cfg.privacy.lambda_decay}"
    if default_key in schedule_data:
        eps_w = schedule_data[default_key]["epsilon_w"]
        eps_s = schedule_data[default_key]["epsilon_s"]
        rounds = list(range(len(eps_w)))

        ax1.plot(rounds, eps_w, label="ε_w (Weights)", color="#2196F3", linewidth=2.5)
        ax1.plot(rounds, eps_s, label="ε_s (SHAP)", color="#F44336", linewidth=2.5)
        ax1.fill_between(rounds, eps_w, alpha=0.15, color="#2196F3")
        ax1.fill_between(rounds, eps_s, alpha=0.15, color="#F44336")
        ax1.axhline(y=cfg.privacy.epsilon / 2, color="gray", linestyle=":", alpha=0.5,
                     label="50/50 Split")
        ax1.set_xlabel("Communication Round")
        ax1.set_ylabel("Privacy Budget (ε)")
        ax1.set_title(f"Dynamic ε-Decoupling (λ = {cfg.privacy.lambda_decay})")
        ax1.legend(framealpha=0.9)
        ax1.grid(True)

    # Right: ε_w and ε_s for multiple λ values
    colors_w = ["#BBDEFB", "#90CAF9", "#42A5F5", "#1565C0"]
    colors_s = ["#FFCDD2", "#EF9A9A", "#EF5350", "#B71C1C"]

    for i, (key, data) in enumerate(schedule_data.items()):
        lam = key.replace("lambda_", "λ=")
        ax2.plot(range(len(data["epsilon_w"])), data["epsilon_w"],
                 label=f"ε_w ({lam})", color=colors_w[i % len(colors_w)],
                 linestyle="-", linewidth=1.5)
        ax2.plot(range(len(data["epsilon_s"])), data["epsilon_s"],
                 label=f"ε_s ({lam})", color=colors_s[i % len(colors_s)],
                 linestyle="--", linewidth=1.5)

    ax2.set_xlabel("Communication Round")
    ax2.set_ylabel("Privacy Budget (ε)")
    ax2.set_title("Budget Schedules Across λ Values")
    ax2.legend(fontsize=8, ncol=2, framealpha=0.9)
    ax2.grid(True)

    fig.tight_layout()
    filepath = os.path.join(plots_dir, "fig5_dynamic_budget.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 6: λ Sensitivity Analysis ────────────────────────────────────────

def plot_lambda_sensitivity(csv_dir: str, plots_dir: str, cfg: ExperimentConfig):
    """Figure 6: Effect of λ on final accuracy and SHAP fidelity."""
    setup_plot_style()

    lambdas = []
    final_acc = []
    final_shap = []

    for lam in cfg.privacy.lambda_values:
        df = load_csv(csv_dir, "experiment_D", f"fedl_shap_lambda{lam}")
        if df is not None:
            lambdas.append(lam)
            final_acc.append(df["accuracy"].iloc[-1])
            final_shap.append(df["shap_cosine"].iloc[-1])

    if not lambdas:
        print("  [WARN] No data found for lambda sensitivity. Skipping Figure 6.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax2 = ax1.twinx()

    bar_width = 0.35
    x = np.arange(len(lambdas))

    bars1 = ax1.bar(x - bar_width/2, final_acc, bar_width, label="Accuracy (%)",
                    color="#2196F3", alpha=0.8, edgecolor="black", linewidth=0.5)
    bars2 = ax2.bar(x + bar_width/2, final_shap, bar_width, label="SHAP Cosine Sim",
                    color="#F44336", alpha=0.8, edgecolor="black", linewidth=0.5)

    ax1.set_xlabel("Decay Coefficient (λ)")
    ax1.set_ylabel("Final Test Accuracy (%)", color="#2196F3")
    ax2.set_ylabel("Final SHAP Cosine Similarity", color="#F44336")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"λ={l}" for l in lambdas])
    ax1.set_title("λ Sensitivity Analysis: Accuracy vs. Explanation Fidelity")

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper center", framealpha=0.9)

    ax1.grid(True, axis="y", alpha=0.3)

    filepath = os.path.join(plots_dir, "fig6_lambda_sensitivity.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 7: SHAP Feature Importance Comparison ────────────────────────────

def plot_shap_comparison_bars(csv_dir: str, plots_dir: str):
    """Figure 7: Side-by-side SHAP feature importance bar plots."""
    setup_plot_style()

    # Load final SHAP values from each method
    shap_methods = ["fedavg_global_shap", "fedavg_fixed_split", "fedl_shap"]
    method_shap = {}

    for method in shap_methods:
        df = load_csv(csv_dir, "experiment_B", method)
        if df is not None:
            method_shap[method] = df

    if not method_shap:
        print("  [WARN] No SHAP data found for Figure 7. Skipping.")
        return

    # Create a summary table of final metrics
    fig, axes = plt.subplots(1, len(method_shap), figsize=(5 * len(method_shap), 5))
    if len(method_shap) == 1:
        axes = [axes]

    metrics = ["shap_cosine", "shap_topk", "shap_spearman"]
    metric_names = ["Cosine Sim", "Top-5 Agree", "Spearman ρ"]

    for idx, (method, df) in enumerate(method_shap.items()):
        ax = axes[idx]
        final_values = [df[m].iloc[-1] for m in metrics]

        bars = ax.bar(metric_names, final_values,
                      color=METHOD_COLORS[method], alpha=0.85,
                      edgecolor="black", linewidth=0.5)

        # Add value labels on bars
        for bar, val in zip(bars, final_values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10)

        ax.set_ylim(0, 1.2)
        ax.set_title(METHOD_NAMES[method], fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("SHAP Fidelity Metrics Comparison (Final Round)", fontsize=14, y=1.02)
    fig.tight_layout()

    filepath = os.path.join(plots_dir, "fig7_shap_comparison.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 8: Byzantine Resilience ──────────────────────────────────────────

def plot_byzantine_resilience(csv_dir: str, plots_dir: str, cfg: ExperimentConfig):
    """Figure 8: SHAP fidelity vs. fraction of malicious clients."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    attack_types = ["inflate", "random", "sign_flip"]
    attack_colors = {"inflate": "#F44336", "random": "#FF9800", "sign_flip": "#9C27B0"}
    attack_markers = {"inflate": "o", "random": "s", "sign_flip": "^"}

    has_data = False
    for attack in attack_types:
        fracs = []
        shap_scores = []

        for frac in cfg.byzantine.malicious_fractions:
            df = load_csv(csv_dir, "experiment_E", f"fedl_shap_{attack}_frac{frac}")
            if df is not None:
                fracs.append(frac)
                shap_scores.append(df["shap_cosine"].iloc[-1])

        if fracs:
            has_data = True
            ax.plot(
                fracs, shap_scores,
                label=f"{attack.replace('_', ' ').title()} Attack",
                color=attack_colors[attack],
                marker=attack_markers[attack],
                linewidth=2.0,
                markersize=8,
            )

    if not has_data:
        print("  [WARN] No data found for Experiment E. Skipping Figure 8.")
        plt.close()
        return

    ax.set_xlabel("Fraction of Malicious Clients")
    ax.set_ylabel("Final SHAP Cosine Similarity")
    ax.set_title("Byzantine Resilience: SHAP Fidelity Under Attack")
    ax.set_ylim(-0.1, 1.1)
    ax.legend(framealpha=0.9)
    ax.grid(True)

    filepath = os.path.join(plots_dir, "fig8_byzantine_resilience.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Figure 9: Final Accuracy vs. ε ──────────────────────────────────────────

def plot_accuracy_vs_epsilon(csv_dir: str, plots_dir: str, cfg: ExperimentConfig):
    """Figure 9: Final test accuracy vs. privacy budget for DP methods."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    dp_methods = ["fedavg_uniform_dp", "fedavg_fixed_split", "fedl_shap"]
    has_data = False

    for method in dp_methods:
        epsilons = []
        accuracies = []

        for file in sorted(glob.glob(os.path.join(csv_dir, f"experiment_C_{method}_eps*.csv"))):
            df = pd.read_csv(file)
            eps = float(file.split("eps")[-1].replace(".csv", ""))
            epsilons.append(eps)
            accuracies.append(df["accuracy"].iloc[-1])

        if epsilons:
            has_data = True
            sorted_idx = np.argsort(epsilons)
            ax.plot(
                [epsilons[i] for i in sorted_idx],
                [accuracies[i] for i in sorted_idx],
                label=METHOD_NAMES[method],
                color=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                linewidth=2.0,
                markersize=8,
            )

    if not has_data:
        print("  [WARN] No data found for accuracy vs epsilon. Skipping Figure 9.")
        plt.close()
        return

    ax.set_xlabel("Privacy Budget (ε)")
    ax.set_ylabel("Final Test Accuracy (%)")
    ax.set_title("Classification Performance vs. Privacy Budget")
    ax.legend(framealpha=0.9)
    ax.grid(True)

    filepath = os.path.join(plots_dir, "fig9_accuracy_vs_epsilon.png")
    fig.savefig(filepath)
    plt.close()
    print(f"  [OK] Saved: {filepath}")


# ─── Master Plot Generator ───────────────────────────────────────────────────

def generate_all_plots(cfg: ExperimentConfig):
    """Generate all publication figures from saved CSV results."""
    csv_dir = cfg.csv_dir
    plots_dir = cfg.plots_dir
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "=" * 60)
    print("  GENERATING PUBLICATION FIGURES")
    print("=" * 60)

    plot_accuracy_vs_rounds(csv_dir, plots_dir)
    plot_shap_similarity_vs_rounds(csv_dir, plots_dir)
    plot_pareto_frontier(csv_dir, plots_dir)
    plot_dynamic_budget(csv_dir, plots_dir, cfg)
    plot_lambda_sensitivity(csv_dir, plots_dir, cfg)
    plot_shap_comparison_bars(csv_dir, plots_dir)
    plot_byzantine_resilience(csv_dir, plots_dir, cfg)
    plot_accuracy_vs_epsilon(csv_dir, plots_dir, cfg)

    print("\n  All figures generated successfully! [OK]")


if __name__ == "__main__":
    # Standalone plot generation from existing CSV results
    cfg = ExperimentConfig()
    generate_all_plots(cfg)
