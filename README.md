# FedL-SHAP: Privacy-Preserving Explainability in Federated Learning

**FedL-SHAP** is a federated learning framework that jointly optimizes differential privacy and SHAP explanation fidelity through dynamic privacy budget decoupling (`ε`-decoupling).

> **Paper:** *Privacy-Preserving Explainability in Federated Learning via Dynamic ε-Decoupling*

---

## 🗂️ Repository Structure

```
├── config.py            # All hyperparameters and experiment presets
├── client.py            # Federated client: local SGD, FedProx, SHAP, DP clipping
├── server.py            # Federated server: aggregation, DP noise injection
├── models.py            # Model architectures (SimpleCNN, TabularMLP)
├── data_utils.py        # Data loading + Dirichlet Non-IID partitioning
├── privacy.py           # DP mechanisms: clipping, noise, ε-split schedules
├── run_experiments.py   # Main experiment orchestrator (all experiments A–G)
├── visualization.py     # Plot generation
└── requirements.txt     # Python dependencies
```

---

## ⚡ Quick Start (Google Colab / Kaggle)

### Step 1 — Clone and install

```bash
git clone https://github.com/techsaurav69/federated-learning-research.git
cd federated-learning-research
pip install -r requirements.txt
```

### Step 2 — Smoke test (~5–10 minutes on CPU)

```bash
python run_experiments.py --quick --experiments A B
```

---

## 🚀 Main Experiment Commands

### Run on MNIST (recommended starting point)

```bash
# All experiments, MNIST, single seed (~60–90 min on Colab T4 GPU)
python run_experiments.py --dataset mnist --experiments A B C D E G
```

### Run on Fashion-MNIST (second dataset)

```bash
# Fashion-MNIST — same pipeline, harder classification task
python run_experiments.py --dataset fashionmnist --experiments A B C D E G
```

### Multi-seed statistical run (for mean ± std and significance tests)

```bash
# 5 seeds × Experiments A + B — produces mean ± std and Wilcoxon p-value
python run_experiments.py --dataset mnist --multi-seed --experiments A B F
```

### Custom seeds (e.g., 3 seeds for faster run)

```bash
python run_experiments.py --dataset mnist --multi-seed --seeds 42 123 456 --experiments F
```

### Only overhead profiling

```bash
python run_experiments.py --experiments G
```

---

## 📊 Experiments

| ID | Name | What it measures |
|----|------|-----------------|
| A  | Accuracy vs Rounds | Classification accuracy per round for all 6 methods |
| B  | SHAP Fidelity | Cosine similarity, Spearman ρ, Top-K, MSE vs oracle SHAP |
| C  | Pareto Frontier | Privacy-explainability trade-off across ε values |
| D  | Dynamic Budget + λ Ablation | Effect of decay coefficient; budget trajectories |
| E  | Byzantine Resilience | SHAP fidelity under inflate / random / sign-flip attacks |
| **F** | **Multi-Seed Statistics** | **mean ± std over 5 seeds + Wilcoxon significance test** |
| **G** | **Overhead Profiling** | **Round time, communication bytes (weights + SHAP)** |

---

## 🔬 Methods Compared

| Method | ID | Description |
|--------|----|-------------|
| FedAvg | M1 | No privacy, no SHAP — accuracy upper bound |
| FedAvg + Uniform DP | M2 | DP on weights only, no SHAP |
| FedAvg + Global SHAP | M3 | SHAP computed server-side post-aggregation |
| FedAvg + Fixed-Split DP | M4 | Local SHAP + fixed 50/50 ε budget split |
| **FedL-SHAP** | **M5** | **Proposed: dynamic ε-decoupling (the main contribution)** |
| FedProx | M6 | Proximal-term FL baseline (new), no SHAP |

---

## ⚙️ Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--dataset` | `mnist` | Dataset: `mnist`, `fashionmnist`, `creditcard` |
| `--rounds` | 50 | Communication rounds |
| `--clients` | 10 | Number of federated clients |
| `--epsilon` | 50.0 | Per-round privacy budget ε |
| `--seeds` | `42 123 456 789 1024` | Seeds for multi-seed runs |

---

## 📁 Outputs

After running, results are saved to:

```
results/
├── csv/
│   ├── experiment_A_*.csv       # Per-round metrics (accuracy, SHAP, cumulative ε)
│   ├── experiment_F_multiseed_summary.csv   # mean ± std + Wilcoxon p-value
│   ├── experiment_G_overhead.csv            # Communication + timing table
│   └── ...
└── plots/
    └── *.png                    # All generated figures
```

---

## 💻 Running on Google Colab

Paste this into a Colab cell to run everything:

```python
# ── Colab Setup Cell ──────────────────────────────────────────────────────────
!git clone https://github.com/techsaurav69/federated-learning-research.git
%cd federated-learning-research
!pip install -r requirements.txt -q

# Smoke test first
!python run_experiments.py --quick --experiments A B

# Full MNIST run (60-90 min on T4)
!python run_experiments.py --dataset mnist --experiments A B C D E G

# Multi-seed (runs 5× longer — use only experiments F)
!python run_experiments.py --dataset mnist --multi-seed --seeds 42 123 456 --experiments F
```

---

## 💻 Running on Kaggle

1. Create a new **Notebook** on Kaggle.
2. Enable **GPU accelerator** (Settings → Accelerator → GPU P100).
3. In the first code cell:

```python
!git clone https://github.com/techsaurav69/federated-learning-research.git
%cd federated-learning-research
!pip install -r requirements.txt -q
!python run_experiments.py --dataset mnist --experiments A B C D E G
```

---

## 📦 Citation

```bibtex
@inproceedings{fedlshap2026,
  title     = {Privacy-Preserving Explainability in Federated Learning via Dynamic ε-Decoupling},
  author    = {[Author Names]},
  year      = {2026}
}
```
