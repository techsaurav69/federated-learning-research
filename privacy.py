"""
privacy.py — Differential Privacy mechanisms and dynamic ε-decoupling for FedL-SHAP.

Implements:
  - L2-norm clipping (Phase 2: Equations 2–3)
  - Dynamic privacy budget splitting (Phase 3: Equations 4–6)
  - Gaussian noise calibration and injection (Phase 3: Equations 7–10)
  - Sequential composition tracking
"""

import math
import torch
import numpy as np
from typing import Tuple


# ─── Phase 2: Dual-Path Gradient Clipping (Equations 2–3) ────────────────────

def clip_vector(vector: torch.Tensor, clip_threshold: float) -> torch.Tensor:
    """
    Project a vector onto an L2-norm ball of radius `clip_threshold`.

    Implements:
        v̄ = v · min(1, C / ||v||₂)

    This bounds the L2-sensitivity of the vector for the subsequent
    Gaussian noise mechanism.

    Args:
        vector: Flat 1D tensor (weight update or SHAP vector).
        clip_threshold: C_w or C_s — maximum allowed L2 norm.

    Returns:
        Clipped vector with ||v̄||₂ ≤ clip_threshold.
    """
    l2_norm = torch.norm(vector, p=2)
    scale = min(1.0, clip_threshold / (l2_norm.item() + 1e-10))
    return vector * scale


def clip_model_update(
    model_update: dict,
    clip_threshold: float
) -> dict:
    """
    Clip a model update (state_dict delta) by flattening, clipping, and reshaping.

    Args:
        model_update: Dictionary of parameter name → tensor delta.
        clip_threshold: C_w — maximum allowed L2 norm for the flattened update.

    Returns:
        Clipped model update dictionary.
    """
    # Flatten all parameters into a single vector
    flat = torch.cat([v.flatten() for v in model_update.values()])
    clipped_flat = clip_vector(flat, clip_threshold)

    # Reshape back to original parameter shapes
    clipped_update = {}
    offset = 0
    for key, val in model_update.items():
        numel = val.numel()
        clipped_update[key] = clipped_flat[offset:offset + numel].reshape(val.shape)
        offset += numel

    return clipped_update


# ─── Phase 3: Dynamic Privacy Budgeting — ε-Decoupling (Equations 4–6) ──────

def dynamic_epsilon_split(
    epsilon_t: float,
    t: int,
    T_max: int,
    lambda_decay: float
) -> Tuple[float, float]:
    """
    Dynamically split the per-round privacy budget into weight and SHAP components.

    Implements the exponential decoupling:
        ε_w(t) = ε(t) × exp(-λ × (t / T_max))      (Equation 5)
        ε_s(t) = ε(t) × (1 - exp(-λ × (t / T_max))) (Equation 6)

    Early rounds (t << T_max): ε_w is large (prioritize convergence)
    Late rounds  (t → T_max): ε_s is large (prioritize explanation fidelity)

    Args:
        epsilon_t: Total per-round privacy budget ε(t).
        t: Current communication round (0-indexed).
        T_max: Total number of communication rounds.
        lambda_decay: λ — controls the speed of budget transfer.

    Returns:
        (epsilon_w, epsilon_s): Budget for weights and SHAP respectively.
        Satisfies ε_w + ε_s = ε_t.
    """
    ratio = t / max(T_max, 1)
    decay = math.exp(-lambda_decay * ratio)

    epsilon_w = epsilon_t * decay
    epsilon_s = epsilon_t * (1.0 - decay)

    # Ensure neither budget is zero (avoid division by zero in sigma computation)
    min_eps = epsilon_t * 0.01  # Floor at 1% of total budget
    epsilon_w = max(epsilon_w, min_eps)
    epsilon_s = max(epsilon_s, min_eps)

    # Renormalize to sum to epsilon_t
    total = epsilon_w + epsilon_s
    epsilon_w = epsilon_w * (epsilon_t / total)
    epsilon_s = epsilon_s * (epsilon_t / total)

    return epsilon_w, epsilon_s


def fixed_epsilon_split(
    epsilon_t: float,
    ratio: float = 0.5
) -> Tuple[float, float]:
    """
    Static 50/50 (or custom ratio) split of the privacy budget.
    Used as a baseline for comparison against dynamic splitting.

    Args:
        epsilon_t: Total per-round privacy budget.
        ratio: Fraction allocated to weights (remainder goes to SHAP).

    Returns:
        (epsilon_w, epsilon_s)
    """
    epsilon_w = epsilon_t * ratio
    epsilon_s = epsilon_t * (1.0 - ratio)
    return epsilon_w, epsilon_s


# ─── Phase 3: Gaussian Noise Calibration (Equations 7–8) ─────────────────────

def compute_sigma(
    clip_threshold: float,
    epsilon: float,
    delta: float
) -> float:
    """
    Compute the standard deviation of Gaussian noise for (ε, δ)-DP.

    Implements the analytic Gaussian mechanism:
        σ = C × sqrt(2 × ln(1.25 / δ)) / ε    (Equations 7–8)

    Args:
        clip_threshold: C_w or C_s — L2 sensitivity bound.
        epsilon: Privacy budget (ε_w or ε_s).
        delta: Privacy failure probability δ.

    Returns:
        σ: Standard deviation for the Gaussian noise.
    """
    return clip_threshold * math.sqrt(2.0 * math.log(1.25 / delta)) / epsilon


# ─── Phase 3: Noise Injection (Equations 9–10) ───────────────────────────────

def add_gaussian_noise(
    vector: torch.Tensor,
    sigma: float
) -> torch.Tensor:
    """
    Add calibrated Gaussian noise to a vector for differential privacy.

    Implements:
        ṽ = v̄ + N(0, σ² I)    (Equations 9–10)

    Args:
        vector: Clipped vector (weight update or SHAP vector).
        sigma: Standard deviation from compute_sigma().

    Returns:
        Noised vector.
    """
    noise = torch.randn_like(vector) * sigma
    return vector + noise


def add_gaussian_noise_to_model_update(
    model_update: dict,
    sigma: float
) -> dict:
    """
    Add Gaussian noise to each parameter tensor in a model update dictionary.

    Args:
        model_update: Clipped model update (state_dict delta).
        sigma: Standard deviation for the noise.

    Returns:
        Noised model update.
    """
    noised_update = {}
    for key, val in model_update.items():
        noise = torch.randn_like(val) * sigma
        noised_update[key] = val + noise
    return noised_update


# ─── Privacy Accounting ──────────────────────────────────────────────────────

def compute_cumulative_epsilon(
    per_round_epsilon: float,
    num_rounds: int,
    delta: float
) -> float:
    """
    Compute the cumulative privacy loss over T rounds using basic composition.

    Uses the advanced composition theorem:
        ε_total = sqrt(2T × ln(1/δ')) × ε + T × ε × (e^ε - 1)

    For simplicity we use the standard sequential composition:
        ε_total = T × ε  (loose upper bound)

    And the strong composition:
        ε_total ≈ ε × sqrt(2T × ln(1/δ))

    Args:
        per_round_epsilon: ε per round.
        num_rounds: T total rounds completed.
        delta: δ failure probability.

    Returns:
        Cumulative ε under strong composition.
    """
    # Strong composition theorem (tighter than linear)
    return per_round_epsilon * math.sqrt(2 * num_rounds * math.log(1.0 / delta))


def get_privacy_schedule(
    epsilon: float,
    T_max: int,
    lambda_decay: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate the full ε_w(t) and ε_s(t) schedules for visualization.

    Args:
        epsilon: Per-round total budget.
        T_max: Total communication rounds.
        lambda_decay: λ decay coefficient.

    Returns:
        (epsilon_w_schedule, epsilon_s_schedule) as numpy arrays of length T_max.
    """
    eps_w = np.zeros(T_max)
    eps_s = np.zeros(T_max)
    for t in range(T_max):
        eps_w[t], eps_s[t] = dynamic_epsilon_split(epsilon, t, T_max, lambda_decay)
    return eps_w, eps_s
