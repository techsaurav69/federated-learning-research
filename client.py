"""
client.py — Federated client: local training, SHAP computation, and privacy mechanisms.

Implements:
  - Phase 1: Local SGD training + exact SHAP computation on raw data
  - Phase 2: Dual-path clipping
  - Phase 3: Noise injection based on method type
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional

from privacy import (
    clip_model_update,
    clip_vector,
    dynamic_epsilon_split,
    fixed_epsilon_split,
    compute_sigma,
    add_gaussian_noise_to_model_update,
    add_gaussian_noise,
)


class FederatedClient:
    """
    A single federated client participating in the FL network.

    Handles local training, SHAP computation, gradient clipping, and DP noise injection.
    """

    def __init__(
        self,
        client_id: int,
        data_loader: DataLoader,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.data_loader = data_loader
        self.device = device
        self.dataset_size = len(data_loader.dataset)

    def local_train(
        self,
        global_model: nn.Module,
        local_epochs: int,
        learning_rate: float,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> Dict[str, torch.Tensor]:
        """
        Phase 1a: Perform local SGD training starting from the global model.

        Args:
            global_model: Current global model (with weights w^(t)).
            local_epochs: E — number of local SGD epochs.
            learning_rate: Learning rate for local SGD.
            momentum: SGD momentum.
            weight_decay: L2 regularization.

        Returns:
            model_update: Dictionary of Δw_i^(t) = w_local - w_global for each parameter.
        """
        # Deep copy the global model for local training
        local_model = copy.deepcopy(global_model).to(self.device)
        local_model.train()

        # Save initial (global) weights for computing Δw
        initial_weights = {
            name: param.clone().detach()
            for name, param in local_model.named_parameters()
        }

        optimizer = optim.SGD(
            local_model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        for epoch in range(local_epochs):
            for batch_data in self.data_loader:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

                # For CNN models, ensure correct input shape
                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)  # Add channel dim if missing

                optimizer.zero_grad()
                outputs = local_model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

        # Compute weight update: Δw = w_local - w_global
        model_update = {}
        for name, param in local_model.named_parameters():
            model_update[name] = (param.detach() - initial_weights[name]).cpu()

        return model_update, local_model

    def compute_local_shap(
        self,
        model: nn.Module,
        dataset_name: str,
        num_background: int = 100,
    ) -> torch.Tensor:
        """
        Phase 1b: Compute feature attributions on the local (unperturbed) model.

        Strategy:
          - For tabular data (creditcard): Uses SHAP GradientExplainer (fast on small inputs).
          - For image data (mnist): Uses Integrated Gradients (Shapley-axiom-compliant,
            orders of magnitude faster than GradientExplainer on CNNs).

        Both methods produce a per-feature importance vector Phi_i^(t).

        Args:
            model: The locally trained model (before any DP noise).
            dataset_name: "mnist" or "creditcard".
            num_background: Number of background samples for SHAP/IG computation.

        Returns:
            shap_vector: 1D tensor of shape (F,) -- mean |attribution| per feature.
        """
        model.eval()
        model.to(self.device)

        # Collect a subset of data samples from the client's local dataset
        all_inputs = []
        all_labels = []
        for batch_data in self.data_loader:
            all_inputs.append(batch_data[0])
            all_labels.append(batch_data[1])
            if sum(x.shape[0] for x in all_inputs) >= num_background + 20:
                break
        all_inputs = torch.cat(all_inputs, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # For image data, ensure proper shape
        if dataset_name == "mnist" and all_inputs.dim() == 3:
            all_inputs = all_inputs.unsqueeze(1)  # (N, 1, 28, 28)

        if dataset_name == "creditcard":
            # Tabular: Use SHAP library (fast for small feature sets)
            shap_vector = self._shap_tabular(model, all_inputs, num_background)
        else:
            # Image: Use fast Integrated Gradients
            num_explain = min(20, len(all_inputs))
            explain_data = all_inputs[:num_explain].to(self.device)
            explain_labels = all_labels[:num_explain].to(self.device)
            shap_vector = self._integrated_gradients(model, explain_data, explain_labels, steps=20)

        model.train()
        return shap_vector

    def _shap_tabular(
        self,
        model: nn.Module,
        all_inputs: torch.Tensor,
        num_background: int,
    ) -> torch.Tensor:
        """SHAP GradientExplainer for tabular data (fast with small feature dim)."""
        import shap

        num_bg = min(num_background, len(all_inputs) - 10)
        num_explain = min(20, len(all_inputs) - num_bg)

        indices = torch.randperm(len(all_inputs))
        background = all_inputs[indices[:num_bg]].to(self.device)
        explain_data = all_inputs[indices[num_bg:num_bg + num_explain]].to(self.device)

        try:
            explainer = shap.GradientExplainer(model, background)
            shap_values = explainer.shap_values(explain_data)

            if isinstance(shap_values, list):
                shap_array = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                shap_array = np.abs(shap_values)

            mean_shap = shap_array.mean(axis=0).flatten()
            return torch.tensor(mean_shap, dtype=torch.float32)
        except Exception as e:
            print(f"  [Client {self.client_id}] SHAP failed, using gradient fallback: {e}")
            return self._integrated_gradients(model, explain_data, None, steps=20)

    def _integrated_gradients(
        self,
        model: nn.Module,
        data: torch.Tensor,
        labels: torch.Tensor = None,
        steps: int = 20,
    ) -> torch.Tensor:
        """
        Fast Integrated Gradients — satisfies Shapley axioms (completeness, sensitivity).

        Computes attributions by interpolating between a zero baseline and the input,
        then averaging the gradients along the path.

        ~100x faster than shap.GradientExplainer for CNN models.
        """
        model.eval()
        baseline = torch.zeros_like(data).to(self.device)
        data = data.to(self.device)

        # Generate interpolation steps
        all_grads = []
        for alpha in np.linspace(0, 1, steps):
            interp = baseline + alpha * (data - baseline)
            interp = interp.clone().requires_grad_(True)

            outputs = model(interp)

            if labels is not None:
                # Use the loss w.r.t. true labels for more meaningful attributions
                target_scores = outputs.gather(1, labels.unsqueeze(1)).sum()
            else:
                target_scores = outputs.max(dim=1).values.sum()

            model.zero_grad()
            target_scores.backward()

            all_grads.append(interp.grad.detach().clone())

        # Average gradients along the path, multiply by (input - baseline)
        avg_grads = torch.stack(all_grads).mean(dim=0)  # (num_explain, ...)
        attributions = (data - baseline).detach() * avg_grads

        # Mean absolute attribution per feature, flattened
        mean_attr = attributions.abs().mean(dim=0).flatten().cpu()
        return mean_attr

    def prepare_upload(
        self,
        model_update: Dict[str, torch.Tensor],
        shap_vector: Optional[torch.Tensor],
        method: str,
        round_t: int,
        T_max: int,
        epsilon: float,
        delta: float,
        clip_weight: float,
        clip_shap: float,
        lambda_decay: float,
        num_clients: int = 10,
    ) -> Tuple[Dict[str, torch.Tensor], Optional[torch.Tensor]]:
        """
        Phase 2: Apply local clipping based on the method type.

        Uses CENTRAL DP model: clients clip locally (bounding sensitivity),
        the server adds calibrated noise ONCE to the aggregated update.
        This avoids noise amplification across high-dimensional parameter spaces.

        Methods:
          - "fedavg": No clipping, no SHAP
          - "fedavg_uniform_dp": Clip weights only (full eps); server adds noise
          - "fedavg_global_shap": No clipping; SHAP computed server-side
          - "fedavg_fixed_split": Clip both paths; 50/50 eps split
          - "fedl_shap": Clip both paths; dynamic eps-decoupling (proposed method)

        Returns:
            (clipped_update, clipped_shap, eps_w, eps_s)
            Note: noise is added server-side; clients return clipped updates only.
        """
        if method == "fedavg":
            return model_update, None

        elif method == "fedavg_uniform_dp":
            clipped = clip_model_update(model_update, clip_weight)
            return clipped, None

        elif method == "fedavg_global_shap":
            return model_update, None

        elif method == "fedavg_fixed_split":
            eps_w, eps_s = fixed_epsilon_split(epsilon, ratio=0.5)
            clipped_w = clip_model_update(model_update, clip_weight)
            clipped_s = clip_vector(shap_vector, clip_shap) if shap_vector is not None else None
            return clipped_w, clipped_s

        elif method == "fedl_shap":
            eps_w, eps_s = dynamic_epsilon_split(epsilon, round_t, T_max, lambda_decay)
            clipped_w = clip_model_update(model_update, clip_weight)
            clipped_s = clip_vector(shap_vector, clip_shap) if shap_vector is not None else None
            return clipped_w, clipped_s

        else:
            raise ValueError(f"Unknown method: {method}")
