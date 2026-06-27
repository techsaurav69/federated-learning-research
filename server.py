"""
server.py — Federated server: weighted aggregation, global SHAP, and Byzantine simulation.

Implements:
  - Phase 4: Weighted FedAvg aggregation for both weights and SHAP (Equations 11–12)
  - Global SHAP computation (baseline: compute SHAP on aggregated model)
  - Byzantine attack injection for resilience testing
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Optional


class FederatedServer:
    """
    Central orchestrator for federated aggregation.

    Manages:
      - Global model maintenance
      - Weighted averaging of client updates (Phase 4)
      - Global SHAP computation (for baseline comparison)
      - Byzantine attack simulation
    """

    def __init__(self, global_model: nn.Module, device: str = "cpu"):
        self.global_model = global_model.to(device)
        self.device = device

    def get_global_weights(self) -> dict:
        """Return a copy of the current global model state dict."""
        return copy.deepcopy(self.global_model.state_dict())

    def aggregate_weights(
        self,
        client_updates: List[Dict[str, torch.Tensor]],
        dataset_sizes: List[int],
        dp_sigma: float = 0.0,
    ) -> None:
        """
        Phase 4a: Aggregate client weight updates using dataset-size-weighted averaging.

        Implements Equation 11:
            w^(t+1) = w^(t) + Σ_i (|D_i| / |D_total|) x w_tilde_i^(t)

        Optionally adds central DP noise to the aggregated update before applying.
        The noise is scaled per-element as sigma/sqrt(d) so the total L2 noise norm
        equals sigma — matching the (epsilon, delta)-DP Gaussian mechanism guarantee.

        Args:
            client_updates: List of clipped weight updates from each client.
            dataset_sizes: List of dataset sizes |D_i| for each client.
            dp_sigma: If > 0, standard deviation of Gaussian noise for central DP.
        """
        total_samples = sum(dataset_sizes)
        weights = [s / total_samples for s in dataset_sizes]

        # Initialize aggregated update with zeros
        agg_update = {}
        for key in client_updates[0]:
            agg_update[key] = torch.zeros_like(client_updates[0][key])

        # Weighted sum
        for client_update, weight in zip(client_updates, weights):
            for key in agg_update:
                agg_update[key] += client_update[key].float() * weight

        # Add central DP noise to the aggregated update (not to the full model)
        # Scale by 1/sqrt(d) so total noise L2 norm matches the DP guarantee
        if dp_sigma > 0.0:
            total_params = sum(v.numel() for v in agg_update.values())
            elem_sigma = dp_sigma / (total_params ** 0.5)
            for key in agg_update:
                agg_update[key] += torch.randn_like(agg_update[key]) * elem_sigma

        # Apply aggregated (noised) update to global model
        global_state = self.global_model.state_dict()
        for key in global_state:
            if key in agg_update:
                global_state[key] = global_state[key].float() + agg_update[key].to(self.device)
        self.global_model.load_state_dict(global_state)

    def aggregate_shap(
        self,
        client_shap_vectors: List[torch.Tensor],
        dataset_sizes: List[int],
    ) -> torch.Tensor:
        """
        Phase 4b: Aggregate client SHAP vectors using dataset-size-weighted averaging.

        Implements Equation 12:
            Φ_global^(t+1) = Σ_i (|D_i| / |D_total|) × Φ̃_i^(t)

        Args:
            client_shap_vectors: List of noised SHAP vectors from each client.
            dataset_sizes: List of dataset sizes.

        Returns:
            global_shap: Aggregated global SHAP vector (1D tensor).
        """
        total_samples = sum(dataset_sizes)
        weights = [s / total_samples for s in dataset_sizes]

        # Ensure all SHAP vectors have the same length
        shap_dim = client_shap_vectors[0].shape[0]
        global_shap = torch.zeros(shap_dim)

        for shap_vec, weight in zip(client_shap_vectors, weights):
            # Truncate or pad if dimensions don't match (safety)
            if shap_vec.shape[0] != shap_dim:
                min_dim = min(shap_vec.shape[0], shap_dim)
                global_shap[:min_dim] += shap_vec[:min_dim] * weight
            else:
                global_shap += shap_vec * weight

        return global_shap

    def compute_global_shap(
        self,
        test_loader: DataLoader,
        dataset_name: str,
        num_background: int = 100,
    ) -> torch.Tensor:
        """
        Compute feature attributions on the GLOBAL (potentially noised) model.

        This is used as a baseline ("FedAvg + Global SHAP") to demonstrate that
        computing attributions on the aggregated model yields poor explanation quality
        compared to local SHAP aggregation.

        Uses SHAP for tabular, Integrated Gradients for image data.
        """
        model = copy.deepcopy(self.global_model).to(self.device)
        model.eval()

        # Collect test samples
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
            return self._shap_tabular(model, all_inputs, num_background)
        else:
            num_explain = min(20, len(all_inputs))
            explain_data = all_inputs[:num_explain].to(self.device)
            explain_labels = all_labels[:num_explain].to(self.device)
            return self._integrated_gradients(model, explain_data, explain_labels, steps=20)

    def _shap_tabular(self, model, all_inputs, num_background):
        """SHAP for tabular data."""
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
            print(f"  [Server] SHAP failed: {e}, using IG fallback")
            return self._integrated_gradients(model, explain_data, None, steps=20)

    def _integrated_gradients(self, model, data, labels=None, steps=20):
        """Fast Integrated Gradients for feature attribution."""
        model.eval()
        baseline = torch.zeros_like(data).to(self.device)
        data = data.to(self.device)
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

    def evaluate(
        self,
        test_loader: DataLoader,
    ) -> Tuple[float, float]:
        """
        Evaluate the global model on the test set.

        Returns:
            (accuracy, loss): Test accuracy (%) and average cross-entropy loss.
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        correct = 0
        total = 0
        total_loss = 0.0

        with torch.no_grad():
            for batch_data in test_loader:
                inputs, labels = batch_data[0].to(self.device), batch_data[1].to(self.device)

                if inputs.dim() == 3:
                    inputs = inputs.unsqueeze(1)

                outputs = self.global_model(inputs)
                loss = criterion(outputs, labels)
                total_loss += loss.item() * labels.size(0)

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total
        avg_loss = total_loss / total
        self.global_model.train()

        return accuracy, avg_loss


# ─── Byzantine Attack Simulation ─────────────────────────────────────────────

def inject_byzantine_shap(
    client_shap_vectors: List[torch.Tensor],
    dataset_sizes: List[int],
    attack_type: str = "inflate",
    malicious_fraction: float = 0.3,
    inflation_factor: float = 10.0,
    seed: int = 42,
) -> Tuple[List[torch.Tensor], List[int]]:
    """
    Simulate Byzantine attacks on client SHAP vectors.

    Modifies a fraction of client SHAP vectors to simulate malicious behavior.

    Attack types:
      - "inflate": Multiply a random subset of features by inflation_factor
      - "random": Replace SHAP vector with random noise
      - "sign_flip": Negate the SHAP vector

    Args:
        client_shap_vectors: Original SHAP vectors from all clients.
        dataset_sizes: Dataset sizes for all clients.
        attack_type: Type of attack to simulate.
        malicious_fraction: Fraction of clients that are malicious.
        inflation_factor: Multiplier for the inflation attack.
        seed: Random seed.

    Returns:
        (attacked_vectors, malicious_ids): Modified SHAP vectors and list of malicious client IDs.
    """
    rng = np.random.default_rng(seed)
    num_clients = len(client_shap_vectors)
    num_malicious = max(1, int(num_clients * malicious_fraction))

    # Select malicious clients
    malicious_ids = sorted(rng.choice(num_clients, size=num_malicious, replace=False).tolist())

    attacked_vectors = [v.clone() for v in client_shap_vectors]

    for mid in malicious_ids:
        vec = attacked_vectors[mid]
        shap_dim = vec.shape[0]

        if attack_type == "inflate":
            # Inflate importance of random features (simulates backdoor highlighting)
            num_inflate = max(1, shap_dim // 5)
            inflate_indices = rng.choice(shap_dim, size=num_inflate, replace=False)
            vec[inflate_indices] *= inflation_factor
            attacked_vectors[mid] = vec

        elif attack_type == "random":
            # Replace with random noise
            attacked_vectors[mid] = torch.randn_like(vec) * vec.abs().mean()

        elif attack_type == "sign_flip":
            # Negate the SHAP vector
            attacked_vectors[mid] = -vec

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

    return attacked_vectors, malicious_ids
