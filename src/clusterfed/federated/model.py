"""The binary classifier (PRD 9.9, v1; unchanged shape in v2 PRD Section 4.5).

Stated architecture: input -> 200 -> 100 -> 100 -> 2, tanh on hidden layers,
sigmoid on the output. Trained with BCE, so we treat the 2 output units as
independent logits passed through sigmoid and supervise the positive (attack)
unit; argmax over the two sigmoids gives the predicted class.

``dropout_p`` is 0.0 (no-op) for the federated model, matching v1/the paper's
stated shape exactly. Stage 4 refinement (PRD 4.5) sets it > 0 to build the
same architecture as an MC-Dropout uncertainty estimator.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClusterFedNet(nn.Module):
    """Feed-forward net matching the paper's stated shape, with optional MC-Dropout."""

    def __init__(self, input_dim: int, hidden_layers=(200, 100, 100), output_dim: int = 2,
                 dropout_p: float = 0.0):
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for width in hidden_layers:
            layers.append(nn.Linear(prev, width))
            layers.append(nn.Tanh())  # stated hidden activation
            if dropout_p > 0:
                layers.append(nn.Dropout(p=dropout_p))
            prev = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(prev, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return sigmoid activations over the output units (PRD: sigmoid output)."""
        h = self.features(x)
        return torch.sigmoid(self.head(h))


def build_model(input_dim: int, model_cfg, *, dropout_p: float = 0.0) -> ClusterFedNet:
    hidden = list(model_cfg.get("hidden_layers", [200, 100, 100]))
    output_dim = int(model_cfg.get("output_dim", 2))
    return ClusterFedNet(input_dim, hidden_layers=hidden, output_dim=output_dim, dropout_p=dropout_p)


def enable_mc_dropout(model: ClusterFedNet) -> None:
    """Put the model in eval mode except for Dropout layers (Gal & Ghahramani MC-Dropout)."""
    model.eval()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.train()
