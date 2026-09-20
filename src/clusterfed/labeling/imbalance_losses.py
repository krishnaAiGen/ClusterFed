"""Imbalance-aware losses for Stage 4's local refinement MLP (PRD 4.4, second imbalance defense).

Operates on the 2-independent-sigmoid-output-unit convention shared with v1's
federated classifier: each unit is supervised against a one-hot target. These
losses apply only to the refinement-phase local MLP (Stage 4); the final
FedAvg classifier keeps plain BCE, since the PRD specifies federated training
stays "exactly as in v1" (ablation A5 studies plain BCE / focal / logit-adjusted
as three levels for this stage only).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _logits(model, x: torch.Tensor) -> torch.Tensor:
    """Pre-sigmoid logits for a ClusterFedNet, via its ``features``/``head`` submodules."""
    return model.head(model.features(x))


def focal_loss(logits: torch.Tensor, target: torch.Tensor, *, gamma: float = 2.0) -> torch.Tensor:
    """Binary focal loss (Lin et al. 2017) over independent sigmoid output units."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy(p, target, reduction="none")
    p_t = p * target + (1 - p) * (1 - target)
    loss = ((1 - p_t) ** gamma) * ce
    return loss.mean()


def logit_adjusted_loss(logits: torch.Tensor, target: torch.Tensor, class_priors: torch.Tensor,
                         *, tau: float = 1.0) -> torch.Tensor:
    """Logit adjustment (Menon et al. 2021): shift logits by tau*log(prior) before BCE."""
    adjusted = logits + tau * torch.log(class_priors.clamp_min(1e-6))
    p = torch.sigmoid(adjusted)
    return F.binary_cross_entropy(p, target)


def compute_loss(model, x: torch.Tensor, target: torch.Tensor, cfg, *, class_priors=None) -> torch.Tensor:
    """Dispatch on ``imbalance.loss``: bce | focal | logit_adjusted."""
    method = cfg.get("loss", "focal")
    logits = _logits(model, x)
    if method == "bce":
        return F.binary_cross_entropy(torch.sigmoid(logits), target)
    if method == "focal":
        return focal_loss(logits, target, gamma=float(cfg.get("focal_gamma", 2.0)))
    if method == "logit_adjusted":
        if class_priors is None:
            raise ValueError("logit_adjusted loss requires class_priors")
        return logit_adjusted_loss(
            logits, target, class_priors, tau=float(cfg.get("logit_adjustment_tau", 1.0))
        )
    raise ValueError(f"Unknown imbalance.loss: {method}")
