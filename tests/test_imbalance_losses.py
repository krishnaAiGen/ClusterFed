"""Tests for labeling/imbalance_losses.py: Stage 3's second imbalance defense (PRD 4.4)."""
import torch

from clusterfed.federated.model import build_model
from clusterfed.labeling.imbalance_losses import compute_loss, focal_loss, logit_adjusted_loss


def _toy_batch(seed=0, n=16, d=5):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    y = torch.zeros(n, 2)
    y[:, 0] = 1.0
    return x, y


def test_focal_loss_is_finite_and_decreases_confident_correct_predictions():
    logits_confident_correct = torch.tensor([[10.0, -10.0]])
    logits_confident_wrong = torch.tensor([[-10.0, 10.0]])
    target = torch.tensor([[1.0, 0.0]])
    loss_correct = focal_loss(logits_confident_correct, target)
    loss_wrong = focal_loss(logits_confident_wrong, target)
    assert torch.isfinite(loss_correct) and torch.isfinite(loss_wrong)
    assert loss_correct < loss_wrong


def test_logit_adjusted_loss_shifts_by_prior():
    logits = torch.tensor([[0.0, 0.0]])
    target = torch.tensor([[1.0, 0.0]])
    even_priors = torch.tensor([0.5, 0.5])
    skewed_priors = torch.tensor([0.9, 0.1])
    loss_even = logit_adjusted_loss(logits, target, even_priors, tau=1.0)
    loss_skewed = logit_adjusted_loss(logits, target, skewed_priors, tau=1.0)
    assert torch.isfinite(loss_even) and torch.isfinite(loss_skewed)
    assert not torch.allclose(loss_even, loss_skewed)


def test_compute_loss_dispatch_and_backprop():
    model = build_model(5, {"hidden_layers": [8, 4], "output_dim": 2}, dropout_p=0.0)
    x, y = _toy_batch()
    priors = torch.tensor([0.5, 0.5])
    for method in ("bce", "focal", "logit_adjusted"):
        model.zero_grad()
        loss = compute_loss(model, x, y, {"loss": method, "focal_gamma": 2.0, "logit_adjustment_tau": 1.0},
                             class_priors=priors)
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert len(grads) > 0
