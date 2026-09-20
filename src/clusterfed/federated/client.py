"""Local client training step (PRD 9.10).

Each client trains the broadcast global model on its own pseudo-labeled data for
``local_epochs`` with Adam + BCE, then returns its updated weights and sample
count so the server can weight the FedAvg aggregation by n_k.
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import ClusterFedNet


def _onehot(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    """BCE over the 2 sigmoid output units needs a one-hot target."""
    y = np.asarray(y, dtype=np.int64)
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return torch.from_numpy(oh)


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, *, shuffle: bool, seed: int):
    ds = TensorDataset(
        torch.tensor(np.asarray(X), dtype=torch.float32),
        _onehot(y),
    )
    gen = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=gen)


def local_train(global_model: ClusterFedNet, X: np.ndarray, y: np.ndarray, cfg, *, seed: int = 42):
    """Train a copy of ``global_model`` on (X, y). Returns ``(state_dict, n_samples)``.

    ``cfg`` is the ``federated`` config section. With ``fedprox_mu > 0`` the
    FedProx proximal term (mu/2)*||w - w_global||^2 is added to the local loss
    (PRD 4.5 / ablation A7: the non-IID-robust aggregator arm) -- it anchors
    each client's update to the broadcast global weights, countering the
    client drift that makes plain FedAvg degrade on one-attack-per-client
    partitions.
    """
    model = copy.deepcopy(global_model)
    model.train()
    lr = float(cfg.get("learning_rate", 1e-3))
    epochs = int(cfg.get("local_epochs", 1))
    batch_size = int(cfg.get("batch_size", 64))
    mu = float(cfg.get("fedprox_mu", 0.0))

    global_params = None
    if mu > 0:
        global_params = [p.detach().clone() for p in global_model.parameters()]

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loader = make_loader(X, y, batch_size, shuffle=True, seed=seed)

    # Inverse-frequency class weighting, computed per client from its own label
    # counts. Default "none" reproduces the unweighted BCE every earlier
    # experiment used; "balanced" is the imbalance-corrected reference the
    # supervised arm needs, since per-client class priors on this partition
    # range from 1:1 to 1:4 in both directions.
    weighting = str(cfg.get("loss_weighting", "none"))
    if weighting == "balanced":
        y_arr = np.asarray(y)
        n_pos = max(1, int((y_arr == 1).sum()))
        n_neg = max(1, int((y_arr == 0).sum()))
        n_tot = n_pos + n_neg
        w_pos = 0.5 * n_tot / n_pos
        w_neg = 0.5 * n_tot / n_neg
        criterion = nn.BCELoss(reduction="none")
    elif weighting == "none":
        criterion = nn.BCELoss()
    else:
        raise ValueError(f"unknown loss_weighting {weighting!r}")

    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            if weighting == "balanced":
                # yb is one-hot over 2 classes; column 1 marks the attack class.
                sw = torch.where(yb[:, 1] > 0.5, w_pos, w_neg).unsqueeze(1)
                loss = (criterion(out, yb) * sw).mean()
            else:
                loss = criterion(out, yb)
            if mu > 0:
                prox = sum(((p - g) ** 2).sum() for p, g in zip(model.parameters(), global_params))
                loss = loss + (mu / 2.0) * prox
            loss.backward()
            optimizer.step()

    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    return state, int(len(X))
