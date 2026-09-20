"""E2 baseline reimplementations (PRD Section 5).

Four running baselines, each granted the labeled budget its source paper
assumes (the asymmetry is stated in the comparison-table caption -- it makes
ClusterFed's zero-label setting look harder, which is the point):

  B1  zhao_dynamic_threshold   Zhao et al. (IEEE IoTJ 2022) core rule:
      server holds a small labeled set; global model pseudo-labels client
      data each round with a dynamically decreasing confidence threshold.
      Budget: 2000 labeled flows at the server.
  B3  cbafed_class_balanced    CBAFed (CVPR 2023) core rule: partially
      labeled clients; per-class adaptive thresholds from the predicted
      class distribution. Budget: 5% labeled rows per client.
  B4  fedmse_autoencoder       FedMSE (Computers & Security 2025) style:
      federated autoencoder trained on clients' normal traffic;
      reconstruction-error threshold. Budget: each client's benign rows
      (the "clean pre-attack period" deployment assumption).
  B5  fedups_uncertainty       FedUPS (EAAI 2025) style: like B1 but
      pseudo-label selection additionally requires low MC-Dropout
      uncertainty. Budget: same 2000-flow labeled server set.

Shared protocol with ClusterFed (identical where the baselines permit):
same partition, same hybrid-selected 32 features, same MLP backbone, same
200 rounds + FedAvg, same unsupervised best-train-acc model selection, same
per-node test splits, evaluation against true labels.
"""
from __future__ import annotations

import logging

import numpy as np
import torch

from .federated.client import local_train, make_loader
from .federated.model import build_model, enable_mc_dropout
from .federated.server import fedavg

logger = logging.getLogger("clusterfed.baselines")


# ---- shared helpers -------------------------------------------------------------

def make_baseline_splits(nodes, feature_idx, fed_cfg, *, seed: int):
    """Same permutation/test-split protocol as federated.train.build_splits,
    but exposing true train labels + names so baselines can draw their own
    labeled budgets."""
    test_split = float(fed_cfg.get("test_split", 0.2))
    rng = np.random.default_rng(seed)
    splits = {}
    for node_id in sorted(nodes):
        node = nodes[node_id]
        X = node.X if feature_idx is None else node.X[:, feature_idx]
        n = len(X)
        idx = rng.permutation(n)
        n_test = max(1, int(round(test_split * n)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        splits[node_id] = {
            "row_label": node.row_label,
            "X_train": X[train_idx], "y_train_true": node.y_binary[train_idx],
            "X_test": X[test_idx], "y_test_true": node.y_binary[test_idx],
        }
    return splits


@torch.no_grad()
def _probs(model, X):
    model.eval()
    return model(torch.tensor(np.asarray(X), dtype=torch.float32)).cpu().numpy()


def _train_on(model, X, y, fed_cfg, *, seed, epochs=None):
    cfg = dict(fed_cfg)
    if epochs is not None:
        cfg["local_epochs"] = epochs
    state, n = local_train(model, X, y, cfg, seed=seed)
    return state, n


def _baseline_val_acc(global_model, splits):
    """Pooled accuracy against TRUE held-out labels, so a baseline's trajectory
    can be priced in attainment cost on exactly the same footing as ClusterFed's
    (see federated.train.run_federated). Evaluation only -- never fed back."""
    correct = total = 0
    for node_id in sorted(splits):
        s = splits[node_id]
        X, y = s["X_test"], s["y_test_true"]
        if len(X) == 0:
            continue
        pred = _probs(global_model, X).argmax(axis=1)
        correct += int((pred == np.asarray(y)).sum())
        total += len(y)
    return float(correct / total) if total else float("nan")


def _fed_loop_with_selection(global_model, round_fn, splits, rounds, *, eval_every=1):
    """Generic FedAvg loop with the same unsupervised best-train-acc selection
    used by ClusterFed. ``round_fn(rnd, global_model) -> list[(state, n)]``.
    Train accuracy is measured on whatever (X, y) pairs round_fn trained on,
    supplied via the closure's ``train_pool`` list (rebuilt each round)."""
    best_acc, best_state, best_round = -1.0, None, -1
    history = {"rounds": [], "train_acc": [], "val_acc": []}
    for rnd in range(1, rounds + 1):
        client_states, train_pool = round_fn(rnd, global_model)
        if not client_states:
            continue
        global_model.load_state_dict(fedavg(client_states))
        if rnd % eval_every == 0 or rnd == rounds:
            accs, weights = [], []
            for X, y in train_pool:
                if len(X) == 0:
                    continue
                pred = _probs(global_model, X).argmax(axis=1)
                accs.append(float((pred == np.asarray(y)).mean()))
                weights.append(len(X))
            if accs:
                acc = float(np.average(accs, weights=weights))
                history["rounds"].append(rnd)
                history["train_acc"].append(acc)
                history["val_acc"].append(_baseline_val_acc(global_model, splits))
                if acc > best_acc:
                    best_acc, best_round = acc, rnd
                    best_state = {k: v.detach().cpu().clone()
                                  for k, v in global_model.state_dict().items()}
    if best_state is not None:
        global_model.load_state_dict(best_state)
    return global_model, best_round, history


def _stratified_labeled_budget(splits, n_total, *, seed):
    """Server-side labeled set: stratified across nodes and classes."""
    rng = np.random.default_rng(seed + 777)
    Xs, ys = [], []
    per_node = max(2, n_total // len(splits))
    for node_id in sorted(splits):
        s = splits[node_id]
        for cls in (0, 1):
            cls_idx = np.where(s["y_train_true"] == cls)[0]
            k = min(len(cls_idx), max(1, per_node // 2))
            if k:
                pick = rng.choice(cls_idx, size=k, replace=False)
                Xs.append(s["X_train"][pick])
                ys.append(np.full(k, cls))
    return np.vstack(Xs), np.concatenate(ys)


# ---- B1: Zhao-style dynamic-confidence-threshold pseudo-labeling ---------------

def run_zhao_dynamic_threshold(splits, model_cfg, fed_cfg, *, seed, labeled_budget=2000):
    input_dim = next(iter(splits.values()))["X_train"].shape[1]
    model = build_model(input_dim, model_cfg)
    X_lab, y_lab = _stratified_labeled_budget(splits, labeled_budget, seed=seed)
    state, _ = _train_on(model, X_lab, y_lab, fed_cfg, seed=seed, epochs=30)
    model.load_state_dict(state)
    rounds = int(fed_cfg.get("communication_rounds", 200))

    def round_fn(rnd, gm):
        tau = max(0.60, 0.95 - 0.002 * rnd)   # dynamically decreasing threshold
        states, pool = [], []
        for node_id in sorted(splits):
            s = splits[node_id]
            p = _probs(gm, s["X_train"])
            conf, pseudo = p.max(axis=1), p.argmax(axis=1)
            keep = conf >= tau
            if keep.sum() < 2 or np.unique(pseudo[keep]).size < 2:
                continue
            st, n = _train_on(gm, s["X_train"][keep], pseudo[keep], fed_cfg, seed=seed + rnd)
            states.append((st, n))
            pool.append((s["X_train"][keep], pseudo[keep]))
        # Server anchor update on its labeled set (Zhao's labeled supervision).
        st, n = _train_on(gm, X_lab, y_lab, fed_cfg, seed=seed + rnd)
        states.append((st, n))
        pool.append((X_lab, y_lab))
        return states, pool

    return _fed_loop_with_selection(model, round_fn, splits, rounds)


# ---- B3: CBAFed-style class-balanced adaptive pseudo-labeling -------------------

def run_cbafed_class_balanced(splits, model_cfg, fed_cfg, *, seed, labeled_frac=0.05):
    input_dim = next(iter(splits.values()))["X_train"].shape[1]
    model = build_model(input_dim, model_cfg)
    rng = np.random.default_rng(seed + 555)
    labeled = {}
    for node_id in sorted(splits):
        s = splits[node_id]
        n = len(s["X_train"])
        k = max(2, int(labeled_frac * n))
        pick = rng.choice(n, size=min(k, n), replace=False)
        labeled[node_id] = pick
    rounds = int(fed_cfg.get("communication_rounds", 200))

    def round_fn(rnd, gm):
        states, pool = [], []
        for node_id in sorted(splits):
            s = splits[node_id]
            lab_idx = labeled[node_id]
            p = _probs(gm, s["X_train"])
            conf, pseudo = p.max(axis=1), p.argmax(axis=1)
            # CBAFed principle: per-class threshold lowered for rarely-predicted classes.
            counts = np.bincount(pseudo, minlength=2).astype(float)
            share = counts / max(counts.sum(), 1)
            thresholds = 0.6 + 0.35 * share   # majority-predicted class needs higher conf
            keep = conf >= thresholds[pseudo]
            keep[lab_idx] = True
            y_mix = pseudo.copy()
            y_mix[lab_idx] = s["y_train_true"][lab_idx]
            if keep.sum() < 2 or np.unique(y_mix[keep]).size < 2:
                keep = np.zeros(len(y_mix), dtype=bool)
                keep[lab_idx] = True
            st, n = _train_on(gm, s["X_train"][keep], y_mix[keep], fed_cfg, seed=seed + rnd)
            states.append((st, n))
            pool.append((s["X_train"][keep], y_mix[keep]))
        return states, pool

    return _fed_loop_with_selection(model, round_fn, splits, rounds)


# ---- B4: FedMSE-style federated autoencoder anomaly detection -------------------

class _AE(torch.nn.Module):
    def __init__(self, d, h=16):
        super().__init__()
        self.enc = torch.nn.Sequential(torch.nn.Linear(d, h), torch.nn.Tanh())
        self.dec = torch.nn.Linear(h, d)

    def forward(self, x):
        return self.dec(self.enc(x))


def run_fedmse_autoencoder(splits, fed_cfg, *, seed, rounds=50, threshold_q=0.95):
    """Federated AE on each client's benign (normal) training rows; a test row
    is flagged attack when its reconstruction error exceeds the per-client
    q-th percentile of training-benign error."""
    input_dim = next(iter(splits.values()))["X_train"].shape[1]
    torch.manual_seed(seed)
    ae = _AE(input_dim)

    benign_rows = {nid: s["X_train"][s["y_train_true"] == 0] for nid, s in splits.items()}

    for rnd in range(1, rounds + 1):
        states = []
        for node_id in sorted(splits):
            Xb = benign_rows[node_id]
            if len(Xb) < 2:
                continue
            local = _AE(input_dim)
            local.load_state_dict(ae.state_dict())
            opt = torch.optim.Adam(local.parameters(), lr=1e-3)
            Xt = torch.tensor(Xb, dtype=torch.float32)
            gen = torch.Generator().manual_seed(seed + rnd)
            for start in range(0, len(Xt), 256):
                idx = torch.randperm(len(Xt), generator=gen)[start:start + 256]
                opt.zero_grad()
                loss = ((local(Xt[idx]) - Xt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            states.append(({k: v.detach().cpu().clone() for k, v in local.state_dict().items()},
                            len(Xb)))
        ae.load_state_dict(fedavg(states))

    # Per-client threshold from benign training error.
    predictions = {}
    with torch.no_grad():
        for node_id in sorted(splits):
            s = splits[node_id]
            Xb = torch.tensor(benign_rows[node_id], dtype=torch.float32)
            err_b = ((ae(Xb) - Xb) ** 2).mean(dim=1).numpy()
            thresh = float(np.quantile(err_b, threshold_q)) if len(err_b) else 0.0
            Xt = torch.tensor(s["X_test"], dtype=torch.float32)
            err_t = ((ae(Xt) - Xt) ** 2).mean(dim=1).numpy()
            predictions[node_id] = (err_t > thresh).astype(int)
    return predictions


# ---- B5: FedUPS-style uncertainty-filtered pseudo-labeling ----------------------

def run_fedups_uncertainty(splits, model_cfg, fed_cfg, *, seed, labeled_budget=2000,
                            mc_passes=10, tau_u=1.0):
    input_dim = next(iter(splits.values()))["X_train"].shape[1]
    model = build_model(input_dim, model_cfg, dropout_p=0.2)
    X_lab, y_lab = _stratified_labeled_budget(splits, labeled_budget, seed=seed)
    state, _ = _train_on(model, X_lab, y_lab, fed_cfg, seed=seed, epochs=30)
    model.load_state_dict(state)
    rounds = int(fed_cfg.get("communication_rounds", 200))

    @torch.no_grad()
    def mc_predict(gm, X):
        enable_mc_dropout(gm)
        Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
        logits = torch.stack([gm.head(gm.features(Xt)) for _ in range(mc_passes)], dim=0)
        probs = torch.sigmoid(logits).mean(dim=0)
        std = logits.std(dim=0)
        cls = probs.argmax(dim=1)
        return (cls.numpy(), probs.max(dim=1).values.numpy(),
                std.gather(1, cls.unsqueeze(1)).squeeze(1).numpy())

    def round_fn(rnd, gm):
        states, pool = [], []
        for node_id in sorted(splits):
            s = splits[node_id]
            pseudo, conf, std = mc_predict(gm, s["X_train"])
            keep = (conf >= 0.8) & (std <= tau_u)
            if keep.sum() < 2 or np.unique(pseudo[keep]).size < 2:
                continue
            st, n = _train_on(gm, s["X_train"][keep], pseudo[keep], fed_cfg, seed=seed + rnd)
            states.append((st, n))
            pool.append((s["X_train"][keep], pseudo[keep]))
        st, n = _train_on(gm, X_lab, y_lab, fed_cfg, seed=seed + rnd)
        states.append((st, n))
        pool.append((X_lab, y_lab))
        return states, pool

    return _fed_loop_with_selection(model, round_fn, splits, rounds)


# ---- evaluation shared by all baselines -----------------------------------------

def evaluate_model_baseline(model, splits):
    """Per-node P/R/Acc/F1 of a classifier baseline on the shared test splits."""
    from .labeling.evaluate import score

    rows = []
    for node_id in sorted(splits):
        s = splits[node_id]
        pred = _probs(model, s["X_test"]).argmax(axis=1)
        m = score(s["y_test_true"], pred)
        rows.append({"node": node_id, "category": s["row_label"],
                     "P": m["precision"], "R": m["recall"], "Acc": m["accuracy"], "F1": m["f1"]})
    return rows


def evaluate_predictions_baseline(predictions, splits):
    """Per-node metrics for baselines that emit per-node predictions directly (B4)."""
    from .labeling.evaluate import score

    rows = []
    for node_id in sorted(splits):
        s = splits[node_id]
        m = score(s["y_test_true"], predictions[node_id])
        rows.append({"node": node_id, "category": s["row_label"],
                     "P": m["precision"], "R": m["recall"], "Acc": m["accuracy"], "F1": m["f1"]})
    return rows
