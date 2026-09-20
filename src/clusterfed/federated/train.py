"""Federated training loop: Table II analog and the convergence curve.

200 rounds of: broadcast global model -> each node trains locally on its own
refined-pseudo-labels (Stages 0-4) -> FedAvg aggregate -> evaluate the global
model per attack category on held-out test sets. Unchanged from v1 except the
training label source is the v2 refinement pipeline's output instead of raw
PCA(2D)+k-means, per PRD 4.5 ("the refined labels feed federated training
exactly as in v1").

Evaluation is always against TRUE binary labels (detection performance), even
though training uses pseudo-labels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from .client import local_train
from .model import build_model
from .server import aggregate, compress_delta, uplink_bytes

logger = logging.getLogger("clusterfed.federated")


@dataclass
class NodeSplit:
    node_id: int
    row_label: str
    X_train: np.ndarray
    y_train: np.ndarray          # training target (v2-refined pseudo-labels, or true for the upper bound)
    X_test: np.ndarray
    y_test_true: np.ndarray      # TRUE binary labels for evaluation
    y_test_name: np.ndarray = None   # TRUE attack-type names for the test rows (E4 per-type breakdown)
    n_train: int = 0


@dataclass
class History:
    rounds: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    label_source: str = "pseudo"
    selected_round: int = -1     # round whose weights the returned model carries (best_train_acc selection)
    zeta_rounds: list = field(default_factory=list)   # E6: rounds where zeta^2 was measured
    zeta_sq: list = field(default_factory=list)       # E6: empirical gradient-diversity values
    uplink_bytes_per_round: int = 0        # ICC: per-client uplink payload (compression-dependent)
    downlink_bytes_per_round: int = 0      # ICC: dense fp32 broadcast
    client_fraction: float = 1.0           # ICC: fraction of clients sampled per round
    participants_per_round: list = field(default_factory=list)
    total_train_rows: int = 0        # rows actually trained on, summed over clients


def build_splits(nodes, feature_idx, label_source, pseudo_labels, fed_cfg, *, seed: int,
                 seed_masks=None):
    """Per-node train/test split. ``pseudo_labels`` maps node_id -> refined binary
    labels (``pipeline.compute_v2_labels`` output's ``.y_refined``); required
    when ``label_source == "pseudo"``.

    With ``fed_cfg.train_on == "seed"`` (v2.1 default) and ``seed_masks``
    provided, each client trains only on its curated seed-set rows (PRD 4.4:
    ambiguous points are held out, not mislabeled). Evaluation always uses the
    full held-out test split with TRUE labels, so the test set is unaffected.
    """
    test_split = float(fed_cfg.get("test_split", 0.2))
    train_on = str(fed_cfg.get("train_on", "seed"))
    rng = np.random.default_rng(seed)
    splits: dict[int, NodeSplit] = {}
    for node_id in sorted(nodes):
        node = nodes[node_id]
        X = node.X if feature_idx is None else node.X[:, feature_idx]
        n = len(X)
        idx = rng.permutation(n)
        n_test = max(1, int(round(test_split * n)))
        test_idx, train_idx = idx[:n_test], idx[n_test:]

        if label_source == "true":
            y_train_full = node.y_binary
        elif label_source == "pseudo":
            y_train_full = pseudo_labels[node_id]
        else:
            raise ValueError(f"label_source must be 'pseudo' or 'true', got {label_source}")

        # Applies to both label sources: with true labels + seed masks this is
        # the "supervised on curated rows" control arm (EXPERIMENT_LOG Entry 18).
        if (train_on == "seed"
                and seed_masks is not None and seed_masks.get(node_id) is not None):
            mask = np.asarray(seed_masks[node_id], dtype=bool)
            keep = mask[train_idx]
            if keep.sum() >= 2:  # degenerate seed sets fall back to all rows
                train_idx = train_idx[keep]
        elif train_on in ("random", "random_stratified"):
            # Size-matched control for the curation experiment. Curation both
            # SELECTS rows and SHRINKS the training set; this arm shrinks by the
            # same amount without selecting, so the two effects can be told
            # apart. By default it matches |S_k| per client exactly rather than
            # applying one global fraction, because seed-set sizes differ widely
            # across clients and a global fraction would confound the comparison.
            frac = fed_cfg.get("random_fraction", None)
            if frac is not None:
                n_keep = int(round(float(frac) * len(train_idx)))
            elif seed_masks is not None and seed_masks.get(node_id) is not None:
                m = np.asarray(seed_masks[node_id], dtype=bool)
                n_keep = int(m[train_idx].sum())
            else:
                n_keep = len(train_idx)
            n_keep = max(2, min(n_keep, len(train_idx)))
            rr = np.random.default_rng(seed * 31337 + node_id)
            y_pool = np.asarray(y_train_full)[train_idx]
            if train_on == "random_stratified":
                # Preserve the seed set's class balance, isolating "which rows"
                # from "what class mix".
                picks = []
                for cls in (0, 1):
                    idx_c = np.flatnonzero(y_pool == cls)
                    if len(idx_c) == 0:
                        continue
                    k = max(1, int(round(n_keep * len(idx_c) / len(y_pool))))
                    picks.append(rr.choice(idx_c, size=min(k, len(idx_c)), replace=False))
                sel = np.concatenate(picks) if picks else np.arange(len(train_idx))
            else:
                sel = rr.choice(len(train_idx), size=n_keep, replace=False)
            train_idx = train_idx[np.sort(sel)]

        splits[node_id] = NodeSplit(
            node_id=node_id,
            row_label=node.row_label,
            X_train=X[train_idx],
            y_train=np.asarray(y_train_full)[train_idx],
            X_test=X[test_idx],
            y_test_true=node.y_binary[test_idx],
            y_test_name=node.y_true_name[test_idx],
            n_train=len(train_idx),
        )
    return splits


@torch.no_grad()
def _predict(model, X: np.ndarray) -> np.ndarray:
    model.eval()
    xb = torch.tensor(np.asarray(X), dtype=torch.float32)
    out = model(xb)
    return out.argmax(dim=1).cpu().numpy()


def _accuracy(model, X, y) -> float:
    if len(X) == 0:
        return float("nan")
    pred = _predict(model, X)
    return float((pred == np.asarray(y)).mean())


def run_federated(nodes, feature_idx, label_source, pseudo_labels, cfg, *, seed: int = 42,
                  seed_masks=None):
    """Run the FedAvg loop. Returns ``(history, global_model, splits)``."""
    fed_cfg = cfg.federated
    splits = build_splits(nodes, feature_idx, label_source, pseudo_labels, fed_cfg, seed=seed,
                          seed_masks=seed_masks)

    sample_X = next(iter(splits.values())).X_train
    input_dim = sample_X.shape[1]
    global_model = build_model(input_dim, cfg.model)  # dropout_p=0.0: identical shape to v1

    rounds = int(fed_cfg.get("communication_rounds", 200))
    eval_every = int(fed_cfg.get("eval_every", 1))
    model_selection = str(fed_cfg.get("model_selection", "best_train_acc"))
    history = History(label_source=label_source)

    history.total_train_rows = int(sum(s.n_train for s in splits.values()))
    pool_train_X = np.vstack([s.X_train for s in splits.values()])
    pool_train_y = np.concatenate([s.y_train for s in splits.values()])
    pool_test_X = np.vstack([s.X_test for s in splits.values()])
    pool_test_y = np.concatenate([s.y_test_true for s in splits.values()])

    ema_decay = float(fed_cfg.get("server_ema_decay", 0.95))
    ema_state = None
    eval_model = build_model(input_dim, cfg.model)

    # E8 robustness knobs (all default-off).
    agg_method = str(fed_cfg.get("aggregation_method", "fedavg"))
    poison_frac = float(fed_cfg.get("poison_fraction", 0.0))
    dp_sigma = float(fed_cfg.get("dp_sigma", 0.0))
    poisoned_ids: set = set()
    if poison_frac > 0:
        ids = sorted(splits)
        n_poison = int(round(poison_frac * len(ids)))
        poison_rng = np.random.default_rng(seed + 999)
        poisoned_ids = set(poison_rng.choice(ids, size=n_poison, replace=False).tolist())
        for nid in poisoned_ids:
            splits[nid].y_train = 1 - splits[nid].y_train   # label-flip attack
        logger.info("E8: poisoned clients (label-flip): %s", sorted(poisoned_ids))
    # E6: empirical gradient-diversity (zeta) measurement.
    measure_zeta = bool(fed_cfg.get("measure_zeta", False))
    zeta_every = int(fed_cfg.get("zeta_every", 20))

    # ICC Tier C: bytes-per-round knobs. client_fraction < 1 samples a subset of
    # clients each round (partial participation); compression shrinks the uplink
    # payload of each client's update. Both default to the dense, full-
    # participation behaviour used by every earlier experiment.
    client_fraction = float(fed_cfg.get("client_fraction", 1.0))
    compression = str(fed_cfg.get("compression", "none"))
    topk_fraction = float(fed_cfg.get("topk_fraction", 0.1))
    error_feedback = bool(fed_cfg.get("error_feedback", False))
    residuals: dict = {}
    n_params = sum(p.numel() for p in global_model.parameters())
    history.uplink_bytes_per_round = uplink_bytes(n_params, compression, topk_fraction)
    history.downlink_bytes_per_round = 4 * n_params
    history.client_fraction = client_fraction

    best_train_acc, best_state, best_round = -1.0, None, -1
    for rnd in range(1, rounds + 1):
        prev_state = ({k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
                      if measure_zeta and rnd % zeta_every == 0 else None)
        eligible = [nid for nid in sorted(splits) if splits[nid].n_train > 0]
        if client_fraction < 1.0 and len(eligible) > 1:
            m = max(1, int(round(client_fraction * len(eligible))))
            rr = np.random.default_rng(seed * 100003 + rnd)
            participants = sorted(rr.choice(eligible, size=m, replace=False).tolist())
        else:
            participants = eligible
        history.participants_per_round.append(len(participants))

        gstate = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        client_states = []
        for node_id in participants:
            s = splits[node_id]
            state, n_k = local_train(global_model, s.X_train, s.y_train, fed_cfg, seed=seed + rnd)
            if compression != "none":
                # Compress the update, not the weights: the server already holds
                # gstate, so only the delta crosses the link.
                raw = {k: state[k] - gstate[k] for k in state}
                if error_feedback:
                    # Deep-gradient-compression style: carry the coordinates
                    # this client failed to send into its next update instead of
                    # discarding them, so no gradient information is lost, only
                    # delayed.
                    res = residuals.get(node_id)
                    if res is not None:
                        raw = {k: raw[k] + res[k] for k in raw}
                delta = compress_delta(raw, compression, topk_fraction)
                if error_feedback:
                    residuals[node_id] = {k: raw[k] - delta[k] for k in raw}
                state = {k: gstate[k] + delta[k] for k in state}
            if dp_sigma > 0:   # E8 DP arm: Gaussian noise on each client update
                gen = torch.Generator().manual_seed(seed + rnd * 1000 + node_id)
                state = {k: v + torch.normal(0.0, dp_sigma, size=v.shape, generator=gen)
                         if v.dtype.is_floating_point else v
                         for k, v in state.items()}
            client_states.append((state, n_k))

        if prev_state is not None:
            # zeta^2 ~= weighted mean ||u_k - u_bar||^2 / ||u_bar||^2 over client updates.
            updates = []
            for state, n_k in client_states:
                u = torch.cat([(state[k] - prev_state[k]).flatten().to(torch.float64)
                               for k in prev_state])
                updates.append((u, n_k))
            total_n = sum(n for _, n in updates)
            u_bar = sum(u * (n / total_n) for u, n in updates)
            denom = float((u_bar ** 2).sum()) or 1e-12
            zeta_sq = float(sum((n / total_n) * float(((u - u_bar) ** 2).sum())
                                for u, n in updates) / denom)
            history.zeta_rounds.append(rnd)
            history.zeta_sq.append(zeta_sq)
            logger.info("[E6] round %d empirical zeta^2 = %.3f", rnd, zeta_sq)

        global_model.load_state_dict(aggregate(client_states, agg_method))

        # Server-side EMA of the global weights: FedAvg on strongly non-IID
        # clients oscillates round to round; the EMA (~1/(1-decay)-round
        # horizon) tracks the basin rather than the swing. decay<=0 disables
        # (v1 behavior: evaluate the raw round-R model).
        if ema_decay > 0:
            cur = global_model.state_dict()
            if ema_state is None:
                ema_state = {k: v.detach().cpu().clone().to(torch.float64) for k, v in cur.items()}
            else:
                for k in ema_state:
                    ema_state[k].mul_(ema_decay).add_(cur[k].detach().cpu().to(torch.float64),
                                                       alpha=1.0 - ema_decay)
            eval_model.load_state_dict({k: v.to(cur[k].dtype) for k, v in ema_state.items()})
        else:
            eval_model.load_state_dict(global_model.state_dict())

        if rnd % eval_every == 0 or rnd == rounds:
            tr = _accuracy(eval_model, pool_train_X, pool_train_y)
            va = _accuracy(eval_model, pool_test_X, pool_test_y)
            history.rounds.append(rnd)
            history.train_acc.append(tr)
            history.val_acc.append(va)
            # Unsupervised model selection: track the (smoothed) round with the
            # best POOLED TRAIN accuracy -- measured only against the clients'
            # own (pseudo-)training labels, so no test data or ground truth
            # informs the choice.
            if tr > best_train_acc:
                best_train_acc = tr
                best_round = rnd
                best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            if rnd == 1 or rnd % max(1, rounds // 10) == 0 or rnd == rounds:
                logger.info("[%s] round %3d/%d  train_acc=%.3f  val_acc=%.3f",
                            label_source, rnd, rounds, tr, va)

    if model_selection == "best_train_acc" and best_state is not None:
        global_model.load_state_dict(best_state)
        history.selected_round = best_round
        logger.info("[%s] selected round %d (train_acc=%.3f) as the final global model",
                    label_source, best_round, best_train_acc)
    else:
        history.selected_round = rounds
        if ema_decay > 0 and ema_state is not None:
            global_model.load_state_dict(eval_model.state_dict())
    return history, global_model, splits


def run_table2(nodes, feature_idx, pseudo_labels, cfg, *, seed: int = 42, seed_masks=None):
    """Train federated on v2-refined pseudo-labels; report per-category P/R/Acc/F1 of
    the final global model on each category's held-out test set. Returns
    ``(df, history, model)``.
    """
    import pandas as pd
    from ..labeling.evaluate import score

    history, model, splits = run_federated(nodes, feature_idx, "pseudo", pseudo_labels, cfg, seed=seed,
                                           seed_masks=seed_masks)
    rows = []
    for node_id in sorted(splits):
        s = splits[node_id]
        pred = _predict(model, s.X_test)
        m = score(s.y_test_true, pred)
        rows.append({
            "node": node_id, "category": s.row_label,
            "P": m["precision"], "R": m["recall"], "Acc": m["accuracy"], "F1": m["f1"],
        })
        logger.info("Table II %-26s F1=%.2f", s.row_label, m["f1"])
    df = pd.DataFrame(rows)
    avg = {c: df[c].mean() for c in ("P", "R", "Acc", "F1")}
    avg.update({"node": "", "category": "Average"})
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df, history, model
