"""Stage 4 -- iterative self-training refinement with an MC-Dropout co-agreement
filter (PRD 4.5). Orchestrates Stages 2-4 end to end for one federated client.

Per client, per refinement iteration r = 1..R:
  1. Train a local MLP (MC-Dropout enabled) on the current seed set.
  2. Predict on held-out low-confidence points; estimate uncertainty via
     MC-Dropout (T stochastic passes).
  3. Promote a held-out point into the seed set only when the model prediction
     and the Stage-2 cluster assignment AGREE and predictive uncertainty is
     below tau_u -- a co-agreement filter that blocks confirmation bias.
  4. Points where model and cluster disagree with high model confidence are
     RELABELED to the model's prediction (this is where Bot / DoS slowloris-
     style failures are meant to recover, per PRD 4.5).

The refined labels then feed federated training exactly as in v1 (unchanged
FedAvg/MLP in ``federated/``).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import torch

from ..federated.model import ClusterFedNet, build_model, enable_mc_dropout
from ..theory.convergence import estimate_label_noise_delta
from .clustering import bootstrap_labels
from .imbalance_losses import compute_loss
from .threshold import select_seed_set

logger = logging.getLogger("clusterfed.refinement")


@dataclass
class RefinementResult:
    y_refined: np.ndarray                          # final binary pseudo-labels (0 benign, 1 malicious)
    y_cluster_raw: np.ndarray                       # Stage-2 raw cluster labels, pre-refinement (the "L" bound)
    seed_mask: np.ndarray = None                    # final curated seed set (PRD 4.4: ambiguous points held out)
    delta_r: list = field(default_factory=list)     # empirical label-noise rate vs ground truth, per iteration (diagnostic only, never trained on)
    seed_set_size_r: list = field(default_factory=list)
    uncertainty_diagnostics: list = field(default_factory=list)  # per-iteration dict: is pred_std informative about agree/disagree?
    seed_purity: float = float("nan")               # diagnostic: label accuracy inside the final seed set (vs ground truth)
    overall_purity: float = float("nan")            # diagnostic: label accuracy over all rows (vs ground truth)


def _uncertainty_diagnostic(agree: np.ndarray, pred_std: np.ndarray) -> dict:
    """Does MC-Dropout logit-std actually discriminate agreeing from disagreeing
    points? AUROC of "predict disagreement from higher pred_std": ~0.5 means
    uncertainty carries no signal about whether a point will disagree with the
    cluster (tau_u tuning would be pointless); well above 0.5 means it does
    (tau_u is worth calibrating properly).
    """
    from sklearn.metrics import roc_auc_score

    disagree = (~agree).astype(int)
    diag = {
        "n_agree": int(agree.sum()),
        "n_disagree": int((~agree).sum()),
        "agree_std_median": float(np.median(pred_std[agree])) if agree.any() else float("nan"),
        "disagree_std_median": float(np.median(pred_std[~agree])) if (~agree).any() else float("nan"),
    }
    if disagree.min() != disagree.max():
        diag["auroc_std_predicts_disagreement"] = float(roc_auc_score(disagree, pred_std))
    else:
        diag["auroc_std_predicts_disagreement"] = float("nan")  # only one group present this round
    return diag


def _make_onehot(y: np.ndarray, num_classes: int = 2) -> torch.Tensor:
    y = np.asarray(y, dtype=np.int64)
    oh = np.zeros((len(y), num_classes), dtype=np.float32)
    oh[np.arange(len(y)), y] = 1.0
    return torch.from_numpy(oh)


def _train_local_mlp(X, y, model_cfg, refinement_cfg, imbalance_cfg, *, seed: int):
    input_dim = X.shape[1]
    model = build_model(input_dim, model_cfg, dropout_p=float(refinement_cfg.get("dropout_p", 0.2)))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=float(refinement_cfg.get("learning_rate", 1e-3)))
    epochs = int(refinement_cfg.get("local_epochs_per_iteration", 5))
    batch_size = int(refinement_cfg.get("batch_size", 64))

    Xt_all = torch.tensor(np.asarray(X), dtype=torch.float32)
    yt_all = _make_onehot(y)
    n = len(X)
    counts = np.bincount(np.asarray(y), minlength=2).astype("float64")
    priors = torch.tensor(counts / max(counts.sum(), 1.0), dtype=torch.float32)

    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=gen)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb, yb = Xt_all[idx], yt_all[idx]
            opt.zero_grad()
            loss = compute_loss(model, xb, yb, imbalance_cfg, class_priors=priors)
            loss.backward()
            opt.step()
    return model


@torch.no_grad()
def _mc_dropout_predict(model: ClusterFedNet, X: np.ndarray, T: int):
    """Return ``(pred_class, pred_conf, pred_std)`` over T stochastic MC-Dropout passes.

    Uncertainty is measured on the pre-sigmoid logits, not the post-sigmoid
    probabilities: once training pushes logits to large magnitudes, sigmoid
    saturates (derivative ~0), so dropout perturbations barely move the
    probability even when the underlying weights differ a lot per pass --
    probability-space std collapses to ~0 almost everywhere and the
    co-agreement filter stops filtering anything. Logits don't saturate, so
    their spread across passes stays a meaningful epistemic-uncertainty signal.
    """
    enable_mc_dropout(model)
    Xt = torch.tensor(np.asarray(X), dtype=torch.float32)
    logits = torch.stack([model.head(model.features(Xt)) for _ in range(T)], dim=0)  # (T, n, 2)
    probs = torch.sigmoid(logits)
    mean_probs = probs.mean(dim=0)
    logit_std = logits.std(dim=0)
    pred_class = mean_probs.argmax(dim=1)
    pred_conf = mean_probs.max(dim=1).values.cpu().numpy()
    pred_std = logit_std.gather(1, pred_class.unsqueeze(1)).squeeze(1).cpu().numpy()
    return pred_class.cpu().numpy(), pred_conf, pred_std


def refine_node_labels(
    X: np.ndarray,
    y_true_name: np.ndarray,
    model_cfg,
    clustering_cfg,
    threshold_cfg,
    imbalance_cfg,
    refinement_cfg,
    *,
    seed: int = 42,
) -> RefinementResult:
    """Run Stages 2-4 for one federated client/node.

    ``X`` is already Stage-0-selected and lives in the Stage-1 representation
    (full reduced-feature space, not the 2D PCA projection). ``y_true_name`` is
    used only (a) to fix cluster orientation via the known-benign point, exactly
    as in v1, and (b) to compute a read-only diagnostic delta_r -- neither use
    ever changes a label directly.
    """
    # Stage 1/2: choose the clustering space (A6: v1-legacy 'pca2d' projects to
    # 2 components first; v2 default 'reduced' clusters in the full selected-
    # feature space), then run the bootstrap (binary or multi-cluster depending
    # on clustering.n_clusters -- see clustering.bootstrap_labels).
    space = str(clustering_cfg.get("clustering_space", "reduced"))
    if space == "pca2d":
        from .reduce import pca_2d
        X_cluster, _ = pca_2d(X, n_components=2, seed=seed)
    else:
        X_cluster = X
    y_cluster, conf = bootstrap_labels(X_cluster, y_true_name, clustering_cfg, seed=seed)
    y_cluster_raw = y_cluster.copy()

    # Stage 3: class-balanced adaptive seed-set selection.
    seed_mask, _ = select_seed_set(y_cluster, conf, threshold_cfg)

    y_true_bin = (np.asarray(y_true_name) != "BENIGN").astype(np.int64)
    y_current = y_cluster.copy()
    result = RefinementResult(y_refined=y_current, y_cluster_raw=y_cluster_raw)

    R = int(refinement_cfg.get("iterations", 3))
    tau_u = float(refinement_cfg.get("uncertainty_threshold_tau_u", 0.15))
    relabel_conf = float(refinement_cfg.get("relabel_confidence_threshold", 0.8))
    mc_T = int(refinement_cfg.get("mc_dropout_passes", 10))
    # Engineering safeguard (ASSUMPTION, not in the PRD): cap how much of the
    # held-out set a single iteration can promote, so refinement is genuinely
    # iterative across R rounds even if tau_u is miscalibrated for this node's
    # logit scale -- otherwise a too-loose tau_u dumps ~everything into the
    # seed set in iteration 1, and R-1 further iterations do nothing.
    max_promote_frac = float(refinement_cfg.get("max_promote_fraction_per_iteration", 1.0 / max(R, 1)))

    held_out_mask = ~seed_mask
    for r in range(1, R + 1):
        if seed_mask.sum() < 2 or np.unique(y_current[seed_mask]).size < 2:
            logger.warning("Refinement r=%d: seed set degenerate (size=%d); stopping early",
                            r, int(seed_mask.sum()))
            break

        model = _train_local_mlp(
            X[seed_mask], y_current[seed_mask], model_cfg, refinement_cfg, imbalance_cfg,
            seed=seed + r,
        )

        held_idx = np.where(held_out_mask)[0]
        if len(held_idx) == 0:
            break
        pred_class, pred_conf, pred_std = _mc_dropout_predict(model, X[held_idx], mc_T)

        cluster_assignment = y_cluster[held_idx]  # Stage-2 assignment never changes
        agree = pred_class == cluster_assignment
        confident_enough = pred_std < tau_u

        promote_candidates = np.where(agree & confident_enough)[0]    # Step 3 candidates
        # Step 4 with asymmetric trust + hard caps (EXPERIMENT_LOG Entries 8, 11):
        # with a strong multi-cluster bootstrap, the local MLP's confident
        # disagreements are usually wrong (it was trained on the cluster's own
        # labels), so correction must be conservative: the model may only
        # override the cluster where the cluster was unsure (bottom-quantile
        # centroid margin), with very high model confidence, and only a tiny
        # capped volume per iteration ranked by model confidence.
        gate_q = float(refinement_cfg.get("relabel_cluster_conf_quantile", 0.25))
        held_conf = conf[held_idx]
        gate_thresh = float(np.quantile(held_conf, gate_q)) if len(held_conf) else 0.0
        cluster_unsure = held_conf <= gate_thresh
        relabel = (~agree) & (pred_conf >= relabel_conf) & cluster_unsure

        relabel_cap_frac = float(refinement_cfg.get("relabel_max_fraction_per_iteration", 0.01))
        relabel_cap = max(0, int(np.floor(relabel_cap_frac * len(held_idx))))
        relabel_candidates = np.where(relabel)[0]
        if len(relabel_candidates) > relabel_cap:
            order = np.argsort(-pred_conf[relabel_candidates])
            relabel_candidates = relabel_candidates[order[:relabel_cap]]
        relabel = np.zeros(len(held_idx), dtype=bool)
        relabel[relabel_candidates] = True

        max_new = max(1, int(np.ceil(max_promote_frac * len(held_idx))))
        if len(promote_candidates) > max_new:
            # Keep only the most-confident (lowest-uncertainty) candidates this round.
            order = np.argsort(pred_std[promote_candidates])
            promote_candidates = promote_candidates[order[:max_new]]

        promote_idx = held_idx[promote_candidates]
        relabel_idx = held_idx[relabel]

        # Diagnostic only (never used to pick labels): is a relabel more often
        # right than the cluster label it replaces, against ground truth?
        old_labels = y_current[relabel_idx]      # == the Stage-2 cluster label here (unchanged so far)
        new_labels = pred_class[relabel]
        true_labels = y_true_bin[relabel_idx]
        diag = _uncertainty_diagnostic(agree, pred_std)
        diag["n_relabeled"] = int(len(relabel_idx))
        diag["relabel_correct_before"] = int((old_labels == true_labels).sum())
        diag["relabel_correct_after"] = int((new_labels == true_labels).sum())
        diag["relabel_net_correctness_gain"] = diag["relabel_correct_after"] - diag["relabel_correct_before"]
        result.uncertainty_diagnostics.append(diag)
        logger.info(
            "Refinement r=%d uncertainty check: agree_std_median=%.4f disagree_std_median=%.4f "
            "auroc(std predicts disagreement)=%.3f (n_agree=%d n_disagree=%d) | relabel "
            "n=%d correct_before=%d correct_after=%d net_gain=%+d",
            r, diag["agree_std_median"], diag["disagree_std_median"],
            diag["auroc_std_predicts_disagreement"], diag["n_agree"], diag["n_disagree"],
            diag["n_relabeled"], diag["relabel_correct_before"], diag["relabel_correct_after"],
            diag["relabel_net_correctness_gain"],
        )

        y_current = y_current.copy()
        y_current[relabel_idx] = pred_class[relabel]

        seed_mask = seed_mask.copy()
        seed_mask[promote_idx] = True
        seed_mask[relabel_idx] = True
        held_out_mask = ~seed_mask

        delta_r = estimate_label_noise_delta(y_true_bin, y_current)
        result.delta_r.append(delta_r)
        result.seed_set_size_r.append(int(seed_mask.sum()))
        logger.info(
            "Refinement r=%d: seed_set=%d/%d promoted=%d relabeled=%d "
            "delta_vs_ground_truth(diagnostic)=%.4f",
            r, int(seed_mask.sum()), len(X), len(promote_idx), len(relabel_idx), delta_r,
        )

    result.y_refined = y_current
    result.seed_mask = seed_mask.copy()
    # Read-only diagnostics: purity of the curated seed set vs all rows. This is
    # the number that justifies training the federated classifier on the seed
    # set only (PRD 4.4: "ambiguous points are held out, not mislabeled").
    result.overall_purity = float((y_current == y_true_bin).mean()) if len(y_current) else float("nan")
    if seed_mask.any():
        result.seed_purity = float((y_current[seed_mask] == y_true_bin[seed_mask]).mean())
    logger.info(
        "Refinement done: seed_set=%d/%d seed_purity(diagnostic)=%.4f overall_purity=%.4f",
        int(seed_mask.sum()), len(X), result.seed_purity, result.overall_purity,
    )
    return result
