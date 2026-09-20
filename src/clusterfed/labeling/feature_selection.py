"""Stage 0 -- unsupervised feature selection (PRD 4.1; fixes R2.1).

v1 used Random Forest Gini importance, which consumes ground-truth labels and
made the paper's "no pre-labeled data" claim false. v2's default is Laplacian
score, a purely unsupervised ranking that keeps features preserving local
manifold structure -- no labels touch this stage. RF-Gini is kept as the
``rf_gini`` legacy option for the A1 ablation (it should land only marginally
above Laplacian, which is the honest answer to R2.1). ``variance`` and
``autoencoder`` are the two other unsupervised ablation arms PRD 4.1 lists.

All four methods target the same reduced dimensionality (~32 features) so v1
and v2 stay comparable.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import kneighbors_graph

logger = logging.getLogger("clusterfed.feature_selection")


# ---- Stage 0 default: Laplacian score (unsupervised) ---------------------------

def laplacian_score(X: np.ndarray, cfg, feature_names=None, *, seed: int = 42) -> pd.Series:
    """Rank features by Laplacian score (He, Cai & Niyogi 2005); lower = more relevant.

    Builds a k-NN heat-kernel affinity graph, then for each feature r scores
    how well it varies smoothly over that graph:

        score_r = (f_r~^T L f_r~) / (f_r~^T D f_r~)

    where f_r~ is f_r minus its degree-weighted mean, L = D - W the graph
    Laplacian. No labels are used anywhere in this computation.

    Returns a Series indexed by feature name, ascending (best/most-relevant first).
    """
    import scipy.sparse as sp

    X = np.asarray(X, dtype="float64")
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]

    subsample = cfg.get("laplacian_subsample", None)
    if subsample and n > int(subsample):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=int(subsample), replace=False)
        X = X[idx]
        n = X.shape[0]

    k = min(int(cfg.get("laplacian_knn", 5)), max(1, n - 1))
    dist_graph = kneighbors_graph(X, n_neighbors=k, mode="distance", include_self=False)

    sigma = cfg.get("laplacian_sigma", None)
    nz = dist_graph.data
    if sigma is None:
        sigma = float(np.median(nz)) if len(nz) else 1.0
        sigma = sigma if sigma > 1e-12 else 1.0
    else:
        sigma = float(sigma)

    W = dist_graph.copy()
    W.data = np.exp(-(dist_graph.data ** 2) / (2.0 * sigma ** 2))
    W = W.maximum(W.T)  # symmetrize the (possibly asymmetric) kNN graph

    deg = np.asarray(W.sum(axis=1)).ravel()
    deg_safe = np.where(deg <= 1e-12, 1e-12, deg)
    L = sp.diags(deg) - W

    weighted_mean = (X * deg[:, None]).sum(axis=0) / deg_safe.sum()
    F_tilde = X - weighted_mean

    LF = L @ F_tilde
    numerator = np.einsum("ij,ij->j", F_tilde, LF)
    denominator = np.einsum("ij,ij->j", F_tilde, deg[:, None] * F_tilde)
    denominator = np.where(denominator <= 1e-12, 1e-12, denominator)
    scores = numerator / denominator
    return pd.Series(scores, index=feature_names).sort_values(ascending=True)


# ---- v2.1 default: hybrid Laplacian + variance ensemble -------------------------

def hybrid_ranking(X: np.ndarray, cfg, feature_names=None, *, seed: int = 42) -> pd.Series:
    """Interleave Laplacian-score and variance rankings (both label-free).

    Motivation (EXPERIMENT_LOG Entry 7/10): Laplacian score prefers features
    that vary smoothly over the k-NN manifold and therefore *drops* spiky,
    near-categorical discriminators like Destination Port -- which cost
    FTP-Patator 0.29 F1. Variance ranking keeps exactly those spiky features
    but misses manifold structure. Alternating picks from the two rankings
    (deduplicated) covers both failure modes with zero label usage, keeping
    the R2.1 fix intact.

    Returns a Series whose *order* is the combined ranking (values are the
    combined rank positions, ascending = best).
    """
    lap = laplacian_score(X, cfg, feature_names, seed=seed)      # ascending = best first
    var = variance_ranking(X, feature_names)                      # descending = best first
    order: list[str] = []
    seen: set[str] = set()
    for lap_f, var_f in zip(lap.index, var.index):
        for f in (lap_f, var_f):
            if f not in seen:
                seen.add(f)
                order.append(f)
    # Append any stragglers (identical set, so this is defensive only).
    for f in list(lap.index) + list(var.index):
        if f not in seen:
            seen.add(f)
            order.append(f)
    return pd.Series(np.arange(len(order), dtype="float64"), index=order)


# ---- Ablation arm: variance thresholding ---------------------------------------

def variance_ranking(X: np.ndarray, feature_names=None) -> pd.Series:
    """Rank features by variance, descending (higher variance = kept first)."""
    X = np.asarray(X, dtype="float64")
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    var = X.var(axis=0)
    return pd.Series(var, index=feature_names).sort_values(ascending=False)


# ---- Ablation arm: autoencoder reconstruction-error importance -----------------

def autoencoder_importance(X: np.ndarray, feature_names=None, *, seed: int = 42,
                            epochs: int = 20, subsample: int | None = 50000) -> pd.Series:
    """Rank features by per-feature reconstruction error of a small unsupervised AE.

    A bottleneck autoencoder trained on all features reconstructs well-correlated
    (redundant) features cheaply; features the bottleneck reconstructs poorly
    carry information the others don't, so they rank as more important. No
    labels are used.
    """
    import torch
    import torch.nn as nn

    X = np.asarray(X, dtype="float64")
    n, d = X.shape
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(d)]
    if subsample and n > int(subsample):
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=int(subsample), replace=False)
        X = X[idx]

    torch.manual_seed(seed)
    hidden = max(4, min(32, d // 2))
    ae = nn.Sequential(
        nn.Linear(d, hidden), nn.Tanh(),
        nn.Linear(hidden, d),
    )
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    Xt = torch.tensor(X, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        recon = ae(Xt)
        loss = ((recon - Xt) ** 2).mean()
        loss.backward()
        opt.step()

    with torch.no_grad():
        recon = ae(Xt)
        per_feature_mse = ((recon - Xt) ** 2).mean(dim=0).numpy()
    return pd.Series(per_feature_mse, index=feature_names).sort_values(ascending=False)


# ---- Legacy arm: Random Forest Gini importance (v1; uses ground-truth labels) --

def rank_features_rf_gini(X, y, cfg, feature_names=None, *, seed: int = 42) -> pd.Series:
    """v1's RF/Gini importance. Kept only for the A1 ablation ("v1 legacy").

    Uses ground-truth labels ``y`` -- this is exactly the R2.1 problem the v2
    Stage 0 default (Laplacian score) fixes.
    """
    X = np.asarray(X, dtype="float64")
    y = np.asarray(y)
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(X.shape[1])]

    subsample = cfg.get("rf_subsample", None)
    if subsample and X.shape[0] > int(subsample):
        rng = np.random.default_rng(seed)
        idx = rng.choice(X.shape[0], size=int(subsample), replace=False)
        X, y = X[idx], y[idx]

    rf = RandomForestClassifier(
        n_estimators=int(cfg.get("rf_n_estimators", 200)),
        max_depth=cfg.get("rf_max_depth", None),
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X, y)
    importances = pd.Series(rf.feature_importances_, index=feature_names)
    return importances.sort_values(ascending=False)


def select_by_threshold(importances: pd.Series, T: float = 0.8, *, expected: int | None = None):
    """RF-legacy only: smallest prefix whose cumulative normalized importance reaches T."""
    ranked = importances.sort_values(ascending=False)
    normalized = ranked / ranked.sum()
    cumulative = normalized.cumsum()
    reached = np.searchsorted(cumulative.to_numpy(), T) + 1
    reached = int(min(reached, len(ranked)))
    selected = list(ranked.index[:reached])

    if expected is not None:
        logger.info(
            "Cumulative-importance T=%.2f selects %d features (expect ~%d)",
            T, len(selected), expected,
        )
        if abs(len(selected) - expected) > 4:
            logger.warning(
                "Selected count %d far from expected %d; keeping threshold result.",
                len(selected), expected,
            )
    return selected


def select_top_k(ranked: pd.Series, k: int) -> list[str]:
    """Laplacian/variance/autoencoder: take the top-k best-ranked (ascending-sorted for
    Laplacian since lower=better, descending-sorted for variance/autoencoder)."""
    return list(ranked.index[: min(k, len(ranked))])


# ---- Dispatcher ------------------------------------------------------------------

def select_features_stage0(X, y_bin, cfg, feature_names, *, seed: int = 42) -> dict:
    """Run the configured Stage 0 method. ``y_bin`` is used only by ``rf_gini``.

    Returns ``{"ranking": Series, "selected": [names], "selected_idx": [ints]}``.
    """
    method = cfg.get("method", "hybrid")
    expected = int(cfg.get("expected_selected", 32))

    if method == "laplacian":
        ranking = laplacian_score(X, cfg, feature_names, seed=seed)
        selected = select_top_k(ranking, expected)
    elif method == "hybrid":
        ranking = hybrid_ranking(X, cfg, feature_names, seed=seed)
        selected = select_top_k(ranking, expected)
    elif method == "variance":
        ranking = variance_ranking(X, feature_names)
        selected = select_top_k(ranking, expected)
    elif method == "autoencoder":
        ranking = autoencoder_importance(X, feature_names, seed=seed)
        selected = select_top_k(ranking, expected)
    elif method == "rf_gini":
        ranking = rank_features_rf_gini(X, y_bin, cfg, feature_names, seed=seed)
        selected = select_by_threshold(
            ranking, float(cfg.get("importance_threshold_T", 0.8)), expected=expected,
        )
    else:
        raise ValueError(f"Unknown feature_selection.method: {method}")

    selected_idx = [feature_names.index(s) for s in selected]
    logger.info("Stage 0 [%s] selected %d/%d features", method, len(selected), len(feature_names))
    return {"method": method, "ranking": ranking, "selected": selected, "selected_idx": selected_idx}


def reduction_summary(
    ranking: pd.Series,
    selected: list[str],
    *,
    method: str,
    total_features: int,
) -> dict:
    """Assemble a Stage-0 reduction summary dict, analogous to v1's."""
    k = len(selected)
    return {
        "method": method,
        "total_features": int(total_features),
        "selected_features": int(k),
        "reduction_pct": round(100.0 * (total_features - k) / total_features, 2),
        "top10_features": list(ranking.index[:10]),
        "selected": selected,
    }
