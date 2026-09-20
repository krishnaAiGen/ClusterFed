"""Clustering bootstrap with confidence scores (PRD 4.3, Stage 2).

Two bootstrap modes, selected by ``clustering.n_clusters``:

- ``2`` (v1-legacy binary): k-means k=2, oriented by the known-benign
  reference point g (cluster nearest g = benign). Kept for ablation A2/A6
  comparability.
- ``>2`` (v2.1 multi-cluster, default): over-cluster with k-means k=K, then
  label each cluster benign/malicious by a *benign-density likelihood-ratio
  rule* against the same small known-benign sample v1 already used for
  orientation: cluster c is benign iff the share of the known-benign sample
  landing in c exceeds c's share of total mass (P(c|benign) > P(c), i.e.
  P(benign|c) > P(benign)). Rationale (EXPERIMENT_LOG Entry 10): binary
  k-means splits by *mass*, not by class -- on nodes where attack traffic
  outweighs benign 25:1 (DDoS, Hulk, PortScan) the k=2 boundary bisects the
  attack mass and half the attacks land in the "benign" cluster (observed:
  DDoS precision 0.98 / recall 0.45). Attack traffic is also multimodal, so
  over-clustering + per-cluster orientation is strictly safer than
  under-clustering. No new labels: the same <=100-point benign anchor sample.

Every point also gets a confidence score from its margin between the nearest
benign and nearest malicious centroid (PRD 4.3; reduces to the two-centroid
formula when k=2), which Stage 3 (threshold.py) turns into per-class adaptive
selection thresholds.
"""
from __future__ import annotations

import logging
from typing import Callable

import numpy as np
from sklearn.cluster import KMeans

logger = logging.getLogger("clusterfed.clustering")


def kmeans_cluster(X: np.ndarray, n_clusters: int = 2, *, n_init: int = 10, seed: int = 42):
    """Fit k-means; return ``(labels, centroids)``."""
    X = np.asarray(X, dtype="float64")
    km = KMeans(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X)
    return labels, km.cluster_centers_


def known_benign_point(X: np.ndarray, y_true_name: np.ndarray, n: int, *, seed: int = 42,
                       benign_name: str = "BENIGN") -> np.ndarray:
    """Reference point g = mean of up to ``n`` known-benign rows.

    This is the single place real labels enter the clustering bootstrap
    (unchanged from v1; it only fixes which of the two clusters is "benign",
    it never touches feature selection or the refinement loop).
    """
    X = np.asarray(X, dtype="float64")
    benign_mask = np.asarray(y_true_name) == benign_name
    benign_X = X[benign_mask]
    if len(benign_X) == 0:
        logger.warning("No known-benign rows; using global mean as reference g.")
        return X.mean(axis=0)
    if len(benign_X) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(benign_X), size=n, replace=False)
        benign_X = benign_X[idx]
    return benign_X.mean(axis=0)


def orient_labels(labels: np.ndarray, centroids: np.ndarray, g: np.ndarray) -> np.ndarray:
    """Relabel so the cluster nearest g becomes 0 (benign), the rest 1 (malicious)."""
    distances = np.linalg.norm(centroids - g.reshape(1, -1), axis=1)
    benign_cluster = int(np.argmin(distances))
    return (np.asarray(labels) != benign_cluster).astype(np.int64)


def kmeans_label(
    X: np.ndarray,
    g: np.ndarray,
    cfg,
    *,
    seed: int = 42,
):
    """Cluster X (k=2) and return orientation-corrected pseudo-labels (0 benign, 1 attack).

    ``cfg`` is the ``clustering`` config section. ``g`` is the known-benign point.
    Returns ``(y_pseudo, labels, centroids)`` so callers can also derive confidence.
    """
    n_clusters = int(cfg.get("n_clusters", 2))
    n_init = int(cfg.get("n_init", 10))
    labels, centroids = kmeans_cluster(X, n_clusters=n_clusters, n_init=n_init, seed=seed)
    if n_clusters != 2:
        distances = np.linalg.norm(centroids - g.reshape(1, -1), axis=1)
        benign_cluster = int(np.argmin(distances))
        y_pseudo = (labels != benign_cluster).astype(np.int64)
    else:
        y_pseudo = orient_labels(labels, centroids, g)
    return y_pseudo, labels, centroids


def confidence_scores(X: np.ndarray, centroids: np.ndarray, y_pseudo: np.ndarray) -> np.ndarray:
    """Centroid-margin confidence (PRD 4.3):

        conf_i = |d(x_i, c_benign) - d(x_i, c_malicious)| / max(d(x_i, c_benign), d(x_i, c_malicious))

    Only meaningful for the k=2 benign/malicious bootstrap. Returns values in
    [0, 1]; points near the decision boundary get low confidence.
    """
    X = np.asarray(X, dtype="float64")
    if centroids.shape[0] != 2:
        raise ValueError(f"confidence_scores expects 2 centroids (benign, malicious); got {centroids.shape[0]}")
    # y_pseudo is 0=benign,1=malicious after orientation, so centroids must be
    # re-associated with that orientation rather than assumed to be [benign, malicious].
    d_to_c0 = np.linalg.norm(X - centroids[0], axis=1)
    d_to_c1 = np.linalg.norm(X - centroids[1], axis=1)
    # Determine which raw centroid index corresponds to benign (y_pseudo==0) by
    # looking at which one the benign-labeled points are, on average, closer to.
    if np.any(y_pseudo == 0):
        benign_closer_to_c0 = d_to_c0[y_pseudo == 0].mean() <= d_to_c1[y_pseudo == 0].mean()
    else:
        benign_closer_to_c0 = True
    d_benign, d_malicious = (d_to_c0, d_to_c1) if benign_closer_to_c0 else (d_to_c1, d_to_c0)
    denom = np.maximum(d_benign, d_malicious)
    denom = np.where(denom == 0, 1e-12, denom)
    conf = np.abs(d_benign - d_malicious) / denom
    return np.clip(conf, 0.0, 1.0)


def known_benign_sample(X: np.ndarray, y_true_name: np.ndarray, n: int, *, seed: int = 42,
                        benign_name: str = "BENIGN") -> np.ndarray | None:
    """Up to ``n`` known-benign rows (the same anchor sample whose mean is g).

    The multi-cluster orientation rule needs the sample points themselves, not
    just their mean. Same rows, same label budget as v1's reference point.
    """
    X = np.asarray(X, dtype="float64")
    benign_X = X[np.asarray(y_true_name) == benign_name]
    if len(benign_X) == 0:
        return None
    if len(benign_X) > n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(benign_X), size=n, replace=False)
        benign_X = benign_X[idx]
    return benign_X


def _squared_dists(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """(N, K) squared distances via the quadratic expansion -- avoids the
    (N, K, D) broadcast intermediate, which at 177k x 12 x 32 would be ~0.5GB."""
    X = np.asarray(X, dtype="float64")
    x2 = (X * X).sum(axis=1)[:, None]
    c2 = (centroids * centroids).sum(axis=1)[None, :]
    d2 = x2 + c2 - 2.0 * (X @ centroids.T)
    return np.maximum(d2, 0.0)


def orient_clusters_by_benign_density(labels: np.ndarray, centroids: np.ndarray,
                                       benign_sample: np.ndarray) -> np.ndarray:
    """Return the set of benign cluster ids via the likelihood-ratio rule.

    Cluster c is benign iff  hits_c / |sample|  >  n_c / N
    (the known-benign sample is over-represented in c relative to c's mass).
    Since both sides sum to 1 over clusters, at least one cluster always falls
    on each side except under exact equality; fallbacks below handle the
    degenerate cases.
    """
    n_clusters = centroids.shape[0]
    N = len(labels)
    d2 = _squared_dists(np.asarray(benign_sample, dtype="float64"), centroids)
    sample_assign = d2.argmin(axis=1)

    hits = np.bincount(sample_assign, minlength=n_clusters).astype("float64")
    mass = np.bincount(labels, minlength=n_clusters).astype("float64")
    benign_score = hits / max(hits.sum(), 1.0) - mass / max(N, 1)

    benign_ids = np.where(benign_score > 0)[0]
    if len(benign_ids) == 0:  # degenerate: sample spread exactly like mass
        benign_ids = np.array([int(benign_score.argmax())])
    if len(benign_ids) == n_clusters:  # degenerate: everything looks benign
        benign_ids = np.delete(benign_ids, int(benign_score.argmin()))
    return benign_ids


def multicluster_label(X: np.ndarray, y_true_name: np.ndarray, cfg, *, seed: int = 42):
    """v2.1 Stage-2 bootstrap: over-cluster + benign-density orientation.

    Returns ``(y_pseudo, confidence)`` where confidence is the margin between
    the nearest benign and nearest malicious centroid.
    """
    n_clusters = int(cfg.get("n_clusters", 12))
    n_init = int(cfg.get("n_init", 10))
    n_clusters = max(2, min(n_clusters, len(X) // 2 if len(X) >= 4 else 2))
    benign_n = int(cfg.get("known_benign_n", 100))

    labels, centroids = kmeans_cluster(X, n_clusters=n_clusters, n_init=n_init, seed=seed)
    sample = known_benign_sample(X, y_true_name, benign_n, seed=seed)
    if sample is None:
        logger.warning("No known-benign rows; falling back to global-mean orientation.")
        g = np.asarray(X, dtype="float64").mean(axis=0)
        y_pseudo = orient_labels(labels, centroids, g)
        conf = np.full(len(X), 0.5)
        return y_pseudo, conf

    benign_ids = orient_clusters_by_benign_density(labels, centroids, sample)
    benign_mask_c = np.zeros(centroids.shape[0], dtype=bool)
    benign_mask_c[benign_ids] = True
    y_pseudo = (~benign_mask_c[labels]).astype(np.int64)

    d2 = _squared_dists(np.asarray(X, dtype="float64"), centroids)
    d_benign = np.sqrt(d2[:, benign_mask_c].min(axis=1))
    d_attack = np.sqrt(d2[:, ~benign_mask_c].min(axis=1))
    denom = np.maximum(np.maximum(d_benign, d_attack), 1e-12)
    conf = np.clip(np.abs(d_benign - d_attack) / denom, 0.0, 1.0)

    logger.info(
        "Multicluster bootstrap: k=%d, benign clusters=%s (%d/%d rows benign-labeled)",
        centroids.shape[0], benign_ids.tolist(), int((y_pseudo == 0).sum()), len(X),
    )
    return y_pseudo, conf


def bootstrap_labels(X: np.ndarray, y_true_name: np.ndarray, cfg, *, seed: int = 42):
    """Stage-2 dispatcher: k=2 -> v1-legacy binary path, k>2 -> multi-cluster.

    Returns ``(y_pseudo, confidence)`` in both modes.
    """
    n_clusters = int(cfg.get("n_clusters", 12))
    if n_clusters <= 2:
        g = known_benign_point(X, y_true_name, int(cfg.get("known_benign_n", 100)), seed=seed)
        y_pseudo, _, centroids = kmeans_label(X, g, cfg, seed=seed)
        conf = confidence_scores(X, centroids, y_pseudo)
        return y_pseudo, conf
    return multicluster_label(X, y_true_name, cfg, seed=seed)


# Registry so alternative algorithms can be plugged in by name later (ablation A2).
CLUSTERERS: dict[str, Callable] = {"kmeans": kmeans_cluster}
