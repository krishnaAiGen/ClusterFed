"""Shared pipeline steps + caching, so the experiment scripts stay thin.

Provides: preprocess-and-cache, cache load, Stage-0 feature selection, scaled-
node construction, and Stage 2-4 label refinement (computed once per node and
shared by the labeling-quality report and the federated-training step, since
refinement trains its own local MLP and is not cheap to repeat).
"""
from __future__ import annotations

import logging
import os
import pickle

from .data import loader, partition, preprocess
from .labeling import feature_selection as fs
from .labeling.refinement import refine_node_labels

logger = logging.getLogger("clusterfed.pipeline")

_CACHE_NAME = "preprocessed.pkl"


def cache_path(cfg) -> str:
    return os.path.join(cfg.paths["cache_dir"], _CACHE_NAME)


def preprocess_and_cache(cfg, *, files=None) -> dict:
    """Load raw -> normalize -> clean -> MinMax scale -> cache. Returns the bundle."""
    if str(cfg.preprocess.get("dataset", "cicids2017")) == "cicids2018":
        from .data.loader2018 import load_raw_2018
        raw = load_raw_2018(cfg.paths["raw_dir"], cfg.preprocess, files=files)
    else:
        raw = loader.load_raw(cfg.paths["raw_dir"], files=files)
    clean = preprocess.clean(raw, cfg.preprocess)
    feats = loader.feature_columns(clean, cfg.preprocess.get("label_column", "Label"))
    X_scaled, scaler = preprocess.scale(clean[feats].to_numpy("float64"), cfg.preprocess)
    clean[feats] = X_scaled

    bundle = {"df": clean, "features": feats}
    os.makedirs(cfg.paths["cache_dir"], exist_ok=True)
    with open(cache_path(cfg), "wb") as fh:
        pickle.dump(bundle, fh)
    logger.info("Cached %d rows x %d features -> %s", len(clean), len(feats), cache_path(cfg))
    return bundle


def load_cache(cfg) -> dict:
    path = cache_path(cfg)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No preprocessing cache at {path}. Run experiments/run_preprocess.py first."
        )
    with open(path, "rb") as fh:
        return pickle.load(fh)


def select_features(bundle: dict, cfg, *, seed: int) -> dict:
    """Stage 0 (PRD 4.1): unsupervised-by-default feature selection. Returns dict of results."""
    df, feats = bundle["df"], bundle["features"]
    X = df[feats].to_numpy("float64")
    y_bin = preprocess.binary_labels(df[cfg.preprocess.get("label_column", "Label")].to_numpy())
    return fs.select_features_stage0(X, y_bin, cfg.feature_selection, feats, seed=seed)


def build_scaled_nodes(bundle: dict, cfg, *, seed: int):
    """Partition the cached (already-scaled) frame into the 10 nodes (unchanged from v1)."""
    return partition.build_nodes(
        bundle["df"], cfg.partition, seed=seed, feature_names=bundle["features"]
    )


def compute_v2_labels(nodes, selected_idx, cfg, *, seed: int) -> dict:
    """Run Stages 2-4 once per node. Returns ``{node_id: RefinementResult}``.

    ``selected_idx`` (Stage 0 output) selects the feature columns; clustering
    and refinement then operate directly in that reduced feature space (PRD
    4.2 -- no 2D PCA projection before clustering, unlike v1).
    """
    results = {}
    for node_id in sorted(nodes):
        node = nodes[node_id]
        X = node.X if selected_idx is None else node.X[:, selected_idx]
        results[node_id] = refine_node_labels(
            X, node.y_true_name,
            cfg.model, cfg.clustering, cfg.threshold, cfg.imbalance, cfg.refinement,
            seed=seed,
        )
    return results
