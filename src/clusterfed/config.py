"""Configuration loading, validation, seeding, and assumption logging.

Same convention as v1: read ``config.yaml`` into a nested attribute-access dict,
seed numpy/sklearn/torch, and surface every ``# ASSUMPTION`` knob at runtime.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any

import numpy as np
import yaml

logger = logging.getLogger("clusterfed")

# Knobs the PRD leaves open. Logged at runtime so the user can see which values
# diverge from a stated paper/PRD value and tune them.
ASSUMPTIONS: dict[str, str] = {
    "partition.benign_per_node": 'benign volume is "chosen randomly" in the source paper',
    "partition.benign_ratio_cap": "cap on benign:attack ratio per node",
    "partition.max_attack_per_node": "optional cap on attack rows per node",
    "feature_selection.laplacian_knn": "k-NN graph neighbors for the Laplacian-score affinity matrix",
    "feature_selection.laplacian_sigma": "heat-kernel bandwidth (null = auto median-distance)",
    "feature_selection.laplacian_subsample": "rows used to build the affinity graph (speed)",
    "feature_selection.rf_subsample": "rows used to fit legacy RF selector (speed)",
    "clustering.known_benign_n": "# known-benign rows anchoring cluster orientation",
    "clustering.n_clusters": "bootstrap cluster count (12 = over-cluster + density orientation; 2 = v1 binary)",
    "refinement.relabel_cluster_conf_quantile": "model may override cluster only below this cluster-conf quantile",
    "threshold.min_quantile": "floor per-class keep-quantile in class-balanced thresholding",
    "threshold.max_quantile": "ceiling per-class keep-quantile in class-balanced thresholding",
    "imbalance.focal_gamma": "focal-loss focusing parameter",
    "imbalance.logit_adjustment_tau": "logit-adjustment temperature",
    "refinement.local_epochs_per_iteration": "epochs training the refinement MLP per self-training round",
    "refinement.dropout_p": "MC-Dropout rate for the refinement-phase MLP",
    "refinement.uncertainty_threshold_tau_u": "predictive logit-std cutoff (tau_u) for co-agreement promotion",
    "refinement.relabel_confidence_threshold": "model-confidence cutoff to relabel a cluster/model disagreement",
    "refinement.max_promote_fraction_per_iteration": "cap on per-iteration seed-set growth (engineering safeguard)",
    "federated.learning_rate": "paper silent; 3e-4 tames non-IID FedAvg oscillation (Entry 13)",
    "federated.model_selection": "best pooled-train-acc round (unsupervised) vs last round",
    "federated.local_epochs": "PRD silent",
    "federated.batch_size": "PRD silent",
    "federated.test_split": "per-node holdout fraction",
}


class Config(dict):
    """Dict with attribute access and dotted ``get_path`` lookups."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value

    def get_path(self, dotted: str, default: Any = None) -> Any:
        node: Any = self
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node


def load_config(path: str = "config.yaml") -> Config:
    """Read and validate ``config.yaml`` into a :class:`Config`."""
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {path} did not parse to a mapping.")
    cfg = Config(raw)
    _validate(cfg)
    return cfg


def _validate(cfg: Config) -> None:
    required_top = ["seed", "paths", "preprocess", "partition", "federated", "model",
                     "feature_selection", "clustering", "threshold", "imbalance", "refinement"]
    missing = [k for k in required_top if k not in cfg]
    if missing:
        raise ValueError(f"config.yaml missing required sections: {missing}")
    if cfg.partition["num_nodes"] != 10:
        logger.warning(
            "partition.num_nodes=%s (PRD keeps the v1 10-node partition)", cfg.partition["num_nodes"]
        )
    if cfg.preprocess["nan_strategy"] not in {"drop", "median_impute"}:
        raise ValueError("preprocess.nan_strategy must be 'drop' or 'median_impute'")
    valid_fs = {"hybrid", "laplacian", "rf_gini", "variance", "autoencoder"}
    if cfg.feature_selection["method"] not in valid_fs:
        raise ValueError(f"feature_selection.method must be one of {valid_fs}")
    valid_thresh = {"per_class_adaptive", "global"}
    if cfg.threshold["policy"] not in valid_thresh:
        raise ValueError(f"threshold.policy must be one of {valid_thresh}")
    valid_loss = {"focal", "bce", "logit_adjusted"}
    if cfg.imbalance["loss"] not in valid_loss:
        raise ValueError(f"imbalance.loss must be one of {valid_loss}")


def set_global_seed(seed: int) -> None:
    """Seed python, numpy, and (if available) torch for deterministic runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # torch optional at import time
        logger.debug("torch not seeded (import failed)")
    logger.info("Global seed set to %d", seed)


def log_assumptions(cfg: Config) -> None:
    """Print every PRD-silent knob and its current value."""
    logger.info("---- ASSUMPTIONS (PRD silent; tune to taste) ----")
    for dotted, reason in ASSUMPTIONS.items():
        logger.info("  %-46s = %-10s  # %s", dotted, cfg.get_path(dotted), reason)
    logger.info("--------------------------------------------------")


def configure_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def init_run(config_path: str = "config.yaml", *, log: bool = True) -> Config:
    """One-call setup: configure logging, load config, seed, log assumptions."""
    configure_logging()
    cfg = load_config(config_path)
    set_global_seed(int(cfg["seed"]))
    if log:
        log_assumptions(cfg)
    return cfg
