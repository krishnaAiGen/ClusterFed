"""Stage 3 -- class-balanced adaptive selection (PRD 4.4; the minority-class mechanism).

Following the CBAFed principle: instead of one global confidence threshold,
set per-class thresholds from the empirical pseudo-class distribution, lowering
the bar for classes the clustering bootstrap rarely assigns so minority-attack
samples are not starved out of the seed set. Points that pass their class's
threshold form the seed set; everything else is held out (ambiguous), never
mislabeled.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger("clusterfed.threshold")


def class_balanced_thresholds(y_pseudo: np.ndarray, confidence: np.ndarray, cfg) -> tuple[dict, dict]:
    """Per-class confidence threshold: rarer classes get a lower (more lenient) bar.

    ``keep_frac_c`` (the fraction of class-c points kept) interpolates between
    ``min_quantile`` (the most common class -- strict) and ``max_quantile`` (the
    rarest class -- lenient), scaled by how under-represented class c is
    relative to the largest pseudo-class.
    """
    y_pseudo = np.asarray(y_pseudo)
    confidence = np.asarray(confidence, dtype="float64")
    min_q = float(cfg.get("min_quantile", 0.2))
    max_q = float(cfg.get("max_quantile", 0.8))

    classes = np.unique(y_pseudo)
    counts = {int(c): int((y_pseudo == c).sum()) for c in classes}
    max_count = max(counts.values()) if counts else 0

    thresholds: dict[int, float] = {}
    keep_fracs: dict[int, float] = {}
    for c in classes:
        c = int(c)
        rarity = 1.0 - (counts[c] / max_count) if max_count > 0 else 0.0
        keep_frac = min_q + (max_q - min_q) * rarity
        keep_fracs[c] = keep_frac
        conf_c = confidence[y_pseudo == c]
        if len(conf_c) == 0:
            thresholds[c] = 1.0
            continue
        thresholds[c] = float(np.quantile(conf_c, 1.0 - keep_frac))
    return thresholds, keep_fracs


def select_seed_set(y_pseudo: np.ndarray, confidence: np.ndarray, cfg) -> tuple[np.ndarray, dict]:
    """Return ``(seed_mask, thresholds)``. ``cfg`` is the ``threshold`` config section.

    ``policy: global`` (ablation A4) uses one confidence quantile across all
    points; ``policy: per_class_adaptive`` (default) uses
    :func:`class_balanced_thresholds`.
    """
    y_pseudo = np.asarray(y_pseudo)
    confidence = np.asarray(confidence, dtype="float64")
    policy = cfg.get("policy", "per_class_adaptive")

    if policy == "global":
        q = float(cfg.get("global_quantile", 0.5))
        thresh = float(np.quantile(confidence, q)) if len(confidence) else 1.0
        seed_mask = confidence >= thresh
        logger.info("Threshold policy=global thresh=%.4f seed_set=%d/%d",
                     thresh, int(seed_mask.sum()), len(y_pseudo))
        return seed_mask, {"global": thresh}

    if policy != "per_class_adaptive":
        raise ValueError(f"Unknown threshold.policy: {policy}")

    thresholds, keep_fracs = class_balanced_thresholds(y_pseudo, confidence, cfg)
    seed_mask = np.zeros(len(y_pseudo), dtype=bool)
    for c, t in thresholds.items():
        seed_mask |= (y_pseudo == c) & (confidence >= t)
    logger.info(
        "Threshold policy=per_class_adaptive thresholds=%s keep_fracs=%s seed_set=%d/%d",
        {k: round(v, 4) for k, v in thresholds.items()},
        {k: round(v, 4) for k, v in keep_fracs.items()},
        int(seed_mask.sum()), len(y_pseudo),
    )
    return seed_mask, thresholds
