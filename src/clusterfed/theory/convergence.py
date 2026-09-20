"""Empirical pseudo-label noise estimate (PRD 9.14; optional, §IV-C support).

delta = fraction of pseudo-labels that disagree with the true labels. The paper's
§IV-C argues feature reduction lowers this noise floor; we estimate it empirically
to annotate Fig 7 / the headline metrics.
"""
from __future__ import annotations

import numpy as np


def estimate_label_noise_delta(y_true: np.ndarray, y_pseudo: np.ndarray) -> float:
    """Fraction of mismatched pseudo-labels (0..1)."""
    y_true = np.asarray(y_true)
    y_pseudo = np.asarray(y_pseudo)
    if len(y_true) == 0:
        return float("nan")
    return float(np.mean(y_true != y_pseudo))
