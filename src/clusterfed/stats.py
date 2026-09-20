"""Multi-seed aggregation + significance tests (R1.6).

PRD Section 7: every number reported as mean +/- std over 5 seeds; Wilcoxon
signed-rank between method pairs, alpha = 0.05.
"""
from __future__ import annotations

import numpy as np


def mean_std(values) -> tuple[float, float]:
    arr = np.asarray([v for v in values if v is not None and not np.isnan(float(v))], dtype="float64")
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(arr.mean()), float(arr.std(ddof=1)) if len(arr) > 1 else 0.0


def format_mean_std(values, decimals: int = 3) -> str:
    m, s = mean_std(values)
    return f"{m:.{decimals}f} ± {s:.{decimals}f}"


def wilcoxon_signed_rank(a, b) -> dict:
    """Paired Wilcoxon signed-rank test (a vs b). Pairs are typically per-seed
    or per-class metric values. Returns statistic, p-value, and n_pairs."""
    from scipy.stats import wilcoxon

    a = np.asarray(a, dtype="float64")
    b = np.asarray(b, dtype="float64")
    if len(a) != len(b):
        raise ValueError(f"paired test needs equal lengths, got {len(a)} vs {len(b)}")
    diffs = a - b
    if np.allclose(diffs, 0):
        return {"statistic": 0.0, "p_value": 1.0, "n_pairs": len(a), "note": "all pairs identical"}
    stat, p = wilcoxon(a, b)
    return {"statistic": float(stat), "p_value": float(p), "n_pairs": int(len(a))}


def aggregate_metric(index_rows, experiment: str, arm: str, metric: str) -> list:
    """Pull a metric across seeds for (experiment, arm) from results.load_index rows."""
    vals = []
    for row in index_rows:
        if row["experiment"] == experiment and row["arm"] == arm and row.get(metric, "") != "":
            vals.append(float(row[metric]))
    return vals
