"""Labeling-quality report (Table I analog; PRD Section 8, E1).

For each node: the Stage-2 raw cluster bootstrap (the "L" lower bound -- what
v1 shipped) vs the fully refined v2 pipeline (Stages 2-4), scored against the
true binary labels. Each node corresponds to a single attack family (or small
group), so this table *is* the per-attack-class breakdown the PRD's minority-
class success criteria reference (Bot = node 1, DoS slowloris = node 6).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger("clusterfed.evaluate")


def score(y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary") -> dict:
    """Precision / recall / accuracy / F1 for binary benign(0) vs attack(1)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    kw = dict(average=average, zero_division=0)
    return {
        "precision": float(precision_score(y_true, y_pred, **kw)),
        "recall": float(recall_score(y_true, y_pred, **kw)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, **kw)),
    }


def run_table1(nodes: dict, results: dict, *, seed: int = 42) -> pd.DataFrame:
    """Build the labeling-quality table from precomputed :class:`RefinementResult`s.

    ``results`` is ``pipeline.compute_v2_labels``'s output, keyed by node id.
    """
    rows = []
    for node_id in sorted(nodes):
        node = nodes[node_id]
        result = results[node_id]
        raw = score(node.y_binary, result.y_cluster_raw)
        refined = score(node.y_binary, result.y_refined)
        rows.append({
            "node": node_id,
            "category": node.row_label,
            "raw_P": raw["precision"], "raw_R": raw["recall"],
            "raw_Acc": raw["accuracy"], "raw_F1": raw["f1"],
            "refined_P": refined["precision"], "refined_R": refined["recall"],
            "refined_Acc": refined["accuracy"], "refined_F1": refined["f1"],
            "refinement_iterations_run": len(result.delta_r),
            "final_seed_set_size": result.seed_set_size_r[-1] if result.seed_set_size_r else int((~np.isnan(result.y_refined)).sum()),
        })
        logger.info(
            "Table I  %-26s raw_F1=%.3f  refined_F1=%.3f  (delta=%+.3f)",
            node.row_label, raw["f1"], refined["f1"], refined["f1"] - raw["f1"],
        )
    df = pd.DataFrame(rows)
    metric_cols = [c for c in df.columns if c not in ("node", "category")]
    avg = {c: df[c].mean() for c in metric_cols}
    avg.update({"node": "", "category": "Average"})
    df = pd.concat([df, pd.DataFrame([avg])], ignore_index=True)
    return df
