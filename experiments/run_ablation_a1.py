#!/usr/bin/env python3
"""Ablation A1 (PRD Section 6): feature-selection method, one factor at a time.

Compares none (all features) / rf_gini (v1 legacy, uses labels) / laplacian
(v2 default, unsupervised) / variance (unsupervised) on raw Stage-2 clustering
quality (refinement held at R=0 so the only thing that varies is the feature
subset feeding K-means) across all 10 nodes.

Directly answers the open question from EXPERIMENT_LOG.md Entry 7: is
Laplacian systematically worse than RF-Gini, and for which attack types
specifically (the FTP-Patator regression was found by eyeballing one node's
selected-feature list; this quantifies it properly across all nodes).

    python experiments/run_ablation_a1.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from clusterfed import pipeline
from clusterfed.config import init_run
from clusterfed.labeling import feature_selection as fs
from clusterfed.labeling.evaluate import score
from clusterfed.labeling.refinement import refine_node_labels

ARMS = ["none", "rf_gini", "laplacian", "variance"]


def _selected_idx_for_arm(bundle, cfg, arm, *, seed):
    if arm == "none":
        return None, {"method": "none", "selected_features": len(bundle["features"])}
    method_cfg = dict(cfg.feature_selection)
    method_cfg["method"] = arm
    df, feats = bundle["df"], bundle["features"]
    from clusterfed.data import preprocess as pp
    X = df[feats].to_numpy("float64")
    y_bin = pp.binary_labels(df[cfg.preprocess.get("label_column", "Label")].to_numpy())
    sel = fs.select_features_stage0(X, y_bin, method_cfg, feats, seed=seed)
    return sel["selected_idx"], {"method": arm, "selected_features": len(sel["selected"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = init_run(args.config, log=False)
    seed = int(cfg["seed"])
    out_tab = os.path.join(cfg.paths["outputs_dir"], "tables")
    os.makedirs(out_tab, exist_ok=True)

    bundle = pipeline.load_cache(cfg)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)

    # Raw Stage-2 clustering only (R=0): isolates the feature-selection effect
    # from Stage 4 refinement, which is a separate ablation arm (A3).
    raw_refinement_cfg = dict(cfg.refinement)
    raw_refinement_cfg["iterations"] = 0

    rows = []
    arm_meta = {}
    for arm in ARMS:
        selected_idx, meta = _selected_idx_for_arm(bundle, cfg, arm, seed=seed)
        arm_meta[arm] = meta
        print(f"Arm [{arm}]: {meta['selected_features']} features selected")

        for node_id in sorted(nodes):
            node = nodes[node_id]
            X = node.X if selected_idx is None else node.X[:, selected_idx]
            result = refine_node_labels(
                X, node.y_true_name,
                cfg.model, cfg.clustering, cfg.threshold, cfg.imbalance, raw_refinement_cfg,
                seed=seed,
            )
            m = score(node.y_binary, result.y_cluster_raw)
            rows.append({"arm": arm, "node": node_id, "category": node.row_label, "F1": m["f1"],
                         "precision": m["precision"], "recall": m["recall"], "accuracy": m["accuracy"]})

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="category", columns="arm", values="F1")[ARMS]
    pivot.loc["Average"] = pivot.mean()

    df.to_csv(os.path.join(out_tab, "ablation_a1_feature_selection_long.csv"), index=False)
    pivot.to_csv(os.path.join(out_tab, "ablation_a1_feature_selection.csv"))
    with open(os.path.join(out_tab, "ablation_a1_meta.json"), "w") as fh:
        json.dump(arm_meta, fh, indent=2)

    print("\n==================== Ablation A1: feature selection (raw Stage-2 F1) ====================")
    print(pivot.round(3).to_string())
    print("============================================================================================")
    print(f"\nWritten to {out_tab}")


if __name__ == "__main__":
    main()
