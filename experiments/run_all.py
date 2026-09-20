#!/usr/bin/env python3
"""End-to-end: preprocess -> Stage 0-4 labeling -> federated; writes
headline_metrics.json, and diffs against v1's actual replicated numbers
(../ClusterFed v1/outputs/tables/table1_labeling.csv) where available.

    python experiments/run_all.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clusterfed import pipeline
from clusterfed.config import init_run
from clusterfed.federated.train import run_table2
from clusterfed.labeling.evaluate import run_table1


def _v1_baseline(cfg) -> dict | None:
    """Read v1's already-computed Table I average, if the sibling project has one."""
    import pandas as pd

    path = os.path.join(os.path.dirname(cfg.paths["raw_dir"]), "outputs", "tables", "table1_labeling.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    avg = df[df["category"] == "Average"].iloc[0]
    return {"all_F1": float(avg["all_F1"]), "red_F1": float(avg["red_F1"])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = init_run(args.config)
    seed = int(cfg["seed"])
    out_tab = os.path.join(cfg.paths["outputs_dir"], "tables")
    os.makedirs(out_tab, exist_ok=True)

    # 1. Preprocess (build cache if missing).
    if not os.path.exists(pipeline.cache_path(cfg)):
        pipeline.preprocess_and_cache(cfg)
    bundle = pipeline.load_cache(cfg)
    sel = pipeline.select_features(bundle, cfg, seed=seed)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)
    selected_idx = sel["selected_idx"]

    # 2. Stages 2-4: refine per-node labels once, reused below.
    results = pipeline.compute_v2_labels(nodes, selected_idx, cfg, seed=seed)
    pseudo_labels = {nid: r.y_refined for nid, r in results.items()}
    seed_masks = {nid: r.seed_mask for nid, r in results.items()}

    # 3. Table I: labeling quality, raw cluster vs refined.
    table1 = run_table1(nodes, results, seed=seed)
    table1.to_csv(os.path.join(out_tab, "table1_labeling.csv"), index=False)
    t1_avg = table1[table1["category"] == "Average"].iloc[0]

    # 4. Table II: federated detection quality on refined pseudo-labels.
    feature_idx = selected_idx if cfg.get_path("features.use_reduced_for_federated", True) else None
    table2, semi_hist, model = run_table2(nodes, feature_idx, pseudo_labels, cfg, seed=seed,
                                          seed_masks=seed_masks)
    table2.to_csv(os.path.join(out_tab, "table2_federated.csv"), index=False)
    t2_avg = table2[table2["category"] == "Average"].iloc[0]

    headline = {
        "labeling_raw_avg_f1": round(float(t1_avg["raw_F1"]), 4),
        "labeling_refined_avg_f1": round(float(t1_avg["refined_F1"]), 4),
        "labeling_refinement_gain": round(float(t1_avg["refined_F1"] - t1_avg["raw_F1"]), 4),
        "federated_detection_avg_f1": round(float(t2_avg["F1"]), 4),
        "federated_detection_avg_acc": round(float(t2_avg["Acc"]), 4),
        "feature_selection_method": sel["method"],
        "selected_feature_count": len(sel["selected"]),
        "seed": seed,
    }

    v1 = _v1_baseline(cfg)
    if v1 is not None:
        headline["v1_labeling_reduced_avg_f1"] = v1["red_F1"]
        headline["v2_vs_v1_labeling_f1_gain"] = round(
            float(t1_avg["refined_F1"]) - v1["red_F1"], 4
        )

    with open(os.path.join(out_tab, "headline_metrics.json"), "w") as fh:
        json.dump(headline, fh, indent=2)

    print("\n==================== HEADLINE METRICS (v2 E1, CICIDS2017) ====================")
    print(json.dumps(headline, indent=2))
    print("================================================================================")


if __name__ == "__main__":
    main()
