#!/usr/bin/env python3
"""Federated phase -> Table II: final global model, per category, trained on
v2's Stage 0-4 refined pseudo-labels. Also reports the fully-supervised upper
bound (the "U" bound from PRD Section 5) for comparison.

    python experiments/run_federated.py --config config.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clusterfed import pipeline
from clusterfed.config import init_run
from clusterfed.federated.train import run_federated, run_table2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = init_run(args.config)
    seed = int(cfg["seed"])
    out_tab = os.path.join(cfg.paths["outputs_dir"], "tables")
    os.makedirs(out_tab, exist_ok=True)

    bundle = pipeline.load_cache(cfg)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)

    feature_idx = None
    if cfg.get_path("features.use_reduced_for_federated", True):
        feature_idx = pipeline.select_features(bundle, cfg, seed=seed)["selected_idx"]

    # ---- Stages 2-4: refine per-node labels ----
    results = pipeline.compute_v2_labels(nodes, feature_idx, cfg, seed=seed)
    pseudo_labels = {nid: r.y_refined for nid, r in results.items()}
    seed_masks = {nid: r.seed_mask for nid, r in results.items()}

    # ---- Table II: final global model, trained on refined pseudo-labels ----
    table2, semi_hist, _ = run_table2(nodes, feature_idx, pseudo_labels, cfg, seed=seed,
                                      seed_masks=seed_masks)
    table2.to_csv(os.path.join(out_tab, "table2_federated.csv"), index=False)
    avg = table2[table2["category"] == "Average"].iloc[0]
    print(f"\nTable II average  P={avg['P']:.3f} R={avg['R']:.3f} "
          f"Acc={avg['Acc']:.3f} F1={avg['F1']:.3f}")

    # ---- Fully-supervised upper bound ("U" in PRD Section 5) ----
    full_hist, _, _ = run_federated(nodes, feature_idx, "true", pseudo_labels, cfg, seed=seed)
    print(f"Semi-supervised (v2) final val_acc={semi_hist.val_acc[-1]:.3f}  "
          f"fully-supervised upper bound final val_acc={full_hist.val_acc[-1]:.3f}")
    print(f"\nTable II written to {out_tab}")


if __name__ == "__main__":
    main()
