#!/usr/bin/env python3
"""Labeling phase -> Table I (Stages 0-4 vs the Stage-2-only "L" lower bound),
Stage-0 feature-reduction summary, and per-node refinement-iteration deltas
(diagnostic support for a future E6).

    python experiments/run_labeling.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clusterfed import pipeline
from clusterfed.config import init_run
from clusterfed.labeling.evaluate import run_table1
from clusterfed.labeling.feature_selection import reduction_summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = init_run(args.config)
    seed = int(cfg["seed"])
    out_tab = os.path.join(cfg.paths["outputs_dir"], "tables")
    os.makedirs(out_tab, exist_ok=True)

    bundle = pipeline.load_cache(cfg)
    sel = pipeline.select_features(bundle, cfg, seed=seed)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)
    selected_idx = sel["selected_idx"]

    # ---- Stage 0 feature-reduction summary (fixes R2.1) ----
    summary = reduction_summary(
        sel["ranking"], sel["selected"], method=sel["method"], total_features=len(bundle["features"]),
    )
    with open(os.path.join(out_tab, "feature_reduction_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Stage 0 [{summary['method']}]: {summary['total_features']} -> "
          f"{summary['selected_features']} features ({summary['reduction_pct']}%)")

    # ---- Stages 2-4: refine per-node labels once, reused by Table I and Table II ----
    results = pipeline.compute_v2_labels(nodes, selected_idx, cfg, seed=seed)

    # ---- Table I: raw cluster bootstrap vs fully refined, per node ----
    table1 = run_table1(nodes, results, seed=seed)
    table1.to_csv(os.path.join(out_tab, "table1_labeling.csv"), index=False)
    avg = table1[table1["category"] == "Average"].iloc[0]
    print(f"\nTable I average  raw_F1={avg['raw_F1']:.3f}  refined_F1={avg['refined_F1']:.3f}"
          f"  (delta={avg['refined_F1'] - avg['raw_F1']:+.3f})")

    # ---- Per-node refinement trace (delta_r, seed-set growth, uncertainty diagnostics) ----
    trace = {
        nodes[nid].row_label: {
            "delta_r": results[nid].delta_r,
            "seed_set_size_r": results[nid].seed_set_size_r,
            "uncertainty_diagnostics": results[nid].uncertainty_diagnostics,
        }
        for nid in sorted(nodes)
    }
    with open(os.path.join(out_tab, "refinement_trace.json"), "w") as fh:
        json.dump(trace, fh, indent=2)

    print(f"\nTables written to {out_tab}")


if __name__ == "__main__":
    main()
