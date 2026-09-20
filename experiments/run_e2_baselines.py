#!/usr/bin/env python3
"""E2: SOTA baseline comparison (PRD Section 5; fixes R1.2).

Runs the four reimplemented baselines on the identical partition/features/
test-split protocol and stores per-node tables per seed. The comparison
table + Wilcoxon significance against ClusterFed's stored E1 results is
assembled afterwards by experiments/make_e2_table.py.

    python experiments/run_e2_baselines.py --seeds 42 43 44 45 46
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pandas as pd

from clusterfed import baselines, pipeline, results
from clusterfed.config import Config, configure_logging, load_config, set_global_seed

BASELINE_RUNNERS = {
    "B1_zhao_dynamic_threshold": "classifier",
    "B3_cbafed_class_balanced": "classifier",
    "B4_fedmse_autoencoder": "predictions",
    "B5_fedups_uncertainty": "classifier",
}


class _AsHistory:
    """Adapt the baselines' plain-dict history to the attribute access that
    results.save_run expects, so baseline trajectories are stored in exactly the
    same per-round CSV format as ClusterFed's and can be priced identically."""

    def __init__(self, d):
        self.rounds = d["rounds"]
        self.train_acc = d["train_acc"]
        self.val_acc = d["val_acc"]
        self.label_source = "baseline"
        self.selected_round = -1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("--only", choices=sorted(BASELINE_RUNNERS), default=None)
    ap.add_argument("--experiment", default="E2", help="results namespace")
    ap.add_argument("--force", action="store_true", help="rerun even if present in the index")
    args = ap.parse_args()

    configure_logging()
    base_cfg = load_config(args.config)
    outputs_dir = base_cfg.paths["outputs_dir"]

    for seed in args.seeds:
        set_global_seed(seed)
        cfg = Config(dict(base_cfg))
        cfg["seed"] = seed
        bundle = pipeline.load_cache(cfg)
        sel = pipeline.select_features(bundle, cfg, seed=seed)
        nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)
        splits = baselines.make_baseline_splits(nodes, sel["selected_idx"], cfg.federated, seed=seed)

        for name in sorted(BASELINE_RUNNERS):
            if args.only and name != args.only:
                continue
            if not args.force and results.already_done(outputs_dir, args.experiment, name, seed):
                print(f"[skip] {args.experiment}/{name}/seed{seed}")
                continue
            t0 = time.time()
            hist = None
            if name == "B1_zhao_dynamic_threshold":
                model, best_round, hist = baselines.run_zhao_dynamic_threshold(
                    splits, cfg.model, cfg.federated, seed=seed)
                rows = baselines.evaluate_model_baseline(model, splits)
            elif name == "B3_cbafed_class_balanced":
                model, best_round, hist = baselines.run_cbafed_class_balanced(
                    splits, cfg.model, cfg.federated, seed=seed)
                rows = baselines.evaluate_model_baseline(model, splits)
            elif name == "B5_fedups_uncertainty":
                model, best_round, hist = baselines.run_fedups_uncertainty(
                    splits, cfg.model, cfg.federated, seed=seed)
                rows = baselines.evaluate_model_baseline(model, splits)
            else:  # B4
                preds = baselines.run_fedmse_autoencoder(splits, cfg.federated, seed=seed)
                rows = baselines.evaluate_predictions_baseline(preds, splits)
                best_round = -1

            table = pd.DataFrame(rows)
            avg = {c: table[c].mean() for c in ("P", "R", "Acc", "F1")}
            avg.update({"node": "", "category": "Average"})
            table = pd.concat([table, pd.DataFrame([avg])], ignore_index=True)
            results.save_run(
                outputs_dir, args.experiment, name, seed,
                config_snapshot={"baseline": name},
                metrics={"federated_detection_avg_f1": round(float(avg["F1"]), 4),
                          "federated_detection_avg_acc": round(float(avg["Acc"]), 4),
                          "selected_round": best_round},
                tables={"table2": table},
                histories=({"pseudo": _AsHistory(hist)} if hist and hist.get("val_acc") else None),
                wall_clock_s=time.time() - t0)
            print(f"[done] {args.experiment}/{name}/seed{seed}: F1={avg['F1']:.4f} acc={avg['Acc']:.4f} "
                  f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
