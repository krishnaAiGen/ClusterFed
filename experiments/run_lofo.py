#!/usr/bin/env python3
"""Leave-one-attack-family-out zero-day protocol (PRD 3.3).

For each attack family f: remove ALL of f's rows from every client's data
before labeling and federated training (so no client has ever seen f), then
measure the final global model's detection of f's held-out rows -- the
head-to-head-comparable zero-day number (cf. Fed-DTCN's withheld-Botnet
protocol on CSE-CIC-IDS2018).

Reports per family: detection rate (recall) on the withheld rows, plus the
benign FPR of the same model for context. Stored under E_LOFO/<family>/seed.

    python experiments/run_lofo.py --seeds 42 [--families Bot ...] [--config config.yaml]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import torch

from clusterfed import pipeline, results
from clusterfed.config import Config, configure_logging, load_config, set_global_seed
from clusterfed.federated.train import run_federated
from run_experiment import apply_overrides, parse_override


@torch.no_grad()
def _predict(model, X):
    model.eval()
    out = model(torch.tensor(np.asarray(X), dtype=torch.float32))
    return out.argmax(dim=1).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--experiment", default="E_LOFO",
                    help="results namespace, e.g. E_LOFO (2017) or E_LOFO_2018")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--families", nargs="+", default=None,
                    help="default: every attack family in the dataset")
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    configure_logging()
    base_cfg = apply_overrides(load_config(args.config),
                               [parse_override(kv) for kv in args.override])
    outputs_dir = base_cfg.paths["outputs_dir"]

    for seed in args.seeds:
        set_global_seed(seed)
        cfg = Config(dict(base_cfg))
        cfg["seed"] = seed
        if not os.path.exists(pipeline.cache_path(cfg)):
            pipeline.preprocess_and_cache(cfg)
        bundle = pipeline.load_cache(cfg)
        sel = pipeline.select_features(bundle, cfg, seed=seed)
        nodes_full = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)

        families = args.families
        if families is None:
            families = sorted({name for n in nodes_full.values()
                               for name in n.attack_names})

        for family in families:
            arm = family.replace(" ", "_").replace("-", "")
            if results.already_done(outputs_dir, args.experiment, arm, seed):
                print(f"[skip] {args.experiment}/{arm}/seed{seed}")
                continue
            t0 = time.time()

            # Withhold family f from every node; pool f's rows as the zero-day set.
            import copy
            nodes = {}
            zero_day_X = []
            for nid, node in nodes_full.items():
                mask = node.y_true_name != family
                if (~mask).any():
                    zero_day_X.append(node.X[~mask])
                n2 = copy.copy(node)
                n2.X = node.X[mask]
                n2.y_true_name = node.y_true_name[mask]
                n2.y_binary = node.y_binary[mask]
                nodes[nid] = n2
            if not zero_day_X:
                print(f"[warn] family {family!r} has no rows; skipping")
                continue
            zero_day_X = np.vstack(zero_day_X)

            refinement_results = pipeline.compute_v2_labels(nodes, sel["selected_idx"], cfg, seed=seed)
            pseudo_labels = {nid: r.y_refined for nid, r in refinement_results.items()}
            seed_masks = {nid: r.seed_mask for nid, r in refinement_results.items()}
            history, model, splits = run_federated(
                nodes, sel["selected_idx"], "pseudo", pseudo_labels, cfg,
                seed=seed, seed_masks=seed_masks)

            zd_pred = _predict(model, zero_day_X[:, sel["selected_idx"]])
            detection_rate = float(zd_pred.mean())
            benign_X = np.vstack([s.X_test[s.y_test_true == 0] for s in splits.values()])
            fpr = float(_predict(model, benign_X).mean())

            metrics = {
                "family": family,
                "n_zero_day_rows": int(len(zero_day_X)),
                "zero_day_detection_rate": round(detection_rate, 4),
                "benign_fpr": round(fpr, 4),
                "selected_round": history.selected_round,
            }
            # Proper F1 on the mixed zero-day + benign-test evaluation set:
            y_true = np.concatenate([np.ones(len(zd_pred)), np.zeros(len(benign_X))])
            y_pred = np.concatenate([zd_pred, _predict(model, benign_X)])
            tp = float(((y_true == 1) & (y_pred == 1)).sum())
            fp = float(((y_true == 0) & (y_pred == 1)).sum())
            fn = float(((y_true == 1) & (y_pred == 0)).sum())
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            metrics["zero_day_f1"] = round(2 * p * r / (p + r), 4) if p + r else 0.0
            metrics["zero_day_precision"] = round(p, 4)

            results.save_run(outputs_dir, args.experiment, arm, seed,
                             config_snapshot={"family": family},
                             metrics=metrics, histories={"pseudo": history},
                             wall_clock_s=time.time() - t0)
            print(f"[done] {args.experiment}/{arm}/seed{seed}: DR={detection_rate:.3f} "
                  f"F1={metrics['zero_day_f1']:.3f} FPR={fpr:.3f} n={len(zero_day_X)}")


if __name__ == "__main__":
    main()
