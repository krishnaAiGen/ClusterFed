#!/usr/bin/env python3
"""E4: minority-class deep dive (PRD Section 8; fixes R3 / the imbalance story).

Per-attack-TYPE (not per-node) detection metrics of the final global model,
plus precision-recall curves for the four rare classes (Infiltration,
Heartbleed, XSS, SQL Injection). Each attack type is evaluated against the
benign test rows of its own node (the deployment-realistic negatives).

Outputs per seed under outputs/results/E4/<arm>/seed<N>/:
    per_type_metrics.csv       every attack type: n, P, R, F1 at the argmax decision
    pr_curve_<type>.csv        precision-recall curve points (rare classes)

    python experiments/run_e4_minority.py --seeds 42 43 44 45 46 [overrides...]
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd
import torch

from clusterfed import pipeline, results
from clusterfed.config import Config, configure_logging, load_config, set_global_seed
from clusterfed.federated.train import run_federated
from run_experiment import apply_overrides, parse_override

RARE_CLASSES = ["Infiltration", "Heartbleed", "Web Attack - XSS", "Web Attack - Sql Injection"]


@torch.no_grad()
def _attack_scores(model, X: np.ndarray) -> np.ndarray:
    """Sigmoid activation of the attack output unit = detection score."""
    model.eval()
    out = model(torch.tensor(np.asarray(X), dtype=torch.float32))
    return out[:, 1].cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--arm", default="final")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("-o", "--override", action="append", default=[])
    args = ap.parse_args()

    configure_logging()
    from sklearn.metrics import precision_recall_curve

    base_cfg = load_config(args.config)
    overrides = [parse_override(kv) for kv in args.override]
    cfg0 = apply_overrides(base_cfg, overrides)
    outputs_dir = cfg0.paths["outputs_dir"]

    for seed in args.seeds:
        if results.already_done(outputs_dir, "E4", args.arm, seed):
            print(f"[skip] E4/{args.arm}/seed{seed}")
            continue
        t0 = time.time()
        set_global_seed(seed)
        cfg = Config(dict(cfg0))
        cfg["seed"] = seed

        bundle = pipeline.load_cache(cfg)
        sel = pipeline.select_features(bundle, cfg, seed=seed)
        nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)
        refinement_results = pipeline.compute_v2_labels(nodes, sel["selected_idx"], cfg, seed=seed)
        pseudo_labels = {nid: r.y_refined for nid, r in refinement_results.items()}
        seed_masks = {nid: r.seed_mask for nid, r in refinement_results.items()}

        history, model, splits = run_federated(
            nodes, sel["selected_idx"], "pseudo", pseudo_labels, cfg, seed=seed, seed_masks=seed_masks)

        rows = []
        pr_curves = {}
        for node_id in sorted(splits):
            s = splits[node_id]
            scores = _attack_scores(model, s.X_test)
            pred = (scores >= 0.5).astype(int)
            names = np.asarray(s.y_test_name)
            benign_mask = names == "BENIGN"
            for attack in sorted(set(names) - {"BENIGN"}):
                a_mask = names == attack
                y_true = np.concatenate([np.ones(a_mask.sum()), np.zeros(benign_mask.sum())])
                y_pred = np.concatenate([pred[a_mask], pred[benign_mask]])
                y_score = np.concatenate([scores[a_mask], scores[benign_mask]])
                tp = int(((y_true == 1) & (y_pred == 1)).sum())
                fp = int(((y_true == 0) & (y_pred == 1)).sum())
                fn = int(((y_true == 1) & (y_pred == 0)).sum())
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * p * r / (p + r) if p + r else 0.0
                rows.append({"node": node_id, "attack_type": attack, "n_test": int(a_mask.sum()),
                             "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)})
                if attack in RARE_CLASSES and a_mask.sum() >= 2:
                    prec, rec, _ = precision_recall_curve(y_true, y_score)
                    pr_curves[attack] = pd.DataFrame({"precision": prec, "recall": rec})

        per_type = pd.DataFrame(rows).sort_values(["node", "attack_type"])
        d = results.save_run(
            outputs_dir, "E4", args.arm, seed,
            config_snapshot={"overrides": dict(overrides)},
            metrics={"minority_mean_f1": round(float(
                per_type[per_type["attack_type"].isin(RARE_CLASSES)]["f1"].mean()), 4),
                "selected_round": history.selected_round},
            tables={"per_type_metrics": per_type},
            histories={"pseudo": history},
            wall_clock_s=time.time() - t0)
        for attack, curve in pr_curves.items():
            safe = attack.replace(" ", "_").replace("-", "").lower()
            curve.to_csv(os.path.join(d, f"pr_curve_{safe}.csv"), index=False)
        print(f"[done] E4/{args.arm}/seed{seed}: minority rows =\n"
              f"{per_type[per_type['attack_type'].isin(RARE_CLASSES)].to_string(index=False)}")


if __name__ == "__main__":
    main()
