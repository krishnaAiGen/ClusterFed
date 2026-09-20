#!/usr/bin/env python3
"""Generic multi-seed experiment runner (Phase 0 of the PRD program).

One (experiment, arm) = one configuration, run across seeds, every artifact
persisted via clusterfed.results (metrics, Table I/II, per-round federated
curves, refinement traces). Resumable: (experiment, arm, seed) triples already
in outputs/results/index.csv are skipped.

Examples:
    # E1 main results, 5 seeds
    python experiments/run_experiment.py --experiment E1 --arm v2.1 --seeds 42 43 44 45 46

    # Supervised upper bound (U)
    python experiments/run_experiment.py --experiment E1 --arm upper_bound_supervised \\
        --seeds 42 43 44 45 46 --label-source true

    # v1-legacy lower bound (L)
    python experiments/run_experiment.py --experiment E1 --arm lower_bound_v1 \\
        --seeds 42 43 44 45 46 \\
        -o feature_selection.method=rf_gini -o clustering.n_clusters=2 \\
        -o clustering.clustering_space=pca2d -o refinement.iterations=0 \\
        -o federated.train_on=all -o federated.server_ema_decay=0 \\
        -o federated.model_selection=last -o federated.learning_rate=0.001 \\
        -o federated.fedprox_mu=0

    # Ablation arm, labeling only (fast; skips federated training)
    python experiments/run_experiment.py --experiment E3 --arm A1_variance \\
        --seeds 42 43 44 -o feature_selection.method=variance --skip-federated
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from clusterfed import pipeline, results
from clusterfed.config import Config, configure_logging, load_config, set_global_seed
from clusterfed.federated.train import run_table2
from clusterfed.labeling.evaluate import run_table1


def parse_override(kv: str):
    key, _, raw = kv.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got {kv!r}")
    for cast in (int, float):
        try:
            return key, cast(raw)
        except ValueError:
            pass
    if raw.lower() in ("true", "false"):
        return key, raw.lower() == "true"
    if raw.lower() in ("null", "none"):
        return key, None
    return key, raw


def apply_overrides(cfg: Config, overrides: list) -> Config:
    cfg = Config(copy.deepcopy(dict(cfg)))
    for key, value in overrides:
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            if part not in node or not isinstance(node[part], dict):
                node[part] = {}
            node = node[part]
        node[parts[-1]] = value
    return cfg


def run_one(cfg: Config, *, seed: int, label_source: str, skip_federated: bool):
    """Run the pipeline for one seed. Returns (metrics, tables, histories, trace)."""
    set_global_seed(seed)
    cfg = Config(copy.deepcopy(dict(cfg)))
    cfg["seed"] = seed

    if not os.path.exists(pipeline.cache_path(cfg)):
        pipeline.preprocess_and_cache(cfg)
    bundle = pipeline.load_cache(cfg)
    sel = pipeline.select_features(bundle, cfg, seed=seed)
    nodes = pipeline.build_scaled_nodes(bundle, cfg, seed=seed)
    selected_idx = sel["selected_idx"]

    metrics: dict = {"feature_selection_method": sel["method"],
                     "selected_feature_count": len(sel["selected"])}
    tables: dict = {}
    histories: dict = {}
    trace = None

    refinement_results = pipeline.compute_v2_labels(nodes, selected_idx, cfg, seed=seed)
    table1 = run_table1(nodes, refinement_results, seed=seed)
    tables["table1"] = table1
    t1_avg = table1[table1["category"] == "Average"].iloc[0]
    metrics["labeling_raw_avg_f1"] = round(float(t1_avg["raw_F1"]), 4)
    metrics["labeling_refined_avg_f1"] = round(float(t1_avg["refined_F1"]), 4)
    trace = {
        nodes[nid].row_label: {
            "delta_r": refinement_results[nid].delta_r,
            "seed_set_size_r": refinement_results[nid].seed_set_size_r,
            "seed_purity": refinement_results[nid].seed_purity,
            "overall_purity": refinement_results[nid].overall_purity,
            "uncertainty_diagnostics": refinement_results[nid].uncertainty_diagnostics,
        }
        for nid in sorted(nodes)
    }

    if not skip_federated:
        # ICC Tier B (E11): controlled pseudo-label noise on the training labels
        # only. Table I above is computed from the unperturbed result, so the
        # labeling report still describes the real pipeline.
        noise_rate = float(cfg.get_path("labeling.inject_noise_rate", 0.0) or 0.0)
        if noise_rate > 0:
            from clusterfed.labeling.noise import inject
            noise_mode = str(cfg.get_path("labeling.inject_noise_mode", "symmetric"))
            pseudo_labels, noise_diag = inject(nodes, refinement_results, noise_rate,
                                               seed=seed, mode=noise_mode)
            metrics.update({"inject_noise_rate": noise_diag["inject_noise_rate"],
                            "inject_noise_mode": noise_diag["inject_noise_mode"],
                            "rho_before": noise_diag["rho_before"],
                            "rho_after": noise_diag["rho_after"]})
        else:
            from clusterfed.labeling.noise import measure_rho
            pseudo_labels = {nid: r.y_refined for nid, r in refinement_results.items()}
            rhos, ws = [], []
            for nid in sorted(nodes):
                m = refinement_results[nid].seed_mask
                rhos.append(measure_rho(pseudo_labels[nid], nodes[nid].y_binary, m))
                ws.append(int(np.asarray(m, dtype=bool).sum()) if m is not None
                          else len(pseudo_labels[nid]))
            w = np.asarray(ws, float)
            metrics["rho_before"] = metrics["rho_after"] = round(
                float(np.dot(w / w.sum(), rhos)), 5) if w.sum() else float("nan")
        seed_masks = {nid: r.seed_mask for nid, r in refinement_results.items()}
        feature_idx = selected_idx if cfg.get_path("features.use_reduced_for_federated", True) else None

        if label_source == "true":
            from clusterfed.federated.train import _predict, run_federated
            from clusterfed.labeling.evaluate import score
            import pandas as pd

            history, model, splits = run_federated(
                nodes, feature_idx, "true", pseudo_labels, cfg, seed=seed, seed_masks=seed_masks)
            rows = []
            for node_id in sorted(splits):
                s = splits[node_id]
                m = score(s.y_test_true, _predict(model, s.X_test))
                rows.append({"node": node_id, "category": s.row_label,
                             "P": m["precision"], "R": m["recall"], "Acc": m["accuracy"], "F1": m["f1"]})
            table2 = pd.DataFrame(rows)
            avg = {c: table2[c].mean() for c in ("P", "R", "Acc", "F1")}
            avg.update({"node": "", "category": "Average"})
            table2 = pd.concat([table2, pd.DataFrame([avg])], ignore_index=True)
        else:
            table2, history, model = run_table2(
                nodes, feature_idx, pseudo_labels, cfg, seed=seed, seed_masks=seed_masks)

        tables["table2"] = table2
        histories[label_source] = history
        t2_avg = table2[table2["category"] == "Average"].iloc[0]
        metrics["federated_detection_avg_f1"] = round(float(t2_avg["F1"]), 4)
        metrics["federated_detection_avg_acc"] = round(float(t2_avg["Acc"]), 4)
        metrics["selected_round"] = history.selected_round
        # ICC: per-round traffic actually implied by this arm's knobs.
        up = getattr(history, "uplink_bytes_per_round", 0)
        down = getattr(history, "downlink_bytes_per_round", 0)
        frac = getattr(history, "client_fraction", 1.0)
        metrics["uplink_bytes_per_round"] = int(up)
        metrics["downlink_bytes_per_round"] = int(down)
        metrics["client_fraction"] = float(frac)
        # Expected per-client cost of a round: a client only pays when sampled.
        metrics["kb_per_client_per_round"] = round(frac * (up + down) / 1024.0, 3)
        metrics["total_train_rows"] = int(getattr(history, "total_train_rows", 0))

    return metrics, tables, histories, trace


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--experiment", required=True, help="e.g. E1, E3, E2")
    ap.add_argument("--arm", required=True, help="configuration name, e.g. v2.1, A1_variance")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    ap.add_argument("-o", "--override", action="append", default=[],
                    help="dotted config override, e.g. clustering.n_clusters=2 (repeatable)")
    ap.add_argument("--label-source", choices=["pseudo", "true"], default="pseudo")
    ap.add_argument("--skip-federated", action="store_true",
                    help="labeling-only (fast) -- for ablation arms that don't touch federated training")
    ap.add_argument("--force", action="store_true", help="rerun even if present in the index")
    args = ap.parse_args()

    configure_logging()
    base_cfg = load_config(args.config)
    overrides = [parse_override(kv) for kv in args.override]
    cfg = apply_overrides(base_cfg, overrides)
    outputs_dir = cfg.paths["outputs_dir"]

    for seed in args.seeds:
        if not args.force and results.already_done(outputs_dir, args.experiment, args.arm, seed):
            print(f"[skip] {args.experiment}/{args.arm}/seed{seed} already in index")
            continue
        t0 = time.time()
        metrics, tables, histories, trace = run_one(
            cfg, seed=seed, label_source=args.label_source, skip_federated=args.skip_federated)
        results.save_run(
            outputs_dir, args.experiment, args.arm, seed,
            config_snapshot={"overrides": {k: v for k, v in overrides},
                             "label_source": args.label_source,
                             "skip_federated": args.skip_federated},
            metrics=metrics, tables=tables, histories=histories,
            refinement_trace=trace, wall_clock_s=time.time() - t0)
        print(f"[done] {args.experiment}/{args.arm}/seed{seed}: {metrics}")


if __name__ == "__main__":
    main()
