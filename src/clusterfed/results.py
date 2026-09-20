"""Persistent experiment-results store (Phase 0 of the PRD experiment program).

Every run -- main results, ablation arms, baselines, seeds -- lands in a
uniform on-disk layout so paper tables/figures are assembled from stored
artifacts, never from scrollback:

    outputs/results/<experiment>/<arm>/seed<seed>/
        metrics.json      headline metrics + full config snapshot + runtime
        table1.csv        per-node labeling quality (when produced)
        table2.csv        per-node federated detection (when produced)
        history_<tag>.csv per-round federated curves (round, train_acc, val_acc)
        refinement_trace.json  per-node delta_r / seed sizes / diagnostics

    outputs/results/index.csv   append-only master index; also the resume
                                ledger (a run present here is skipped on rerun)
"""
from __future__ import annotations

import csv
import json
import logging
import os
import time

logger = logging.getLogger("clusterfed.results")

INDEX_COLUMNS = ["experiment", "arm", "seed", "path", "labeling_raw_f1", "labeling_refined_f1",
                  "detection_f1", "detection_acc", "selected_round", "wall_clock_s", "finished_at"]


def run_dir(outputs_dir: str, experiment: str, arm: str, seed: int) -> str:
    return os.path.join(outputs_dir, "results", experiment, arm, f"seed{seed}")


def index_path(outputs_dir: str) -> str:
    return os.path.join(outputs_dir, "results", "index.csv")


def already_done(outputs_dir: str, experiment: str, arm: str, seed: int) -> bool:
    """Resume support: a (experiment, arm, seed) present in the index is complete."""
    path = index_path(outputs_dir)
    if not os.path.exists(path):
        return False
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            if (row["experiment"] == experiment and row["arm"] == arm
                    and str(row["seed"]) == str(seed)):
                return True
    return False


def save_run(outputs_dir: str, experiment: str, arm: str, seed: int, *,
             config_snapshot: dict, metrics: dict, tables: dict | None = None,
             histories: dict | None = None, refinement_trace: dict | None = None,
             wall_clock_s: float | None = None) -> str:
    """Persist one run. ``tables`` maps name -> DataFrame; ``histories`` maps
    tag -> History (per-round curves). Returns the run directory."""
    d = run_dir(outputs_dir, experiment, arm, seed)
    os.makedirs(d, exist_ok=True)

    payload = {
        "experiment": experiment,
        "arm": arm,
        "seed": int(seed),
        "metrics": metrics,
        "config": config_snapshot,
        "wall_clock_s": wall_clock_s,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(os.path.join(d, "metrics.json"), "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    for name, df in (tables or {}).items():
        df.to_csv(os.path.join(d, f"{name}.csv"), index=False)

    for tag, history in (histories or {}).items():
        with open(os.path.join(d, f"history_{tag}.csv"), "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["round", "train_acc", "val_acc"])
            for rnd, tr, va in zip(history.rounds, history.train_acc, history.val_acc):
                w.writerow([rnd, tr, va])
        zeta_rounds = getattr(history, "zeta_rounds", None)
        if zeta_rounds:
            with open(os.path.join(d, f"history_{tag}_zeta.csv"), "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["round", "zeta_sq"])
                for rnd, z in zip(zeta_rounds, history.zeta_sq):
                    w.writerow([rnd, z])

    if refinement_trace is not None:
        with open(os.path.join(d, "refinement_trace.json"), "w") as fh:
            json.dump(refinement_trace, fh, indent=2, default=str)

    _append_index(outputs_dir, experiment, arm, seed, d, metrics, wall_clock_s)
    logger.info("Saved run %s/%s/seed%s -> %s", experiment, arm, seed, d)
    return d


def _append_index(outputs_dir, experiment, arm, seed, d, metrics, wall_clock_s):
    path = index_path(outputs_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    exists = os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.writer(fh)
        if not exists:
            w.writerow(INDEX_COLUMNS)
        w.writerow([
            experiment, arm, seed, d,
            metrics.get("labeling_raw_avg_f1", ""),
            metrics.get("labeling_refined_avg_f1", ""),
            metrics.get("federated_detection_avg_f1", ""),
            metrics.get("federated_detection_avg_acc", ""),
            metrics.get("selected_round", ""),
            round(wall_clock_s, 1) if wall_clock_s else "",
            time.strftime("%Y-%m-%d %H:%M:%S"),
        ])


def load_index(outputs_dir: str):
    """Return the index as a list of dicts (empty list if no runs yet)."""
    path = index_path(outputs_dir)
    if not os.path.exists(path):
        return []
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def load_metrics(outputs_dir: str, experiment: str, arm: str, seed: int) -> dict:
    d = run_dir(outputs_dir, experiment, arm, seed)
    with open(os.path.join(d, "metrics.json")) as fh:
        return json.load(fh)
