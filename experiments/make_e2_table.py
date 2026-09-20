#!/usr/bin/env python3
"""Assemble the E2 comparison table (PRD Section 5) from stored runs:
mean ± std over seeds + Wilcoxon signed-rank vs ClusterFed over paired
(seed, category) per-class F1 values. Writes CSV + Markdown.

    python experiments/make_e2_table.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import pandas as pd

from clusterfed import stats

OUT = "./outputs"
SEEDS = [42, 43, 44, 45, 46]
OURS = ("E1", "final_r0_mu10_k16", "ClusterFed v2 (ours, zero labels)")
ROWS = [
    ("E2", "B1_zhao_dynamic_threshold", "B1 Zhao et al. 2022 (2k labeled @server)"),
    ("E2", "B3_cbafed_class_balanced", "B3 CBAFed-style (5% labeled/client)"),
    ("E2", "B4_fedmse_autoencoder", "B4 FedMSE-style AE (benign traffic)"),
    ("E2", "B5_fedups_uncertainty", "B5 FedUPS-style (2k labeled @server)"),
    ("E1", "lower_bound_v1", "ClusterFed v1 pipeline (L bound)"),
    ("E1", "upper_bound_supervised", "Fully supervised FedAvg (U bound)"),
]


def per_class_f1(exp, arm, seed):
    path = os.path.join(OUT, "results", exp, arm, f"seed{seed}", "table2.csv")
    df = pd.read_csv(path)
    df = df[df["category"] != "Average"]
    return df.set_index("category")["F1"]


def seed_avgs(exp, arm, col="F1"):
    vals = []
    for s in SEEDS:
        path = os.path.join(OUT, "results", exp, arm, f"seed{s}", "table2.csv")
        df = pd.read_csv(path)
        vals.append(float(df[df["category"] == "Average"][col].iloc[0]))
    return vals


ours_f1 = {s: per_class_f1(*OURS[:2], s) for s in SEEDS}
lines = []
records = []
for exp, arm, label in ROWS:
    f1s = seed_avgs(exp, arm, "F1")
    accs = seed_avgs(exp, arm, "Acc")
    a_pairs, b_pairs = [], []
    for s in SEEDS:
        base = per_class_f1(exp, arm, s)
        common = ours_f1[s].index.intersection(base.index)
        a_pairs.extend(ours_f1[s][common].tolist())
        b_pairs.extend(base[common].tolist())
    w = stats.wilcoxon_signed_rank(a_pairs, b_pairs)
    records.append({
        "method": label,
        "detection_F1": stats.format_mean_std(f1s),
        "accuracy": stats.format_mean_std(accs),
        "wilcoxon_p_vs_ours": f"{w['p_value']:.2e}",
        "n_pairs": w["n_pairs"],
        "ours_better": "yes" if np.mean(a_pairs) > np.mean(b_pairs) else "no",
    })

ours_f1s = seed_avgs(*OURS[:2], "F1")
ours_accs = seed_avgs(*OURS[:2], "Acc")
records.insert(0, {
    "method": OURS[2],
    "detection_F1": stats.format_mean_std(ours_f1s),
    "accuracy": stats.format_mean_std(ours_accs),
    "wilcoxon_p_vs_ours": "--", "n_pairs": "--", "ours_better": "--",
})

table = pd.DataFrame(records)
os.makedirs(os.path.join(OUT, "results", "E2"), exist_ok=True)
table.to_csv(os.path.join(OUT, "results", "E2", "comparison_table.csv"), index=False)
cols = list(table.columns)
md_lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
for _, row in table.iterrows():
    md_lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
md = "\n".join(md_lines)
with open(os.path.join(OUT, "results", "E2", "comparison_table.md"), "w") as fh:
    fh.write(md + "\n")
print(md)
