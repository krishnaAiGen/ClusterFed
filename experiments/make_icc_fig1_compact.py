"""ICC v2 Figure 1, single column: attainment trajectories only.

The three-panel version duplicated Table I (cost-of-target curve) and the
deployment-regime prose, which a six-page limit cannot afford. The trajectory
panel is the one that cannot be written as a number: it shows the conventional
bootstrap failing to settle at all.
"""
from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
OUT = ROOT / "ICC v2"

plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 7.8,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "figure.dpi": 400,
})
C_OURS, C_LEG, C_SUP = "#1b4965", "#c1121f", "#8d99ae"


def curves(exp, arm, tag):
    out = []
    for d in sorted((RES / exp / arm).glob("seed*")):
        h = d / f"history_{tag}.csv"
        if not h.exists():
            cand = sorted(d.glob("history_*.csv"))
            if not cand:
                continue
            h = cand[0]
        r, v = [], []
        for row in csv.DictReader(open(h)):
            if row.get("val_acc"):
                r.append(int(row["round"])); v.append(float(row["val_acc"]))
        if r:
            out.append((np.array(r), np.array(v)))
    return out


def band(ax, cs, color, label, ls="-"):
    n = min(len(v) for _, v in cs)
    R = cs[0][0][:n]
    M = np.vstack([v[:n] for _, v in cs])
    mu, sd = M.mean(0), M.std(0)
    ax.plot(R, mu, color=color, label=label, ls=ls)
    ax.fill_between(R, mu - sd, mu + sd, color=color, alpha=0.16, lw=0)


fig, ax = plt.subplots(figsize=(3.45, 2.35))
fig.subplots_adjust(left=0.145, right=0.985, top=0.97, bottom=0.175)

band(ax, curves("E17", "clusterfed", "pseudo"), C_OURS, "Label-free (no attack labels)")
band(ax, curves("E17", "supervised", "true"), C_SUP, "Supervised", "--")
band(ax, curves("E17", "legacy", "pseudo"), C_LEG, "Conventional bootstrap", "-.")

for t, lab in ((0.85, r"$\epsilon=0.85$"), (0.80, r"$\epsilon=0.80$")):
    ax.axhline(t, color="k", lw=0.5, ls=":", alpha=0.65)
    ax.text(199, t - 0.006, lab, ha="right", va="top", fontsize=6.0, alpha=0.8)

ax.annotate("", xy=(27.8, 0.858), xytext=(27.8, 0.915),
            arrowprops=dict(arrowstyle="->", color=C_OURS, lw=0.8))
ax.text(33, 0.917, "28 rounds, 7.9 MB", fontsize=6.0, color=C_OURS, va="bottom")
ax.annotate("", xy=(10.3, 0.845), xytext=(10.3, 0.60),
            arrowprops=dict(arrowstyle="->", color=C_SUP, lw=0.8))
ax.text(13, 0.60, "10 rounds, 2.9 MB", fontsize=6.0, color=C_SUP, va="bottom")

ax.set_xlabel("Communication round $t$")
ax.set_ylabel("Global validation accuracy")
ax.set_xlim(0, 200); ax.set_ylim(0.55, 0.95)
ax.legend(loc="lower right", frameon=False, handlelength=1.9, labelspacing=0.25)

fig.savefig(OUT / "fig_icc_traj.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_icc_traj.png", bbox_inches="tight", dpi=400)
print("wrote", OUT / "fig_icc_traj.pdf")
