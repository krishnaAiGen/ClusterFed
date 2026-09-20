"""ICC figure 2: noise RATE vs noise STRUCTURE.

E11 injects symmetric (class-uniform) label noise into the curated seed set;
E15 injects class-conditional noise at the same rates. Plotting both against the
*measured* rho separates the two hypotheses:

  H1 (rate)      -- attainment degrades with rho, whatever the structure.
  H2 (structure) -- symmetric noise is benign at any rho; one-sided noise is not.

Reads only stored runs; no training.
"""
from __future__ import annotations

import csv
import json
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
C_SYM, C_ASYM, C_MIR, C_REF = "#1b4965", "#c1121f", "#e09f3e", "#8d99ae"


def sustained(r, v, target, window=10):
    for i in range(len(v) - window + 1):
        if st.mean(v[i:i + window]) >= target and min(v[i:]) >= target - 0.05:
            return r[i]
    return None


def arm(exp, name):
    """Aggregate one arm -> dict or None if absent."""
    d = RES / exp / name
    if not d.is_dir():
        return None
    rho, f1, plat, att = [], [], [], 0
    n = 0
    for sd in sorted(d.glob("seed*")):
        mp, hp = sd / "metrics.json", sd / "history_pseudo.csv"
        if not (mp.exists() and hp.exists()):
            continue
        m = json.load(open(mp))["metrics"]
        r, v = [], []
        for row in csv.DictReader(open(hp)):
            if row.get("val_acc"):
                r.append(int(row["round"])); v.append(float(row["val_acc"]))
        if not r:
            continue
        n += 1
        rho.append(float(m.get("rho_after", float("nan"))))
        f1.append(float(m.get("federated_detection_avg_f1", float("nan"))))
        plat.append(st.mean(v[-50:]))
        if sustained(r, v, 0.85) is not None:
            att += 1
    if not n:
        return None
    return {"rho": st.mean(rho), "f1": st.mean(f1), "f1_sd": st.pstdev(f1),
            "plateau": st.mean(plat), "plateau_sd": st.pstdev(plat),
            "att": att, "n": n}


def series(exp, names):
    out = [arm(exp, nm) for nm in names]
    return [a for a in out if a]


SYM = series("E11", [f"rho_{x}" for x in ("0.00", "0.05", "0.10", "0.15", "0.20", "0.30")])
ASYM = series("E15", [f"a2b_{x}" for x in ("0.05", "0.10", "0.20")])
MIR = series("E15", ["b2a_0.10"])

if not SYM:
    raise SystemExit("no E11 runs found")

SINGLE_COL = True
fig = plt.figure(figsize=(3.45, 2.45))
gs = fig.add_gridspec(1, 1, left=0.155, right=0.985, top=0.93, bottom=0.175)

clean = SYM[0]

# ------------------------------------------------ (a) detection F1 vs rho
ax = fig.add_subplot(gs[0])
ax.axhline(clean["f1"], color=C_REF, lw=0.7, ls=":", zorder=1)
ax.text(0.332, clean["f1"] + 0.012, "uncorrupted", fontsize=5.9,
        color=C_REF, ha="right", va="bottom")


def plot(ax, S, key, color, label, marker="o"):
    if not S:
        return
    x = [a["rho"] for a in S]
    y = [a[key] for a in S]
    e = [a[key + "_sd"] for a in S]
    ax.errorbar(x, y, yerr=e, color=color, marker=marker, ms=3.0, capsize=1.6,
                elinewidth=0.7, label=label)


plot(ax, SYM, "f1", C_SYM, "symmetric noise")
plot(ax, ASYM, "f1", C_ASYM, "attack$\\rightarrow$benign", "s")
plot(ax, MIR, "f1", C_MIR, "benign$\\rightarrow$attack", "^")
ax.set_xlabel(r"Measured seed-set mislabeling rate $\rho$")
ax.set_ylabel("Federated detection F1")
ax.set_xlim(-0.01, 0.335); ax.set_ylim(0.24, 0.99)
ax.legend(loc="lower left", frameon=False, handlelength=1.8, labelspacing=0.22,
          bbox_to_anchor=(0.0, -0.02))


fig.savefig(OUT / "fig_icc_rho.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_icc_rho.png", bbox_inches="tight", dpi=400)
print("wrote", OUT / "fig_icc_rho.pdf")
for nm, S in (("E11 symmetric", SYM), ("E15 attack2benign", ASYM), ("E15 benign2attack", MIR)):
    for a in S:
        print(f"  {nm:<20} rho={a['rho']:.3f} f1={a['f1']:.4f}+-{a['f1_sd']:.4f} "
              f"plateau={a['plateau']:.4f} attain={a['att']}/{a['n']}")
