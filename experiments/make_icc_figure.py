"""ICC 2027 figure: pseudo-label curation as a communication-efficiency lever.

(a) validation trajectories with attainment targets
(b) uplink bytes-to-target vs target level
(c) rounds-to-0.85 across deployment regimes
All inputs are stored per-round curves; no new training.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
OUT = ROOT / "IEEE Paper"
KB = 289.1  # per client per round, round-trip (2x144.5 KB model), E7/cost.json
ROUNDS = 200

plt.rcParams.update({
    "font.size": 7.2, "axes.labelsize": 7.6, "axes.titlesize": 7.8,
    "legend.fontsize": 6.5, "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.4,
    "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.1, "figure.dpi": 400,
})

C_OURS, C_LEG, C_SUP, C_2018 = "#1b4965", "#c1121f", "#8d99ae", "#5fa8d3"


def curves(exp, arm, tag=None):
    out = []
    for d in sorted((RES / exp / arm).glob("seed*")):
        hs = sorted(d.glob(f"history_{tag}.csv" if tag else "history_*.csv"))
        if not hs:
            continue
        r, v = [], []
        with open(hs[0]) as f:
            for row in csv.DictReader(f):
                if row.get("val_acc"):
                    r.append(int(row["round"]))
                    v.append(float(row["val_acc"]))
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
    return mu


def sustained(r, v, target, window=10):
    for i in range(len(v) - window + 1):
        if v[i:i + window].mean() >= target and v[i:].min() >= target - 0.05:
            return r[i]
    return None


fig = plt.figure(figsize=(7.16, 2.25))
gs = fig.add_gridspec(1, 3, width_ratios=[1.15, 1.0, 1.25], wspace=0.33,
                      left=0.062, right=0.995, top=0.90, bottom=0.30)

# ---------------- (a) trajectories ----------------
ax = fig.add_subplot(gs[0])
band(ax, curves("E1", "final_r0_mu10_k16", "pseudo"), C_OURS, "ClusterFed (0 labels)")
band(ax, curves("E1", "upper_bound_supervised", "true"), C_SUP, "Supervised FedAvg", "--")
band(ax, curves("E1", "lower_bound_v1", "pseudo"), C_LEG, "Legacy pipeline", "-.")
for t, lab in ((0.85, r"$\epsilon=0.85$"), (0.80, r"$\epsilon=0.80$")):
    ax.axhline(t, color="k", lw=0.5, ls=":", alpha=0.65)
    ax.text(198, t + 0.006, lab, ha="right", va="bottom", fontsize=6.0, color="k", alpha=0.8)
ax.annotate("", xy=(31.8, 0.858), xytext=(31.8, 0.915),
            arrowprops=dict(arrowstyle="->", color=C_OURS, lw=0.8))
ax.text(38, 0.917, "32 rounds, 9.0 MB", fontsize=6.0, color=C_OURS, va="bottom")
ax.set_xlabel("Communication round $t$")
ax.set_ylabel("Global validation accuracy")
ax.set_xlim(0, 200); ax.set_ylim(0.52, 0.95)
ax.legend(loc="lower right", frameon=False, handlelength=1.6)
ax.set_title("(a) Attainment", loc="left", fontweight="bold")

# ---------------- (b) uplink to target ----------------
ax = fig.add_subplot(gs[1])
targets = np.arange(0.70, 0.875, 0.01)
for exp, arm, tag, c, lab, ls in [
    ("E1", "final_r0_mu10_k16", "pseudo", C_OURS, "ClusterFed 2017", "-"),
    ("E1_2018", "final", "pseudo", C_2018, "ClusterFed 2018", "-"),
]:
    cs = curves(exp, arm, tag)
    mb, tt = [], []
    for t in targets:
        got = [sustained(r, v, t) for r, v in cs]
        got = [g for g in got if g]
        if len(got) == len(cs):
            tt.append(t); mb.append(np.mean(got) * KB / 1024)
    ax.plot(tt, mb, "o-", color=c, label=lab, ms=2.6)
ax.axhline(ROUNDS * KB / 1024, color="k", lw=0.6, ls="--")
ax.text(0.871, ROUNDS * KB / 1024 * 1.10, "200-round budget, 56.5 MB",
        fontsize=5.9, va="bottom", ha="right")
ax.axvspan(0.742, 0.875, color=C_LEG, alpha=0.07, lw=0)
ax.text(0.869, 0.60, "unreachable by\nlegacy / supervised", fontsize=5.9,
        color=C_LEG, ha="right", va="bottom")
ax.set_yscale("log")
ax.set_xlabel(r"Target accuracy $\epsilon$")
ax.set_ylabel("Traffic per client to reach $\\epsilon$ (MB)")
ax.set_xlim(0.697, 0.873); ax.set_ylim(0.42, 420)
ax.legend(loc="upper left", frameon=False, handlelength=1.6,
          borderaxespad=0.15, labelspacing=0.25)
ax.set_title("(b) Cost of a target", loc="left", fontweight="bold")

# ---------------- (c) regimes ----------------
ax = fig.add_subplot(gs[2])
regimes = [
    ("E1", "final_r0_mu10_k16", "pseudo", "reference"),
    ("E5", "clients50", "pseudo", "$N{=}50$"),
    ("E9", "dirichlet_a0.1", "pseudo", r"Dir(.1)"),
    ("E8", "dp_sigma0.01", "pseudo", r"DP $\sigma$.01"),
    ("E8", "dp_sigma0.1", "pseudo", r"DP $\sigma$.1"),
    ("E8", "poison0.1_median", "pseudo", "poison+med"),
    ("E3", "A4_global_threshold", "pseudo", "no curation"),
    ("E3", "A7_no_ema", "pseudo", "no EMA"),
]
BUDGET = ROUNDS * KB / 1024
labs = []
for i, (exp, arm, tag, lab) in enumerate(regimes):
    cs = curves(exp, arm, tag)
    got = [sustained(r, v, 0.85) for r, v in cs]
    ok = [g for g in got if g]
    labs.append(lab)
    if ok:  # attained by at least one seed: bar = mean over attaining seeds
        val, partial = np.mean(ok) * KB / 1024, len(ok) < len(cs)
        ax.bar(i, val, color=C_OURS, alpha=0.38 if partial else 1.0,
               edgecolor=C_OURS, lw=0.7, width=0.68,
               hatch="\\\\\\" if partial else None)
    else:   # no seed ever reaches the target within the 200-round budget
        val = BUDGET
        ax.bar(i, val, color="none", edgecolor=C_LEG, hatch="////", lw=0.7, width=0.68)
    ax.text(i, val + 1.4, f"{len(ok)}/{len(cs)}", ha="center", va="bottom",
            fontsize=5.6, color="k" if ok else C_LEG)
ax.axhline(BUDGET, color="k", lw=0.6, ls="--")
ax.text(len(labs) - 0.45, BUDGET * 1.10, "never attained", fontsize=5.9,
        ha="right", color=C_LEG)
ax.set_xticks(np.arange(len(labs)))
ax.set_xticklabels(labs, rotation=40, ha="right")
ax.set_ylabel(r"Traffic to $\epsilon=0.85$ (MB)")
ax.set_ylim(0, 72); ax.set_xlim(-0.7, len(labs) - 0.3)
ax.set_title("(c) Cost across regimes", loc="left", fontweight="bold")

fig.savefig(OUT / "fig_icc_comm.pdf", bbox_inches="tight")
fig.savefig(OUT / "fig_icc_comm.png", bbox_inches="tight", dpi=400)
print("wrote", OUT / "fig_icc_comm.pdf")
