"""ICC 2027 analysis: attainment cost with per-arm traffic accounting.

Extends the Tier A re-analysis so that each arm is costed with ITS OWN
per-round payload (compression and partial participation change it) rather
than the dense-baseline 289.1 KB. Emits:

  outputs/tables/icc_attainment.csv   -- every (exp, arm) with R(eps) + traffic
  outputs/tables/icc_rho.csv          -- E11 rho sweep vs plateau/attainment
  outputs/tables/icc_tierc.csv        -- E12/E13/E14 bytes-per-round axes
"""
from __future__ import annotations

import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
TAB = ROOT / "outputs" / "tables"
DENSE_KB = 289.078          # 2 * 37,002 params * 4 B, full participation
BUDGET_ROUNDS = 200
WINDOW = 10
TARGETS = (0.70, 0.75, 0.80, 0.85)


def read_curve(p: Path):
    r, tr, va = [], [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            if row.get("val_acc") and row.get("train_acc"):
                r.append(int(row["round"]))
                tr.append(float(row["train_acc"]))
                va.append(float(row["val_acc"]))
    return r, tr, va


def sustained(r, v, target, window=WINDOW):
    """First round whose forward mean clears the target and which never later
    falls more than 0.05 below it. None if never attained."""
    n = len(v)
    for i in range(n - window + 1):
        if st.mean(v[i:i + window]) >= target and min(v[i:]) >= target - 0.05:
            return r[i]
    return None


def early_stop(r, tr, patience):
    best, best_i, stale = -1.0, 0, 0
    for i in range(len(tr)):
        if tr[i] > best:
            best, best_i, stale = tr[i], i, 0
        else:
            stale += 1
            if stale >= patience:
                return r[i], best_i
    return r[-1], best_i


def load_runs():
    """Yield one record per stored seed-run that has a federated curve."""
    for mpath in sorted(RES.rglob("metrics.json")):
        d = mpath.parent
        hist = sorted(d.glob("history_*.csv"))
        hist = [h for h in hist if not h.name.endswith("_zeta.csv")]
        if not hist:
            continue
        try:
            meta = json.load(open(mpath))
        except json.JSONDecodeError:
            continue
        m = meta.get("metrics", {})
        r, tr, va = read_curve(hist[0])
        if not r:
            continue
        yield {
            "exp": meta.get("experiment", d.parts[-3]),
            "arm": meta.get("arm", d.parts[-2]),
            "seed": meta.get("seed", d.name),
            "kb": float(m.get("kb_per_client_per_round") or DENSE_KB),
            "rho": m.get("rho_after"),
            "det_f1": m.get("federated_detection_avg_f1"),
            "lab_f1": m.get("labeling_raw_avg_f1"),
            "rounds": r, "train": tr, "val": va,
        }


def summarize(runs):
    groups: dict = {}
    for rec in runs:
        groups.setdefault((rec["exp"], rec["arm"]), []).append(rec)

    out = []
    for (exp, arm), rs in sorted(groups.items()):
        kb = st.mean([r["kb"] for r in rs])
        row = {
            "exp": exp, "arm": arm, "seeds": len(rs),
            "kb_per_round": round(kb, 2),
            "budget_mb": round(BUDGET_ROUNDS * kb / 1024, 2),
            "det_f1": _ms([r["det_f1"] for r in rs]),
            "lab_f1": _ms([r["lab_f1"] for r in rs]),
            "rho": _ms([r["rho"] for r in rs], nd=4),
            "plateau": round(st.mean([st.mean(r["val"][-50:]) for r in rs]), 4),
            "osc": round(st.mean([st.pstdev(r["val"][-50:]) for r in rs]), 4),
        }
        for t in TARGETS:
            k = int(t * 100)
            got = [sustained(r["rounds"], r["val"], t) for r in rs]
            ok = [g for g in got if g is not None]
            row[f"R{k}"] = round(st.mean(ok), 1) if ok else None
            row[f"R{k}_sd"] = round(st.pstdev(ok), 1) if len(ok) > 1 else 0.0
            # 95% t-interval half-width over the attaining seeds. Reported
            # because a cost metric with an unstated spread is not a cost.
            if len(ok) > 1:
                tcrit = {2: 12.71, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571,
                         7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}.get(len(ok), 2.262)
                hw = tcrit * st.stdev(ok) / (len(ok) ** 0.5)
                row[f"R{k}_ci95"] = round(hw, 1)
                row[f"MB{k}_ci95"] = round(hw * kb / 1024, 2)
            else:
                row[f"R{k}_ci95"] = None
                row[f"MB{k}_ci95"] = None
            row[f"R{k}_n"] = len(ok)
            row[f"MB{k}"] = round(st.mean(ok) * kb / 1024, 2) if ok else None
        # label-free early stopping at patience 20
        halts, deploys, fulls = [], [], []
        for r in rs:
            h, bi = early_stop(r["rounds"], r["train"], 20)
            halts.append(h)
            deploys.append(r["val"][bi])
            bt = max(r["train"])
            fulls.append(max(r["val"][i] for i in range(len(r["train"])) if r["train"][i] == bt))
        row["es20_halt"] = round(st.mean(halts), 1)
        row["es20_mb"] = round(st.mean(halts) * kb / 1024, 2)
        row["es20_acc"] = round(st.mean(deploys), 4)
        row["es20_cost"] = round(st.mean(fulls) - st.mean(deploys), 4)
        out.append(row)
    return out


def _ms(vals, nd=4):
    v = [float(x) for x in vals if x is not None]
    return round(st.mean(v), nd) if v else None


def main():
    runs = list(load_runs())
    rows = summarize(runs)
    TAB.mkdir(parents=True, exist_ok=True)

    with open(TAB / "icc_attainment.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote icc_attainment.csv ({len(rows)} arms from {len(runs)} runs)\n")

    def show(title, pred, cols=("rho", "det_f1", "plateau", "R80", "R80_n", "R85", "R85_n", "MB85", "kb_per_round")):
        sel = [r for r in rows if pred(r)]
        if not sel:
            return
        print(f"=== {title} ===")
        hdr = f"{'arm':<26}{'n':>3}" + "".join(f"{c:>11}" for c in cols)
        print(hdr); print("-" * len(hdr))
        for r in sel:
            line = f"{r['exp']+'/'+r['arm']:<26}{r['seeds']:>3}"
            for c in cols:
                v = r.get(c)
                line += f"{'--' if v is None else v:>11}"
            print(line)
        print()

    show("E16 tuned supervised references", lambda r: r["exp"] == "E16")
    show("E17 headline arms at 10 seeds", lambda r: r["exp"] == "E17")
    show("E18 top-k with error feedback", lambda r: r["exp"] == "E18")
    show("E19 external baselines", lambda r: r["exp"] == "E19")
    show("E11 rho sweep (Tier B)", lambda r: r["exp"] == "E11")
    show("E12 partial participation", lambda r: r["exp"] == "E12")
    show("E13 local epochs", lambda r: r["exp"] == "E13")
    show("E14 compression", lambda r: r["exp"] == "E14")
    show("reference arms", lambda r: (r["exp"], r["arm"]) in {
        ("E1", "final_r0_mu10_k16"), ("E1_2018", "final"),
        ("E1", "lower_bound_v1"), ("E1", "upper_bound_supervised")})

    for name, pred in (("icc_rho.csv", lambda r: r["exp"] == "E11"),
                       ("icc_tierc.csv", lambda r: r["exp"] in ("E12", "E13", "E14"))):
        sel = [r for r in rows if pred(r)]
        if sel:
            with open(TAB / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(sel[0].keys()))
                w.writeheader(); w.writerows(sel)
            print(f"wrote {name} ({len(sel)} arms)")


if __name__ == "__main__":
    main()
