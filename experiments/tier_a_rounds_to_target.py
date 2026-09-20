"""Tier A: convert stored per-round federated curves into communication cost.

Reads every outputs/results/<exp>/<arm>/seed<N>/history_*.csv and computes
rounds-to-target and uplink-bytes-to-target. No new training.
"""
import csv
import json
import statistics as st
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "outputs" / "results"
KB_PER_CLIENT_ROUND = 289.1  # E7/cost.json
WINDOW = 10  # sustained-attainment window


def read_curve(p):
    rounds, val = [], []
    with open(p) as f:
        for row in csv.DictReader(f):
            if row.get("val_acc") in (None, ""):
                continue
            rounds.append(int(row["round"]))
            val.append(float(row["val_acc"]))
    return rounds, val


def first_hit(rounds, val, target):
    for r, v in zip(rounds, val):
        if v >= target:
            return r
    return None


def sustained(rounds, val, target, window=WINDOW):
    """First round r where the mean over [r, r+window) is >= target and the
    curve never drops below target-0.05 afterwards. Honest for oscillators."""
    n = len(val)
    for i in range(n - window + 1):
        w = val[i:i + window]
        if st.mean(w) >= target and min(val[i:]) >= target - 0.05:
            return rounds[i]
    return None


def main():
    rows = []
    for hist in sorted(RES.rglob("history_*.csv")):
        rel = hist.relative_to(RES)
        parts = rel.parts
        if len(parts) < 4:
            continue
        exp, arm, seed = parts[0], parts[1], parts[2]
        tag = hist.stem.replace("history_", "")
        rounds, val = read_curve(hist)
        if not rounds:
            continue
        rec = {
            "exp": exp, "arm": arm, "seed": seed, "tag": tag,
            "n_rounds": len(rounds),
            "final_val": val[-1],
            "max_val": max(val),
            "mean_last50": st.mean(val[-50:]) if len(val) >= 50 else st.mean(val),
            "std_last50": st.pstdev(val[-50:]) if len(val) >= 50 else st.pstdev(val),
        }
        for tgt in (0.70, 0.75, 0.80, 0.85, 0.86):
            k = f"{int(tgt*100)}"
            fh = first_hit(rounds, val, tgt)
            su = sustained(rounds, val, tgt)
            rec[f"hit{k}"] = fh
            rec[f"sus{k}"] = su
            rec[f"mb{k}"] = round(su * KB_PER_CLIENT_ROUND / 1024, 2) if su else None
        rows.append(rec)

    out = RES.parent / "tables" / "tierA_rounds_to_target.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # aggregate per (exp, arm, tag) across seeds
    agg = {}
    for r in rows:
        key = (r["exp"], r["arm"], r["tag"])
        agg.setdefault(key, []).append(r)

    summary = []
    for (exp, arm, tag), rs in sorted(agg.items()):
        e = {"exp": exp, "arm": arm, "tag": tag, "seeds": len(rs)}
        e["final_val_mean"] = round(st.mean([x["final_val"] for x in rs]), 4)
        e["mean_last50"] = round(st.mean([x["mean_last50"] for x in rs]), 4)
        e["std_last50"] = round(st.mean([x["std_last50"] for x in rs]), 4)
        for tgt in (0.70, 0.75, 0.80, 0.85, 0.86):
            k = f"{int(tgt*100)}"
            vals = [x[f"sus{k}"] for x in rs]
            ok = [v for v in vals if v is not None]
            e[f"sus{k}_n_attained"] = len(ok)
            e[f"sus{k}_mean"] = round(st.mean(ok), 1) if ok else None
            e[f"sus{k}_std"] = round(st.pstdev(ok), 1) if len(ok) > 1 else 0.0
            e[f"sus{k}_mb"] = round(st.mean(ok) * KB_PER_CLIENT_ROUND / 1024, 2) if ok else None
        summary.append(e)

    out2 = RES.parent / "tables" / "tierA_summary.csv"
    with open(out2, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)

    print(f"wrote {out}  ({len(rows)} curves)")
    print(f"wrote {out2} ({len(summary)} arms)\n")

    focus = [s for s in summary if (s["exp"], s["arm"]) in {
        ("E1", "final_r0_mu10_k16"), ("E1", "v2.1"), ("E1", "lower_bound_v1"),
        ("E1", "upper_bound_supervised"), ("E1_2018", "final"),
        ("E3", "A7_selection_last"), ("E3", "A7_no_ema"),
        ("E3", "A3_naive_selftraining"), ("E3", "A3_no_refinement_no_curation"),
        ("E3", "A10_train_on_all"),
    }]
    hdr = f"{'exp/arm':<42} {'tag':<12} {'n':>2} {'last50':>8} {'osc':>7} {'R@.80':>10} {'MB@.80':>8} {'R@.85':>10} {'MB@.85':>8}"
    print(hdr)
    print("-" * len(hdr))
    for s in sorted(focus, key=lambda x: (x["exp"], x["arm"])):
        r80 = f"{s['sus80_mean']}({s['sus80_n_attained']}/{s['seeds']})" if s["sus80_mean"] else f"--({s['sus80_n_attained']}/{s['seeds']})"
        r85 = f"{s['sus85_mean']}({s['sus85_n_attained']}/{s['seeds']})" if s["sus85_mean"] else f"--({s['sus85_n_attained']}/{s['seeds']})"
        print(f"{s['exp']+'/'+s['arm']:<42} {s['tag']:<12} {s['seeds']:>2} "
              f"{s['mean_last50']:>8.4f} {s['std_last50']:>7.4f} {r80:>10} "
              f"{str(s['sus80_mb']):>8} {r85:>10} {str(s['sus85_mb']):>8}")

    json.dump(summary, open(RES.parent / "tables" / "tierA_summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
