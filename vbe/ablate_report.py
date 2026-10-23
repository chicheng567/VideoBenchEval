"""Paired comparison of ablation runs (vbe/ablate.py) against the baseline.

    python -m vbe.ablate_report results/ablate results/phase3_qbase_full_hybrid_lr1e-5 [out.json]

For every (variant, budget, benchmark): accuracy, the difference to the baseline on the
same questions with a 95% interval from resampling videos (2,000 draws), the same on the
questions the variant actually changed (``reused: false``) and on the long-video subset
(Video-MME long, LongVideoBench >= 3 min, all of MLVU), plus MLVU M-Avg.
"""
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

LONG = {"videomme": lambda m: m["duration"] == "long",
        "longvideobench": lambda m: str(m["duration_group"]) in ("600", "3600"),
        "mlvu": lambda m: True}


def load(run_dir, bench):
    recs = {}
    for f in sorted(glob.glob(os.path.join(run_dir, bench, "*.jsonl"))):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[r["qid"]] = r
    return recs


def paired(var, base, qids, draws=2000, seed=0):
    """(variant acc, baseline acc, delta, lo, hi, n) on ``qids``; interval over videos."""
    if not qids:
        return None
    by_vid = defaultdict(lambda: [0.0, 0])
    a = b = 0
    for q in qids:
        ca, cb = var[q]["pred"] == var[q]["answer"], base[q]["pred"] == base[q]["answer"]
        a += ca; b += cb
        v = by_vid[base[q]["video"]]
        v[0] += ca - cb; v[1] += 1
    d = np.array([x[0] for x in by_vid.values()]); n = np.array([x[1] for x in by_vid.values()])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (draws, len(d)))
    boot = 100 * d[idx].sum(1) / n[idx].sum(1)
    k = len(qids)
    return dict(acc=100 * a / k, base=100 * b / k, delta=100 * (a - b) / k,
                lo=float(np.percentile(boot, 2.5)), hi=float(np.percentile(boot, 97.5)), n=k)


def m_avg(recs, qids):
    by = defaultdict(list)
    for q in qids:
        by[recs[q]["meta"]["task_type"]].append(recs[q]["pred"] == recs[q]["answer"])
    return 100 * float(np.mean([np.mean(v) for v in by.values()]))


def main():
    root, base_prefix = sys.argv[1], sys.argv[2]
    out = {}
    for run in sorted(glob.glob(os.path.join(root, "*_b*"))):
        m = re.match(r"(.+)_b(\d+)$", os.path.basename(run))
        if not m:
            continue
        variant, budget = m.group(1), int(m.group(2))
        for bench in ("videomme", "mlvu", "longvideobench"):
            var = load(run, bench)
            if not var:
                continue
            base = load(f"{base_prefix}_b{budget}", bench)
            qids = sorted(q for q in var if q in base)
            row = dict(n_total=len(base), complete=len(qids) == len(base),
                       all=paired(var, base, qids),
                       changed=paired(var, base, [q for q in qids if not var[q].get("reused")]),
                       long=paired(var, base, [q for q in qids if LONG[bench](base[q]["meta"])]))
            if bench == "mlvu" and row["complete"]:
                row["m_avg"], row["m_avg_base"] = m_avg(var, qids), m_avg(base, qids)
            modes = defaultdict(int)
            for q in qids:
                if not var[q].get("reused"):
                    modes[var[q].get("mode", "error")] += 1
            row["changed_modes"] = dict(modes)
            out.setdefault(variant, {}).setdefault(str(budget), {})[bench] = row

    def fmt(p):
        if p is None:
            return "—"
        star = "*" if p["lo"] > 0 or p["hi"] < 0 else " "
        return f"{p['acc']:5.1f} ({p['delta']:+5.1f}{star} n={p['n']})"

    for variant in sorted(out):
        for budget in sorted(out[variant], key=int):
            for bench, r in out[variant][budget].items():
                flag = "" if r["complete"] else "  [INCOMPLETE]"
                extra = f"  M-Avg {r['m_avg']:.1f} ({r['m_avg'] - r['m_avg_base']:+.1f})" if "m_avg" in r else ""
                print(f"{variant:11s} {budget:>5} {bench:15s} all {fmt(r['all'])}  changed {fmt(r['changed'])}  "
                      f"long {fmt(r['long'])}{extra}{flag}")
    if len(sys.argv) > 3:
        with open(sys.argv[3], "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
