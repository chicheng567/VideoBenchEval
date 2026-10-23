"""Markdown tables of the layout ablations (vbe/ablate.py), paired against a reference.

    python -m vbe.ablate_tables results/ablate results/phase3_qbase_full_hybrid_lr1e-5 > tables.md

Each cell: accuracy (%) and the difference to the reference run on the same questions;
* = the 95% interval from resampling videos (vbe.ablate_report.paired) excludes 0.
"""
import os
import sys

from vbe.ablate_report import LONG, load, m_avg, paired

BENCHES = [("videomme", "Video-MME"), ("mlvu", "MLVU"), ("longvideobench", "LongVideoBench")]
SUBSETS = {"all": lambda b, m: True, "long": lambda b, m: LONG[b](m)}


class Runs:
    def __init__(self, root, base_prefix):
        self.root, self.base_prefix, self.cache = root, base_prefix, {}

    def get(self, name, budget, bench):
        key = (name, budget, bench)
        if key not in self.cache:
            d = f"{self.base_prefix}_b{budget}" if name == "baseline" else os.path.join(self.root, f"{name}_b{budget}")
            self.cache[key] = load(d, bench)
        return self.cache[key]


def cell(runs, name, ref, budget, bench, subset):
    var, base = runs.get(name, budget, bench), runs.get(ref, budget, bench)
    n = len(runs.get("baseline", budget, bench))
    if len(var) < n or len(base) < n:   # a run still in progress
        return "—"
    qids = [q for q in base if SUBSETS[subset](bench, base[q]["meta"])]
    p = paired(var, base, qids)
    if name == ref:
        return f"{p['acc']:.1f}"
    star = "*" if p["lo"] > 0 or p["hi"] < 0 else ""
    return f"{p['acc']:.1f} ({p['delta']:+.1f}{star})"


def table(runs, title, names, ref, budgets, subset, labels):
    out = [f"### {title}", ""]
    head = "| | " + " | ".join(f"{lab} {b // 1024}k" for _, lab in BENCHES for b in budgets) + " |"
    out += [head, "|---" * (1 + len(BENCHES) * len(budgets)) + "|"]
    for name in names:
        cells = [cell(runs, name, ref, b, bench, subset) for bench, _ in BENCHES for b in budgets]
        out.append(f"| {labels.get(name, name)} | " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def mavg_table(runs, names, budgets, labels):
    out = ["### MLVU M-Avg", "", "| | " + " | ".join(f"{b // 1024}k" for b in budgets) + " |",
           "|---" * (1 + len(budgets)) + "|"]
    for name in names:
        row = []
        for b in budgets:
            recs, base = runs.get(name, b, "mlvu"), runs.get("baseline", b, "mlvu")
            row.append(f"{m_avg(recs, list(base)):.1f}" if recs and len(recs) >= len(base) else "—")
        out.append(f"| {labels.get(name, name)} | " + " | ".join(row) + " |")
    return "\n".join(out) + "\n"


LABELS = {
    "baseline": "baseline（B/8 幀，hybrid / forced8）",
    "rho16": "rho16：B/16 幀，全 qbase（每段 4 幀）",
    "rho32": "rho32：B/32 幀，hybrid",
    "rho32nf": "rho32 不做 fill",
    "rho64": "rho64：B/64 幀，hybrid",
    "rho128": "rho128：B/128 幀，hybrid",
    "rho256": "rho256：B/256 幀，全 raw（≤256 tok）",
    "qtf2": "qtf2：B/32 幀，全 qbase（每段 2 幀）",
    "qtf1": "qtf1：B/64 幀，全 qbase（每幀一段）",
    "low64": "low64：B/64 幀，全 raw（≤64 tok）",
    "sel_resize": "選中間幀 + resize（重新編碼）",
    "sel_merge": "選中間幀 + merge（area pool）",
    "mean_merge": "整段平均 + merge（area pool）",
    "uni8": "uniform 切段（baseline 覆蓋率）",
    "uni16": "uniform 切段（rho16 覆蓋率）",
    "hivar": "優先壓縮高變化段",
}


def main():
    runs = Runs(sys.argv[1], sys.argv[2])
    hi, full = [4096, 8192, 16384], [1024, 2048, 4096, 8192, 16384]
    cover = ["baseline", "rho16", "rho32", "rho64", "rho128", "rho256", "qtf2", "qtf1", "low64"]
    print(table(runs, "A. 長影片覆蓋率：長片子集（對 baseline）", cover, "baseline", hi, "long", LABELS))
    print(table(runs, "A. 長影片覆蓋率：全部題目（對 baseline）", cover, "baseline", hi, "all", LABELS))
    print(table(runs, "A'. 同幀數下 raw 低解析 vs qbase：low64 對 qtf1（皆 B/64 幀、每幀 ≤64 token），長片子集",
                ["qtf1", "low64"], "qtf1", hi, "long", LABELS))
    print(table(runs, "A'. 同幀數下混 raw vs 輕壓縮：rho32 對 qtf2（皆約 B/32 幀），長片子集",
                ["qtf2", "rho32"], "qtf2", hi, "long", LABELS))
    comp = ["baseline", "sel_resize", "sel_merge", "mean_merge"]
    print(table(runs, "B. qbase vs 選幀 resize / merge：全部題目（對 baseline）", comp, "baseline", full, "all", LABELS))
    print(table(runs, "B. qbase vs 選幀 resize / merge：長片子集（對 baseline）", comp, "baseline", full, "long", LABELS))
    print(table(runs, "C. adaptive vs uniform 切段：baseline 覆蓋率，全部題目", ["baseline", "uni8"], "baseline", full,
                "all", LABELS))
    print(table(runs, "C. adaptive vs uniform 切段：rho16 覆蓋率（全 qbase），長片子集（對 rho16）",
                ["rho16", "uni16"], "rho16", hi, "long", LABELS))
    if os.path.isdir(os.path.join(sys.argv[1], "hivar_b1024")):
        print(table(runs, "D. 壓縮順序：優先壓縮高變化段，全部題目（對 baseline）", ["baseline", "hivar"], "baseline",
                    full, "all", LABELS))
    if os.path.isdir(os.path.join(sys.argv[1], "rho32_b1024")):
        print(table(runs, "A''. rho32 與 uni16 全預算：全部題目（對 baseline）", ["baseline", "rho32", "uni16"],
                    "baseline", full, "all", LABELS))
    if os.path.isdir(os.path.join(sys.argv[1], "rho32nf_b1024")):
        print(table(runs, "E. fill 有沒有用：rho32 不做 fill，全部題目（對 rho32）", ["rho32", "rho32nf"], "rho32",
                    full, "all", LABELS))
    print(mavg_table(runs, cover, hi, LABELS))
    print(mavg_table(runs, comp + ["uni8"] + (["hivar"] if os.path.isdir(os.path.join(sys.argv[1], "hivar_b1024"))
                                              else []), full, LABELS))


if __name__ == "__main__":
    main()
