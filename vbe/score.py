"""Accuracy for one results directory (results/<run>/<benchmark>/*.jsonl).

    python -m vbe.score results/phase1_qbase_internvid

Writes <run>/scores.json and prints a summary. Unanswered questions (unreadable video,
OOM, no letter in the response) count as wrong and are reported separately.
"""
import glob
import json
import os
import sys
from collections import defaultdict

from vbe.benchmarks import DEFAULT_ROOT, load_benchmark

# per-benchmark breakdown fields; the headline number is in the first entry
BREAKDOWN = {
    "videomme": ["duration", "task_type", "domain"],
    "mlvu": ["task_type"],
    "longvideobench": ["duration_group", "question_category", "level"],
}


def _acc(recs):
    return round(100.0 * sum(r["pred"] == r["answer"] for r in recs) / max(len(recs), 1), 2)


def score_benchmark(bench, files, data_root):
    recs = {}
    for f in files:
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                recs[r["qid"]] = r  # a resumed run may have re-written a qid; keep the last
    expected = {s["qid"] for s in load_benchmark(bench, data_root)}
    recs = [r for q, r in recs.items() if q in expected]
    res = dict(
        n=len(recs),
        expected=len(expected),
        complete=len(recs) == len(expected),
        accuracy=_acc(recs),
        no_answer=sum(r["pred"] == "" for r in recs),
        errors=sum("error" in r for r in recs),
    )
    for field in BREAKDOWN[bench]:
        by = defaultdict(list)
        for r in recs:
            by[str(r["meta"][field])].append(r)
        res[f"by_{field}"] = {k: dict(n=len(v), acc=_acc(v)) for k, v in sorted(by.items())}
    if bench == "mlvu":
        # MLVU's reported M-Avg: mean of per-task accuracies
        tasks = res["by_task_type"].values()
        res["m_avg"] = round(sum(t["acc"] for t in tasks) / max(len(tasks), 1), 2)
    return res


def main():
    run_dir = sys.argv[1]
    data_root = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_ROOT
    scores = {}
    for bench in BREAKDOWN:
        files = sorted(glob.glob(os.path.join(run_dir, bench, "*.jsonl")))
        if files:
            scores[bench] = score_benchmark(bench, files, data_root)
    with open(os.path.join(run_dir, "scores.json"), "w") as f:
        json.dump(scores, f, indent=2, ensure_ascii=False)

    for bench, r in scores.items():
        flag = "" if r["complete"] else f"  [INCOMPLETE {r['n']}/{r['expected']}]"
        head = f"M-Avg {r['m_avg']:.2f}  acc {r['accuracy']:.2f}" if bench == "mlvu" else f"acc {r['accuracy']:.2f}"
        print(f"== {bench}: {head}  (n={r['n']}, no answer {r['no_answer']}, errors {r['errors']}){flag}")
        field = BREAKDOWN[bench][0]
        print("   " + "  ".join(f"{k}: {v['acc']:.2f}" for k, v in r[f"by_{field}"].items()))
    print(f"-> {os.path.join(run_dir, 'scores.json')}")


if __name__ == "__main__":
    main()
