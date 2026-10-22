#!/usr/bin/env bash
# All vision-token budgets in one scripts/run_eval.sh pass (each budget sets how many
# frames it uses: <= B // 64 * 8), then print a summary table.
#
#   bash scripts/run_budgets.sh                     # BUDGETS="1024 2048 4096 8192 16384"
#   BUDGETS="4096" MODEL=... bash scripts/run_budgets.sh
set -euo pipefail
cd "$(dirname "$0")/.."
BUDGETS=${BUDGETS:-"1024 2048 4096 8192 16384"}
MODEL=${MODEL:-SamChen567/phase1_qbase_internvid}
BUDGETS=${BUDGETS} MODEL=${MODEL} RUN_NAME=$(basename "${MODEL}") bash scripts/run_eval.sh
PY=${PY:-.venv/bin/python}
"${PY}" - "$(basename "${MODEL}")" ${BUDGETS} <<'PYEOF'
import json, os, sys
name, budgets = sys.argv[1], sys.argv[2:]
print(f"{'budget':>7} {'Video-MME':>9} {'short':>6} {'medium':>6} {'long':>6} {'MLVU M-Avg':>10} {'LVB':>6}")
for b in budgets:
    p = f"results/{name}_b{b}/scores.json"
    if not os.path.exists(p):
        continue
    s = json.load(open(p))
    vm, ml, lv = s.get("videomme", {}), s.get("mlvu", {}), s.get("longvideobench", {})
    d = vm.get("by_duration", {})
    g = lambda x, k: f"{x[k]:.2f}" if k in x else "-"
    print(f"{b:>7} {g(vm,'accuracy'):>9} {d.get('short',{}).get('acc','-'):>6} {d.get('medium',{}).get('acc','-'):>6} "
          f"{d.get('long',{}).get('acc','-'):>6} {g(ml,'m_avg'):>10} {g(lv,'accuracy'):>6}")
PYEOF
