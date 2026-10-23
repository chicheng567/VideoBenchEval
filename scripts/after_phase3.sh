#!/usr/bin/env bash
# Wait for Phase 3 to finish, upload the final weights to the Hub, then evaluate them with
# the Phase-3 raw-frame protocol (RAW_MODE=phase3). Progress: results/after_phase3.log
set -euo pipefail
cd "$(dirname "$0")/.."

CKPT=${CKPT:-/root/OnlineLong_ft/work_dirs/phase3_qbase_full_hybrid_lr1e-5}
REPO=${REPO:-SamChen567/phase3_qbase_full_hybrid_lr1e-5}
PY=${PY:-/miniconda/envs/ml/bin/python}
SRC=third_party/onlinelong_phase3   # frozen copy of the model code Phase 3 was trained with
log() { echo "[after_phase3 $(date -u '+%F %T') UTC] $*"; }

log "waiting for final weights in ${CKPT}"
until ls "${CKPT}"/*.safetensors >/dev/null 2>&1 && [ -f "${CKPT}/config.json" ]; do
    if ! pgrep -f 'qbase_llm_full_f[t].py' >/dev/null && ! pgrep -f 'chain_phase2_to_phase[3].sh' >/dev/null; then
        sleep 120   # a final save may still be landing
        ls "${CKPT}"/*.safetensors >/dev/null 2>&1 && break
        log "training is gone and left no final weights -- stopping"; exit 1
    fi
    sleep 60
done
until ! pgrep -f 'qbase_llm_full_f[t].py' >/dev/null; do sleep 30; done   # let the save finish
log "phase 3 finished"

log "uploading to https://huggingface.co/${REPO} (public)"
"${PY}" - "${CKPT}" "${REPO}" <<'PYEOF'
import sys
from huggingface_hub import HfApi
ckpt, repo = sys.argv[1:]
api = HfApi()
api.create_repo(repo, repo_type="model", private=False, exist_ok=True)
api.upload_folder(repo_id=repo, repo_type="model", folder_path=ckpt,
                  allow_patterns=["*.safetensors", "*.json", "*.jinja", "*.txt", "launcher.sh"],
                  ignore_patterns=["checkpoint-*/**", "runs/**"])
print("uploaded", len(api.list_repo_files(repo)), "files")
PYEOF
log "upload done"

mkdir -p "${SRC}"
cp -r /root/OnlineLong_ft/videollama3 /root/OnlineLong_ft/qwen2 "${SRC}/"   # videollama3 imports qwen2
log "model code frozen in ${SRC}"

log "waiting for free GPUs"
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 60; done

export DATA_ROOT=/root/benchmarks/eval_root VL3_BASE=/root/OnlineLong/pretrained_models/videollama3_7b_local \
       PY RAW_MODE=phase3 OL_SRC=${SRC} MODEL=${CKPT} RUN_NAME=phase3_qbase_full_hybrid_lr1e-5

log "smoke test: 2 videos of each benchmark, budget 4096, GPU 0"
for b in videomme mlvu longvideobench; do
    PYTHONPATH="${SRC}:." CUDA_VISIBLE_DEVICES=0 "${PY}" -m vbe.infer --model_path "${CKPT}" --benchmark "${b}" \
        --data_root "${DATA_ROOT}" --output "results/smoke_phase3/${b}.jsonl" --raw_mode phase3 \
        --budgets 4096 --limit 2 --num_workers 2
done
if grep -h '"error"' results/smoke_phase3/*.jsonl; then log "smoke test has errors -- stopping"; exit 1; fi
log "smoke test ok: $(cat results/smoke_phase3/*.jsonl | wc -l) answers"

log "full evaluation"
bash scripts/run_eval.sh
log "ALL_DONE"
