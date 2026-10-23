#!/usr/bin/env bash
# Layout ablations (vbe/ablate.py) of one checkpoint, one process per GPU per benchmark,
# then one scores.json per (variant, budget).
#
#   JOBS="rho32:4096:16384,sel_merge:1024" bash scripts/run_ablate.sh
#
# Re-running the same command resumes. Needs the baseline run (scripts/run_eval.sh) of
# the same checkpoint: identical (question, budget) inputs are copied from it.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/root/OnlineLong_ft/work_dirs/phase3_qbase_full_hybrid_lr1e-5}
BASELINE=${BASELINE:-results/phase3_qbase_full_hybrid_lr1e-5}
OUT_ROOT=${OUT_ROOT:-results/ablate}
JOBS=${JOBS:?JOBS=variant:budget[:budget...],...}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
BENCHMARKS=${BENCHMARKS:-"videomme mlvu longvideobench"}
DATA_ROOT=${DATA_ROOT:-/root/benchmarks/eval_root}
OL_SRC=${OL_SRC:-third_party/onlinelong_phase3}
NUM_WORKERS=${NUM_WORKERS:-6}
EXTRA=${EXTRA:-}   # e.g. "--limit 2 --no_reuse"
PY=${PY:-/miniconda/envs/ml/bin/python}
export PATH="$(dirname "${PY}"):${PATH}"
export PYTHONPATH="${OL_SRC}:.${PYTHONPATH:+:${PYTHONPATH}}"

IFS=, read -ra GPU_LIST <<< "${GPUS}"
N=${#GPU_LIST[@]}
LOGS=${OUT_ROOT}/logs
mkdir -p "${LOGS}"
echo "[run_ablate] model=${MODEL} jobs=${JOBS} gpus=${GPUS} benchmarks=${BENCHMARKS} -> ${OUT_ROOT}"
for BENCH in ${BENCHMARKS}; do
    pids=()
    for i in "${!GPU_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "${PY}" -m vbe.ablate \
            --model_path "${MODEL}" --benchmark "${BENCH}" --data_root "${DATA_ROOT}" \
            --out_root "${OUT_ROOT}" --baseline_run "${BASELINE}" --jobs "${JOBS}" \
            --shard_id "${i}" --num_shards "${N}" --num_workers "${NUM_WORKERS}" ${EXTRA} \
            > "${LOGS}/${BENCH}_shard${i}of${N}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "${p}" || fail=1; done
    if [ "${fail}" -ne 0 ]; then
        echo "[run_ablate] ${BENCH}: a shard failed, see ${LOGS}/${BENCH}_*.log" >&2
        exit 1
    fi
    echo "[run_ablate] ${BENCH} done"
done

for d in "${OUT_ROOT}"/*_b*/; do
    "${PY}" -m vbe.score "${d%/}" "${DATA_ROOT}" > /dev/null
done
echo "[run_ablate] scored $(ls -d "${OUT_ROOT}"/*_b*/ | wc -l) runs"
