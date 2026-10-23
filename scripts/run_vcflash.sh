#!/usr/bin/env bash
# VideoChat-Flash (vbe/vcflash.py) at matched visual-token budgets: one process per GPU over
# all benchmarks, budget by budget (smallest first), then one scores.json per budget.
# Re-running resumes (also with a different GPU count).
#
#   bash scripts/run_vcflash.sh
#   BUDGETS=1024,8192 BENCHMARKS=mlvu GPUS=0,1 bash scripts/run_vcflash.sh
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-/root/models/VideoChat-Flash-Qwen2-7B_res448}
RUN=${RUN:-results/vcflash7b_res448}
BUDGETS=${BUDGETS:-1024,2048,4096,8192,16384}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
BENCHMARKS=${BENCHMARKS:-videomme,mlvu,longvideobench}
DATA_ROOT=${DATA_ROOT:-/root/benchmarks/eval_root}
NUM_WORKERS=${NUM_WORKERS:-8}
EXTRA=${EXTRA:-}
PY=${PY:-/root/envs/vcflash/bin/python}          # transformers 4.40.1, as the checkpoint needs
SCORE_PY=${SCORE_PY:-/miniconda/envs/ml/bin/python}
export PYTHONPATH=".${PYTHONPATH:+:${PYTHONPATH}}"

IFS=, read -ra GPU_LIST <<< "${GPUS}"
N=${#GPU_LIST[@]}
LOGS=${RUN}_logs
mkdir -p "${LOGS}"
echo "[run_vcflash] model=${MODEL} budgets=${BUDGETS} gpus=${GPUS} benchmarks=${BENCHMARKS} -> ${RUN}_b*"
pids=()
for i in "${!GPU_LIST[@]}"; do
    CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "${PY}" -m vbe.vcflash \
        --model_path "${MODEL}" --benchmarks "${BENCHMARKS}" --data_root "${DATA_ROOT}" --budgets "${BUDGETS}" \
        --output "${RUN}_b{budget}/{bench}/shard${i}of${N}.jsonl" \
        --shard_id "${i}" --num_shards "${N}" --num_workers "${NUM_WORKERS}" ${EXTRA} \
        > "${LOGS}/shard${i}of${N}.log" 2>&1 &
    pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "${p}" || fail=1; done
for B in ${BUDGETS//,/ }; do
    [ -d "${RUN}_b${B}" ] && "${SCORE_PY}" -m vbe.score "${RUN}_b${B}" "${DATA_ROOT}" > /dev/null
done
if [ "${fail}" -ne 0 ]; then
    echo "[run_vcflash] a shard failed, see ${LOGS}/*.log" >&2
    exit 1
fi
echo "[run_vcflash] scored"
