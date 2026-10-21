#!/usr/bin/env bash
# Evaluate one checkpoint on Video-MME / MLVU / LongVideoBench, one process per GPU.
#
#   bash scripts/run_eval.sh                                   # defaults below
#   MODEL=SamChen567/phase1_qbase_internvid GPUS=0,1 BENCHMARKS="videomme mlvu" bash scripts/run_eval.sh
#
# Re-running the same command resumes: answered questions are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-SamChen567/phase1_qbase_internvid}
RUN_NAME=${RUN_NAME:-$(basename "${MODEL}")_b${BUDGET:-53820}}
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
BENCHMARKS=${BENCHMARKS:-"videomme mlvu longvideobench"}
DATA_ROOT=${DATA_ROOT:-/workspace/chicheng/datasets}
BUDGET=${BUDGET:-53820}            # LLM-side vision tokens per video (vbe/budget.py)
NUM_WORKERS=${NUM_WORKERS:-6}
VL3_BASE=${VL3_BASE:-/workspace/chicheng/OnlineLong/pretrained_models/videollama3_7b_local}
ORIG_PROJ=${ORIG_PROJ:-checkpoints/vl3_orig_projector.pt}
PY=${PY:-.venv/bin/python}
export PATH="$(dirname "${PY}"):${PATH}"   # ffprobe/ffmpeg from setup_env.sh
export PYTHONPATH="third_party/OnlineLong:.${PYTHONPATH:+:${PYTHONPATH}}"

# A hub id is downloaded once into checkpoints/<name>; a local directory is used as-is.
if [ ! -d "${MODEL}" ]; then
    LOCAL=checkpoints/$(basename "${MODEL}")
    "${PY}" -c "from huggingface_hub import snapshot_download as d; d('${MODEL}', local_dir='${LOCAL}')"
    MODEL=${LOCAL}
fi

# raw frames are projected with the original VideoLLaMA3 projector
[ -f "${ORIG_PROJ}" ] || "${PY}" scripts/extract_orig_projector.py "${VL3_BASE}" "${ORIG_PROJ}"

OUT=results/${RUN_NAME}
IFS=, read -ra GPU_LIST <<< "${GPUS}"
N=${#GPU_LIST[@]}
echo "[run_eval] model=${MODEL} gpus=${GPUS} benchmarks=${BENCHMARKS} budget=${BUDGET} -> ${OUT}"

for BENCH in ${BENCHMARKS}; do
    mkdir -p "${OUT}/${BENCH}"
    pids=()
    for i in "${!GPU_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "${PY}" -m vbe.infer \
            --model_path "${MODEL}" --benchmark "${BENCH}" --data_root "${DATA_ROOT}" \
            --output "${OUT}/${BENCH}/shard${i}of${N}.jsonl" --shard_id "${i}" --num_shards "${N}" \
            --orig_projector "${ORIG_PROJ}" --budget "${BUDGET}" --num_workers "${NUM_WORKERS}" \
            > "${OUT}/${BENCH}/shard${i}of${N}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "${p}" || fail=1; done
    if [ "${fail}" -ne 0 ]; then
        echo "[run_eval] ${BENCH}: a shard failed, see ${OUT}/${BENCH}/*.log" >&2
        exit 1
    fi
    echo "[run_eval] ${BENCH} done"
done

"${PY}" -m vbe.score "${OUT}" "${DATA_ROOT}"
