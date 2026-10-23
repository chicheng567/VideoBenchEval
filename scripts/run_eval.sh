#!/usr/bin/env bash
# Evaluate one checkpoint on Video-MME / MLVU / LongVideoBench, one process per GPU.
#
#   bash scripts/run_eval.sh                                   # defaults below
#   BUDGETS="4096 16384" GPUS=0,1 BENCHMARKS="videomme mlvu" bash scripts/run_eval.sh
#
# All budgets run in ONE pass: each video is decoded and encoded once, then planned /
# compressed / answered per budget. Results: results/<name>_b<B>/<benchmark>/.
# Re-running the same command resumes: answered (question, budget) pairs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-SamChen567/phase1_qbase_internvid}
RUN_NAME=${RUN_NAME:-$(basename "${MODEL}")}   # results/${RUN_NAME}_b<B>
GPUS=${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd,)}
BENCHMARKS=${BENCHMARKS:-"videomme mlvu longvideobench"}
DATA_ROOT=${DATA_ROOT:-/workspace/chicheng/datasets}
BUDGETS=${BUDGETS:-"1024 2048 4096 8192 16384"}   # LLM-side vision tokens per video (vbe/budget.py)
NUM_WORKERS=${NUM_WORKERS:-6}
TF_METHODS=${TF_METHODS:-none}   # e.g. "fastvid visionzip": training-free baselines (needs --protocol vl3, RUN_NAME with {method})
EXTRA_ARGS=${EXTRA_ARGS:-}   # e.g. "--protocol vl3": original VideoLLaMA3, resize-only budgets
VL3_BASE=${VL3_BASE:-/workspace/chicheng/OnlineLong/pretrained_models/videollama3_7b_local}
ORIG_PROJ=${ORIG_PROJ:-checkpoints/vl3_orig_projector.pt}
# orig: Phase-1 protocol (raw frames through the original VL3 projector);
# phase3: raw frames as Phase-3 training sends them (<= RAW_MAX_TOKENS rows, checkpoint projector)
RAW_MODE=${RAW_MODE:-orig}
RAW_MAX_TOKENS=${RAW_MAX_TOKENS:-256}
# model code; Phase-3 checkpoints need the tree they were trained with (hybrid_raw.py)
OL_SRC=${OL_SRC:-third_party/OnlineLong}
PY=${PY:-.venv/bin/python}
export PATH="$(dirname "${PY}"):${PATH}"   # ffprobe/ffmpeg from setup_env.sh
export PYTHONPATH="${OL_SRC}:.${PYTHONPATH:+:${PYTHONPATH}}"

# A hub id is downloaded once into checkpoints/<name>; a local directory is used as-is.
if [ ! -d "${MODEL}" ]; then
    LOCAL=checkpoints/$(basename "${MODEL}")
    "${PY}" -c "from huggingface_hub import snapshot_download as d; d('${MODEL}', local_dir='${LOCAL}')"
    MODEL=${LOCAL}
fi

# raw frames are projected with the original VideoLLaMA3 projector
if [ "${RAW_MODE}" = orig ]; then
    [ -f "${ORIG_PROJ}" ] || "${PY}" scripts/extract_orig_projector.py "${VL3_BASE}" "${ORIG_PROJ}"
fi

OUT=results/${RUN_NAME}_b{budget}
IFS=, read -ra GPU_LIST <<< "${GPUS}"
N=${#GPU_LIST[@]}
echo "[run_eval] model=${MODEL} gpus=${GPUS} benchmarks=${BENCHMARKS} budgets=${BUDGETS} raw_mode=${RAW_MODE} -> ${OUT}"

LOGS=results/${RUN_NAME//\{method\}/tf}_logs
mkdir -p "${LOGS}"
for BENCH in ${BENCHMARKS}; do
    pids=()
    for i in "${!GPU_LIST[@]}"; do
        CUDA_VISIBLE_DEVICES=${GPU_LIST[$i]} "${PY}" -m vbe.infer \
            --model_path "${MODEL}" --benchmark "${BENCH}" --data_root "${DATA_ROOT}" \
            --output "${OUT}/${BENCH}/shard${i}of${N}.jsonl" --shard_id "${i}" --num_shards "${N}" \
            --orig_projector "${ORIG_PROJ}" --raw_mode "${RAW_MODE}" --raw_max_tokens "${RAW_MAX_TOKENS}" \
            --budgets "${BUDGETS// /,}" --num_workers "${NUM_WORKERS}" --tf_method "${TF_METHODS// /,}" ${EXTRA_ARGS} \
            > "${LOGS}/${BENCH}_shard${i}of${N}.log" 2>&1 &
        pids+=($!)
    done
    fail=0
    for p in "${pids[@]}"; do wait "${p}" || fail=1; done
    if [ "${fail}" -ne 0 ]; then
        echo "[run_eval] ${BENCH}: a shard failed, see ${LOGS}/${BENCH}_*.log" >&2
        exit 1
    fi
    echo "[run_eval] ${BENCH} done"
done

for M in ${TF_METHODS}; do
    for B in ${BUDGETS}; do
        "${PY}" -m vbe.score "results/${RUN_NAME//\{method\}/${M}}_b${B}" "${DATA_ROOT}"
    done
done
