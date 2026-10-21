# VideoBenchEval

Multiple-choice evaluation of OnlineLong checkpoints (VideoLLaMA3-7B + qbase token
compressor) on **Video-MME**, **MLVU (dev)** and **LongVideoBench (val)**.

The model code comes from the training repo, pinned as a submodule at
`third_party/OnlineLong` (branch `ft`, commit `f76e08c` "qbase timestemps fix": the first
commit with per-segment time marks, which `phase1_qbase_internvid` was trained with).

## Setup

```bash
git submodule update --init
bash scripts/setup_env.sh          # .venv: python 3.12, torch 2.9.1+cu126, flash-attn 2.8.3 built from source
```

Datasets are expected under `/workspace/chicheng/datasets` (`DATA_ROOT` to change):

| benchmark | videos | annotation | questions |
|---|---|---|---|
| Video-MME | `videomme/data/*.mp4` | `videomme/videomme/test-*.parquet` | 2700 (900 videos) |
| MLVU dev | `mlvu/*.mp4` | `mlvu/mlvu/test-*.parquet` | 2174 (7 MCQ tasks) |
| LongVideoBench val | `longvideobench/videos/*.mp4` | `longvideobench/lvb_val.json` | 1337 |

## Run

```bash
bash scripts/run_eval.sh                                     # SamChen567/phase1_qbase_internvid, all GPUs, all 3
MODEL=path/or/hub_id GPUS=0,1 BENCHMARKS="mlvu" bash scripts/run_eval.sh
```

Outputs go to `results/<run>/<benchmark>/shard*.jsonl` (one line per question: response,
parsed letter, frame / vision-token counts) and `results/<run>/scores.json`. Re-running
the same command resumes; `python -m vbe.score results/<run>` re-scores.

## Protocol

Budgeted input (`vbe/budget.py`), B = 53,820 LLM-side vision tokens per video (`BUDGET`):

1. Frames: 1 fps, at most `B // 64 * 8` (= 6,720) frames; longer videos are sampled
   uniformly (fps < 1, must stay >= 0.2).
2. One resolution for all frames: `min(native, 2048, max(256, B // T))` tokens per frame.
   If `T * hw <= B`, every frame goes to the LLM uncompressed (`raw`).
3. Otherwise the Phase-1 adaptive segmenter cuts the video into `T // 4 + 1` groups of
   1-8 frames. Group variation = sum of consecutive-frame cosine distances of the mean
   encoder feature; groups are compressed (-> 64 qbase tokens) from the lowest variation
   up until the video fits (`hybrid`).
4. If compressing every adaptive group still does not fit, uniform 8-frame groups, all
   compressed (`forced8`).

Compressed groups use the checkpoint's projector; raw frames use the ORIGINAL
VideoLLaMA3 projector (`scripts/extract_orig_projector.py`; encoder and LLM are
bit-identical). LLM layout in time order: raw frame `Time 12.0s:<hw rows>,`, compressed
group `Time:{a}s-{b}s:<|compression_start|><64 rows><|compression_end|>`.
`tests/test_equivalence.py` checks the two extremes token-for-token: all-raw == plain
VideoLLaMA3 with the original projector, all-compressed == the Phase-1 training path.

Prompts and answer parsing: official VideoLLaMA3 evaluation
(`VideoLLaMA3/evaluation/benchmarks/*.py`), no subtitles. Greedy, 16 new tokens.
Unparseable / unreadable / OOM count as wrong (`no_answer`, `errors` in scores.json).
Reported: Video-MME overall + short/medium/long; MLVU M-Avg + per task; LongVideoBench
val overall + per duration group; every benchmark also per layout mode.

Not covered: subtitle variants, Video-MME / MLVU test sets, LongVideoBench test (no labels).
