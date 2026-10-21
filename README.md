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

- **Input = Phase-1 training geometry.** 1 fps, at most 180 frames (longer videos are
  sampled uniformly to 180, with their real timestamps); per-video budget 53,820 merged
  tokens, ≤ 2048 per frame (from the checkpoint's `preprocessor_config.json`). The whole
  video is one compression part: adaptive segmentation inside the model, K = 64 queries
  per segment, `Time:{a}s-{b}s:` before each segment.
- **Prompt** (all three, no subtitles):
  `Select the best answer ... Respond with only the letter (A, B, C, D) of the correct option.\n{question}\n A. ...\nThe best answer is:`
- Greedy decoding, 16 new tokens; the first standalone option letter is the prediction.
  No letter, unreadable video or OOM count as wrong (`no_answer` / `errors` in scores.json).
- Reported numbers: Video-MME overall + short/medium/long; MLVU M-Avg (mean of the 7
  task accuracies) + overall; LongVideoBench val overall + by duration group.

Not covered: subtitle variants, Video-MME / MLVU test sets, LongVideoBench test (no labels).
