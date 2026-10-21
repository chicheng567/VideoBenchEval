"""Multiple-choice inference for VideoLLaMA3 + qbase compressor checkpoints (OnlineLong `ft`).

Every video goes through the Phase-1 whole-video compression path exactly as in
training (videollama3/train/data/global_compressor.py): one compression part over all
of the video's vision tokens, adaptive segmentation inside the model, and per-segment
`Time:{a}s-{b}s:` marks built from each frame's integer second.

One process per GPU; videos are sharded round-robin by --shard_id/--num_shards.
Results are appended to a jsonl as they finish, and already-answered qids are skipped,
so a killed run resumes where it stopped.

    python -m vbe.infer --model_path checkpoints/phase1_qbase_internvid \
        --benchmark videomme --output results/x/videomme/shard0.jsonl
"""
import argparse
import json
import os
import time
from collections import OrderedDict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch.utils.data import DataLoader, Dataset

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.mm_utils import load_video
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.train.data.global_compressor import build_range_ts_info

from vbe.benchmarks import DEFAULT_ROOT, build_prompt, extract_choice, load_benchmark


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--benchmark", required=True, choices=["videomme", "mlvu", "longvideobench"])
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--output", required=True, help="jsonl, appended to / resumed from")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    # training geometry (README of the checkpoint): 1 fps, at most 180 frames;
    # longer videos are sampled uniformly down to max_frames.
    p.add_argument("--fps", type=float, default=1)
    p.add_argument("--max_frames", type=int, default=180)
    p.add_argument("--merge_size", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only the first N videos of this shard (debug)")
    return p.parse_args()


class VideoGroups(Dataset):
    """One item = one video, decoded once, with the processed inputs of all its questions."""

    def __init__(self, groups, processor, args):
        self.groups = groups
        self.processor = processor
        self.args = args
        self.image_token_id = processor.tokenizer.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        video, samples = self.groups[i]
        try:
            frames, timestamps = load_video(video, fps=self.args.fps, max_frames=self.args.max_frames)
            n = len(frames)
            frame_sec = [int(round(float(t))) for t in timestamps][:n]
            items = []
            for s in samples:
                content = {"type": "video", "num_frames": n, "timestamps": timestamps}
                messages = [{"role": "user", "content": [content, {"type": "text", "text": build_prompt(s)}]}]
                inputs = self.processor(images=[frames], text=messages, merge_size=self.args.merge_size,
                                        return_tensors="pt")
                n_vis = int((inputs["input_ids"] == self.image_token_id).sum())
                assert n_vis % n == 0, f"{n_vis} vision tokens not divisible by {n} frames"
                items.append(dict(
                    sample=s,
                    inputs=dict(inputs),
                    compression_parts=[[0, n_vis]],
                    compression_ts_info=build_range_ts_info(content, self.processor.tokenizer),
                    compression_frame_sec=[frame_sec],
                    n_frames=n,
                    n_vision_tokens=n_vis,
                ))
            return dict(video=video, items=items, error=None)
        except Exception as e:  # unreadable video: every question of it is recorded as unanswered
            return dict(video=video, items=[dict(sample=s) for s in samples], error=repr(e))


def main():
    args = parse_args()
    samples = load_benchmark(args.benchmark, args.data_root)

    done = set()
    if os.path.exists(args.output):
        with open(args.output) as f:
            done = {json.loads(l)["qid"] for l in f if l.strip()}
    groups = OrderedDict()
    for s in samples:
        groups.setdefault(s["video"], []).append(s)
    groups = sorted(groups.items())[args.shard_id :: args.num_shards]
    if args.limit:
        groups = groups[: args.limit]
    todo = [(v, [s for s in ss if s["qid"] not in done]) for v, ss in groups]
    todo = [(v, ss) for v, ss in todo if ss]
    n_q = sum(len(ss) for _, ss in todo)
    print(f"[{args.benchmark} shard {args.shard_id}/{args.num_shards}] {len(todo)} videos / {n_q} questions to go "
          f"({len(done)} already in {args.output})", flush=True)
    if not todo:
        return

    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2",
    ).cuda().eval()
    if getattr(model.get_model(), "token_compressor", None) is None:
        raise RuntimeError(f"{args.model_path} has no token_compressor")
    processor = Videollama3Processor.from_pretrained(args.model_path, trust_remote_code=False)

    loader = DataLoader(VideoGroups(todo, processor, args), batch_size=None, shuffle=False,
                        num_workers=args.num_workers, prefetch_factor=2 if args.num_workers else None)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fout = open(args.output, "a")
    t0, n_done = time.time(), 0
    for group in loader:
        for it in group["items"]:
            s = it["sample"]
            rec = dict(qid=s["qid"], video=os.path.basename(s["video"]), answer=s["answer"], meta=s["meta"])
            if group["error"] is not None:
                rec.update(pred="", response="", error=group["error"])
            else:
                inputs = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in it["inputs"].items()}
                inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)
                try:
                    with torch.inference_mode():
                        out = model.generate(
                            **inputs,
                            modals=["video"],
                            compression_parts=it["compression_parts"],
                            compression_ts_info=it["compression_ts_info"],
                            compression_frame_sec=it["compression_frame_sec"],
                            compression_is_image=[False],
                            max_new_tokens=args.max_new_tokens,
                            do_sample=False,
                        )
                    resp = processor.tokenizer.batch_decode(out, skip_special_tokens=True)[0].strip()
                    rec.update(pred=extract_choice(resp, len(s["options"])), response=resp)
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec.update(pred="", response="", error=repr(e))
                rec.update(n_frames=it["n_frames"], n_vision_tokens=it["n_vision_tokens"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
        if n_done and n_done % 50 < len(group["items"]):
            el = time.time() - t0
            print(f"  {n_done}/{n_q} questions, {el / n_done:.2f}s/q, eta {(n_q - n_done) * el / n_done / 60:.0f} min",
                  flush=True)
    fout.close()
    print(f"[{args.benchmark} shard {args.shard_id}] done: {n_done} questions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
