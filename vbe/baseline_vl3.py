"""Original VideoLLaMA3 baseline with the official evaluation protocol
(VideoLLaMA3/evaluation): load_video(fps=1, max_frames=180), the image processor's
total-token cap = --budget (official --max_visual_tokens), no compression.

    python -m vbe.baseline_vl3 --model_path <VideoLLaMA3-7B> --benchmark longvideobench \
        --budget 8192 [--subtitles] --output results/<run>/longvideobench/shard0of1.jsonl

--subtitles (LongVideoBench only) interleaves the subtitles between the frames exactly
as the official longvideobench.py does; without it the prompt is the no-subtitle one
used by vbe.infer. Same answer parsing (vbe.benchmarks.extract_choice) and output
format as vbe.infer, so vbe.score reads it unchanged.
"""
import argparse
import json
import os
import time
from collections import OrderedDict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from videollama3.mm_utils import load_video
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor

from vbe.benchmarks import DEFAULT_ROOT, build_prompt, extract_choice, load_benchmark

MAX_FRAMES = 180


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--benchmark", required=True, choices=["videomme", "mlvu", "longvideobench"])
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--output", required=True)
    p.add_argument("--budget", type=int, default=16384, help="official --max_visual_tokens")
    p.add_argument("--strict_budget", action="store_true",
                   help="keep the vision tokens <= budget: when budget // n_frames is below the processor's "
                        "per-frame floor (min_tokens), drop the floor so the official resize alone fits")
    p.add_argument("--subtitles", action="store_true", help="LongVideoBench: official interleaved subtitles")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0)
    return p.parse_args()


# --- copied from VideoLLaMA3/evaluation/benchmarks/longvideobench.py -------------------

def timestamp_to_seconds(timestamp):
    h, m, s = timestamp.split(':')
    return int(h) * 3600 + int(m) * 60 + float(s)


def insert_subtitles_into_frames(frames, frame_timestamps, subtitles, starting_timestamp_for_subtitles, duration):
    interleaved_list = []
    cur_i = 0
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            if not isinstance(end, float):
                end = duration
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["text"]
        else:
            start, end = subtitle["start"], subtitle["end"]
            start = timestamp_to_seconds(start)
            end = timestamp_to_seconds(end)
            start -= starting_timestamp_for_subtitles
            end -= starting_timestamp_for_subtitles
            subtitle_timestamp = (start + end) / 2
            subtitle_text = subtitle["line"]
        for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
            if frame_timestamp <= subtitle_timestamp:
                interleaved_list.append(frame)
                cur_i += 1
            else:
                break
        if end - start < 1:
            end = subtitle_timestamp + 0.5
            start = subtitle_timestamp - 0.5
        covering_frames = False
        for frame, frame_timestamp in zip(frames, frame_timestamps):
            if frame_timestamp < end and frame_timestamp > start:
                covering_frames = True
                break
        if covering_frames:
            interleaved_list.append(subtitle_text)
    for i, (frame, frame_timestamp) in enumerate(zip(frames[cur_i:], frame_timestamps[cur_i:])):
        interleaved_list.append(frame)
    return interleaved_list


def interleaved_content(frames, timestamps, subtitle, start_ts, duration, instruction):
    """official generate_instruction: video chunks split by subtitle strings."""
    items = insert_subtitles_into_frames(frames, timestamps, subtitle, start_ts, duration) + [instruction]
    content, n_cur, idx = [], 0, 0
    for ele in items:
        if isinstance(ele, str):
            if n_cur:
                content.append({"type": "video", "num_frames": n_cur, "timestamps": timestamps[idx - n_cur:idx]})
                n_cur = 0
            content.append({"type": "text", "text": ele})
        else:
            n_cur += 1
            idx += 1
    return content

# ----------------------------------------------------------------------------------------


class Videos(Dataset):
    def __init__(self, groups):
        self.groups = groups

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        video, samples = self.groups[i]
        try:
            frames, ts = load_video(video, fps=1, max_frames=MAX_FRAMES, precise_time=True)
            return dict(video=video, samples=samples, frames=frames, ts=[float(t) for t in ts], error=None)
        except Exception as e:  # noqa: BLE001
            return dict(video=video, samples=samples, error=repr(e))


def main():
    args = parse_args()
    samples = load_benchmark(args.benchmark, args.data_root)
    lvb_raw = {}
    if args.subtitles:
        assert args.benchmark == "longvideobench", "--subtitles is implemented for LongVideoBench only"
        d = os.path.join(args.data_root, "longvideobench")
        for r in json.load(open(os.path.join(d, "lvb_val.json"))):
            lvb_raw[r["id"]] = (json.load(open(os.path.join(d, "subtitles", r["subtitle_path"]))),
                                r["starting_timestamp_for_subtitles"], r["duration"])
    done = set()
    if os.path.exists(args.output):
        done = {json.loads(l)["qid"] for l in open(args.output) if l.strip()}
    groups = OrderedDict()
    for s in samples:
        groups.setdefault(s["video"], []).append(s)
    groups = sorted(groups.items())[args.shard_id::args.num_shards]
    if args.limit:
        groups = groups[: args.limit]
    todo = [(v, [s for s in ss if s["qid"] not in done]) for v, ss in groups]
    todo = [(v, ss) for v, ss in todo if ss]

    model = Videollama3Qwen2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
    processor = Videollama3Processor.from_pretrained(args.model_path)
    processor.image_processor.max_tokens = args.budget
    min_tokens = processor.image_processor.min_tokens
    tok = processor.tokenizer

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fout = open(args.output, "a")
    loader = DataLoader(Videos(todo), batch_size=None, num_workers=args.num_workers, collate_fn=lambda x: x)
    n_q = sum(len(ss) for _, ss in todo)
    n, t0 = 0, time.time()
    for item in loader:
        if item["error"] is None:
            fit = args.strict_budget and args.budget // len(item["frames"]) < min_tokens
            processor.image_processor.min_tokens = 1 if fit else min_tokens
            image_inputs = processor.process_images([item["frames"]], merge_size=2, return_tensors="pt")
            n_vis = int(sum(g.prod() // 4 for g in image_inputs["grid_sizes"]))
            assert not args.strict_budget or n_vis <= args.budget, (n_vis, args.budget)
        for s in item["samples"]:
            rec = dict(qid=s["qid"], video=s["video"], answer=s["answer"], meta=s["meta"], pred="", response="",
                       mode="vl3_raw", vision_tokens=0, n_frames=0)
            if item["error"] is not None:
                rec["error"] = item["error"]
            else:
                try:
                    instruction = build_prompt(args.benchmark, s)
                    if args.subtitles:
                        sub, st, dur = lvb_raw[s["qid"]]
                        content = interleaved_content(item["frames"], item["ts"], sub, st, dur, instruction)
                    else:
                        content = [{"type": "video", "num_frames": len(item["frames"]), "timestamps": item["ts"]},
                                   {"type": "text", "text": instruction}]
                    prompt = processor.apply_chat_template([{"role": "user", "content": content}],
                                                           tokenize=False, add_generation_prompt=True)
                    x = processor.process_text(prompt, image_inputs, return_tensors="pt")
                    with torch.inference_mode():
                        out = model.generate(
                            input_ids=x["input_ids"].cuda(), attention_mask=x["attention_mask"].cuda(),
                            pixel_values=image_inputs["pixel_values"].cuda().to(torch.bfloat16),
                            grid_sizes=image_inputs["grid_sizes"].cuda(), merge_sizes=image_inputs["merge_sizes"].cuda(),
                            modals=["video"], do_sample=False, max_new_tokens=args.max_new_tokens,
                            pad_token_id=tok.eos_token_id)
                    resp = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
                    rec.update(response=resp, pred=extract_choice(args.benchmark, resp, s["options"]),
                               seq_len=int(x["input_ids"].shape[1]), vision_tokens=n_vis, n_frames=len(item["frames"]))
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec["error"] = repr(e)
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n += 1
        if n % 20 < len(item["samples"]):
            el = time.time() - t0
            print(f"  {n}/{n_q} questions, {el / n:.2f}s/q, eta {el / n * (n_q - n) / 60:.0f} min", flush=True)
    print(f"[{args.benchmark} shard {args.shard_id}] done: {n} questions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
