"""Budgeted multiple-choice inference for VideoLLaMA3 + qbase compressor checkpoints.

Per video (see vbe/budget.py for the rules):
  decode (1 fps, <= B//64*8 frames) -> one resolution for all frames -> frozen encoder
  -> plan: all raw / hybrid (lowest-variation groups compressed) / forced 8-frame groups
  -> compressed groups: qbase over the WHOLE video's encoder tokens (as in training, so
     Option A's reference stats are the full window's) -> checkpoint mm_projector
  -> raw frames: ORIGINAL VideoLLaMA3 mm_projector
  -> LLM input in time order:
       raw frame        Time {t:.1f}s:<hw rows>   followed by "," (or "\n" after the last block)
       compressed group Time:{a}s-{b}s:<|compression_start|><K rows><|compression_end|>
     a / b = integer seconds of the group's first / last frame.

One process per GPU; videos sharded round-robin; results appended to a jsonl and
resumed by qid.
"""
import argparse
import json
import math
import os
import time
from collections import OrderedDict

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ffmpeg
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.model.projector import build_vision_projector

from vbe import budget as bd
from vbe.benchmarks import DEFAULT_ROOT, build_prompt, extract_choice, load_benchmark

VIDEO_SLOT = "<<<VIDEO>>>"
NATIVE_DECODE_BYTES = 2e9  # decode at native size (processor resizes, as in training) below this


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--orig_projector", required=True, help="scripts/extract_orig_projector.py output")
    p.add_argument("--benchmark", required=True, choices=["videomme", "mlvu", "longvideobench"])
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--output", required=True, help="jsonl, appended to / resumed from")
    p.add_argument("--budget", type=int, default=53820, help="LLM-side vision tokens per video")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--encoder_chunk_frames", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="only the first N videos of this shard (debug)")
    p.add_argument("--force_mode", default="", choices=["", "all_compressed"],
                   help="debug: all_compressed = the Phase-1 training layout")
    return p.parse_args()


# ----------------------------------------------------------------------------- decoding

def decode_video(path: str, budget: int):
    """Frames at 1 fps (uniformly subsampled to max_frames if longer), the timestamps,
    and the per-frame token cap. Same ffmpeg pipeline as videollama3.mm_utils.load_video,
    but long videos are scaled inside ffmpeg so thousands of frames fit in memory."""
    probe = ffmpeg.probe(path)
    duration = float(probe["format"]["duration"])
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    start = float(vs.get("start_time", 0.0) or 0.0)
    max_frames = bd.max_frames_for(budget)
    n_est = min(int(duration * bd.FPS) + 1, max_frames)
    tok = bd.frame_tokens(h, w, budget, n_est)
    if n_est * h * w * 3 <= NATIVE_DECODE_BYTES:
        dw, dh = w, h
    else:
        dh, dw = bd.frame_size(h, w, tok)
    stream = ffmpeg.input(path)
    stream = ffmpeg.filter(stream, "fps", fps=bd.FPS, round="down")
    if (dw, dh) != (w, h):
        stream = ffmpeg.filter(stream, "scale", dw, dh)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
    out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
    frames = np.frombuffer(out, np.uint8).reshape([-1, dh, dw, 3]).transpose([0, 3, 1, 2])
    ts = np.arange(start, start + duration + 1 / bd.FPS, 1 / bd.FPS)[: len(frames)]
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
        frames, ts = frames[idx], ts[idx]
    T = len(frames)
    assert T > 0, "no frames decoded"
    fps = min(bd.FPS, T / max(duration, 1e-6))
    tok = bd.frame_tokens(h, w, budget, T)  # with the real frame count
    return np.ascontiguousarray(frames), [float(t) for t in ts], tok, dict(
        duration=duration, native_hw=[h, w], n_frames=T, fps=round(fps, 4), fps_ok=fps >= bd.MIN_FPS - 1e-6)


class Videos(Dataset):
    def __init__(self, groups, budget):
        self.groups, self.budget = groups, budget

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        video, samples = self.groups[i]
        try:
            frames, ts, tok, info = decode_video(video, self.budget)
            return dict(video=video, samples=samples, frames=frames, ts=ts, tok=tok, info=info, error=None)
        except Exception as e:  # unreadable video: its questions are recorded as unanswered
            return dict(video=video, samples=samples, error=repr(e))


# ----------------------------------------------------------------------------- model side

class BudgetedVideoLLM:
    def __init__(self, model_path, orig_projector_path, budget, encoder_chunk_frames):
        self.model = Videollama3Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
        self.inner = self.model.get_model()
        if getattr(self.inner, "token_compressor", None) is None:
            raise RuntimeError(f"{model_path} has no token_compressor")
        self.processor = Videollama3Processor.from_pretrained(model_path, trust_remote_code=False)
        self.tok = self.processor.tokenizer
        self.ip = self.processor.image_processor
        self.proj_orig = build_vision_projector(self.model.config, self.model.config.mm_hidden_size)
        self.proj_orig.load_state_dict(torch.load(orig_projector_path, map_location="cpu", weights_only=True))
        self.proj_orig = self.proj_orig.to(device="cuda", dtype=torch.bfloat16).eval()
        self.budget, self.chunk = budget, encoder_chunk_frames
        self.image_token_id = self.tok.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.cs = self.tok.convert_ids_to_tokens(self.model.config.compression_start_token_id)
        self.ce = self.tok.convert_ids_to_tokens(self.model.config.compression_end_token_id)

    @torch.inference_mode()
    def encode(self, frames, tok):
        """Encoder tokens (T*hw, C) of every frame at <= tok merged tokens per frame."""
        self.ip.max_tokens, self.ip.max_tokens_per_frame = 10 ** 9, tok
        enc = self.inner.get_vision_encoder()
        feats, grid = [], None
        for s in range(0, len(frames), self.chunk):
            x = self.ip(images=[list(frames[s:s + self.chunk])], merge_size=2, return_tensors="pt")
            g = x["grid_sizes"][0].tolist()
            assert grid is None or g[1:] == grid[1:], "frame size changed between chunks"
            grid = g
            feats.append(enc(pixel_values=x["pixel_values"].cuda().to(torch.bfloat16),
                             grid_sizes=x["grid_sizes"].cuda(), merge_sizes=x["merge_sizes"].cuda()))
        feats = torch.cat(feats)
        hh, ww = grid[1] // 2, grid[2] // 2
        assert feats.shape[0] == len(frames) * hh * ww, (feats.shape, len(frames), hh, ww)
        return feats, (hh, ww)

    @torch.inference_mode()
    def build_video(self, frames, ts, tok, force_mode=""):
        """Vision rows of the whole video (in LLM order) + the video's text with one
        <image> per block, ready to be spliced into any question's prompt."""
        T = len(frames)
        feats, (hh, ww) = self.encode(frames, tok)
        hw = hh * ww
        per_frame = feats.view(T, hw, -1).float().mean(1)
        plan = bd.plan_groups(per_frame, hw, self.budget, force_mode)

        comp_rows = None
        if plan.compressed:
            qb = self.inner.token_compressor
            cu = torch.tensor([0] + list(np.cumsum([L * hw for L in plan.seg_lens])), device=feats.device)
            saved = qb.adaptive_segmentation
            qb.adaptive_segmentation = False  # we pass the cut; each window -> K rows
            try:
                comp_rows = qb(feats, cu, [(hh, ww)] * len(plan.seg_lens))
            finally:
                qb.adaptive_segmentation = saved
            comp_rows = self.inner.mm_projector(comp_rows)          # checkpoint projector
        starts = np.cumsum([0] + plan.seg_lens[:-1]).tolist()
        raw_idx = [f for j, (s0, L) in enumerate(zip(starts, plan.seg_lens)) if j not in plan.compressed
                   for f in range(s0, s0 + L)]
        raw_rows = None
        if raw_idx:
            sel = feats.view(T, hw, -1)[torch.tensor(raw_idx, device=feats.device)].reshape(-1, feats.shape[-1])
            raw_rows = self.proj_orig(sel)                           # original VL3 projector

        sec = [int(round(t)) for t in ts]
        rows, counts, text, s, r = [], [], [], 0, 0
        n_blocks = sum(1 if j in plan.compressed else L for j, L in enumerate(plan.seg_lens))
        b = 0
        for j, L in enumerate(plan.seg_lens):
            if j in plan.compressed:
                text.append(f"Time:{sec[s]}s-{sec[s + L - 1]}s:{self.cs}{DEFAULT_IMAGE_TOKEN}{self.ce}")
                rows.append(comp_rows[j * bd.K:(j + 1) * bd.K]); counts.append(bd.K)
                b += 1
                if b == n_blocks:
                    text.append("\n")
            else:
                for f in range(s, s + L):
                    b += 1
                    text.append(f"Time {round(ts[f], 1)}s:{DEFAULT_IMAGE_TOKEN}" + ("\n" if b == n_blocks else ","))
                    rows.append(raw_rows[r * hw:(r + 1) * hw]); counts.append(hw)
                    r += 1
            s += L
        stats = dict(mode=plan.mode, hw=hw, n_groups=len(plan.seg_lens), n_compressed=len(plan.compressed),
                     n_raw_frames=plan.n_raw_frames, vision_tokens=sum(counts))
        assert stats["vision_tokens"] == plan.vision_tokens
        return dict(rows=torch.cat(rows), counts=counts, text="".join(text), stats=stats)

    def prompt_embeds(self, video, prompt):
        msgs = [{"role": "user", "content": [{"type": "text", "text": VIDEO_SLOT + prompt}]}]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, add_system_prompt=True)
        assert full.count(VIDEO_SLOT) == 1
        full = full.replace(VIDEO_SLOT, video["text"])
        ids = torch.tensor(self.tok(full, add_special_tokens=False).input_ids, device="cuda")
        is_img = ids == self.image_token_id
        assert int(is_img.sum()) == len(video["counts"]), (int(is_img.sum()), len(video["counts"]))
        rep = torch.ones_like(ids)
        rep[is_img] = torch.tensor(video["counts"], device="cuda")
        ids = ids.repeat_interleave(rep)
        emb = self.inner.embed_tokens(ids)
        emb[ids == self.image_token_id] = video["rows"].to(emb.dtype)
        return ids, emb

    @torch.inference_mode()
    def answer(self, video, prompt, max_new_tokens):
        _, emb = self.prompt_embeds(video, prompt)
        out = super(Videollama3Qwen2ForCausalLM, self.model).generate(
            inputs_embeds=emb[None], attention_mask=torch.ones(1, emb.shape[0], device="cuda", dtype=torch.long),
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip(), emb.shape[0]


# ----------------------------------------------------------------------------- main

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
    print(f"[{args.benchmark} shard {args.shard_id}/{args.num_shards}] budget {args.budget}: {len(todo)} videos / "
          f"{n_q} questions to go ({len(done)} already in {args.output})", flush=True)
    if not todo:
        return

    llm = BudgetedVideoLLM(args.model_path, args.orig_projector, args.budget, args.encoder_chunk_frames)
    loader = DataLoader(Videos(todo, args.budget), batch_size=None, shuffle=False, num_workers=args.num_workers,
                        prefetch_factor=1 if args.num_workers else None)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    fout = open(args.output, "a")
    t0, n_done = time.time(), 0
    for item in loader:
        video, err = None, item["error"]
        if err is None:
            try:
                video = llm.build_video(item["frames"], item["ts"], item["tok"], args.force_mode)
            except torch.cuda.OutOfMemoryError as e:
                err = repr(e)
            torch.cuda.empty_cache()
        for s in item["samples"]:
            rec = dict(qid=s["qid"], video=os.path.basename(s["video"]), answer=s["answer"], meta=s["meta"])
            if err is not None:
                rec.update(pred="", response="", error=err)
            else:
                try:
                    resp, seq_len = llm.answer(video, build_prompt(args.benchmark, s), args.max_new_tokens)
                    rec.update(pred=extract_choice(args.benchmark, resp, s["options"]), response=resp, seq_len=seq_len)
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec.update(pred="", response="", error=repr(e))
                rec.update(video_info=item["info"], **video["stats"])
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            n_done += 1
        if n_done and n_done % 50 < len(item["samples"]):
            el = time.time() - t0
            print(f"  {n_done}/{n_q} questions, {el / n_done:.2f}s/q, eta {(n_q - n_done) * el / n_done / 60:.0f} min",
                  flush=True)
        del video
    fout.close()
    print(f"[{args.benchmark} shard {args.shard_id}] done: {n_done} questions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
