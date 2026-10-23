"""Budgeted multiple-choice inference for VideoLLaMA3 + qbase compressor checkpoints.

Per video (see vbe/budget.py for the rules):
  decode (1 fps, or lower so that T <= B//64*8 frames) -> one resolution for all frames -> frozen encoder
  -> plan: all raw / hybrid (lowest-variation groups compressed) / forced 8-frame groups
  -> compressed groups: qbase over the WHOLE video's encoder tokens (as in training, so
     Option A's reference stats are the full window's) -> checkpoint mm_projector
  -> raw frames: ORIGINAL VideoLLaMA3 mm_projector (--raw_mode orig, Phase-1 checkpoints), or
     (--raw_mode phase3) as Phase-3 training sends them (videollama3/model/hybrid_raw.py):
     at most --raw_max_tokens rows per frame, re-encoded from the frame's pixels when larger,
     through the checkpoint mm_projector
  -> LLM input in time order:
       raw frame        Time {t:.1f}s:<hw rows>   followed by "," (or "\n" after the last block)
       compressed group Time:{a}s-{b}s:<|compression_start|><K rows><|compression_end|>
     a / b = integer seconds of the group's first / last frame.

One process per GPU; videos sharded round-robin; results appended to a jsonl and
resumed by qid.
"""
import argparse
import itertools
import json
import math
import os
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import ffmpeg
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import DynamicCache

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.mm_utils import load_video
from videollama3.model import Videollama3Qwen2ForCausalLM
from videollama3.model.processor import Videollama3Processor
from videollama3.model.projector import build_vision_projector

from vbe import budget as bd
from vbe import tf_compress as tfc
from vbe.benchmarks import DEFAULT_ROOT, build_prompt, extract_choice, load_benchmark

VIDEO_SLOT = "<<<VIDEO>>>"
NATIVE_DECODE_BYTES = 2e9  # decode at native size (processor resizes, as in training) below this


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--orig_projector", default="",
                   help="scripts/extract_orig_projector.py output (required for --raw_mode orig)")
    p.add_argument("--raw_mode", default="orig", choices=["orig", "phase3"],
                   help="orig: raw frames at the groups' resolution, original VL3 projector; "
                        "phase3: <= --raw_max_tokens rows per raw frame, checkpoint projector")
    p.add_argument("--raw_max_tokens", type=int, default=256,
                   help="--raw_mode phase3: per raw frame cap (hybrid_raw_max_tokens_per_frame)")
    p.add_argument("--benchmark", required=True, choices=["videomme", "mlvu", "longvideobench"])
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--output", required=True,
                   help="jsonl, appended to / resumed from; '{budget}' in it is replaced per budget")
    p.add_argument("--budgets", default="16384",
                   help="comma-separated LLM-side vision-token budgets; each video is decoded and encoded "
                        "once for all of them")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--encoder_chunk_frames", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--prep_threads", type=int, default=8, help="threads for the CPU image processor")
    p.add_argument("--tf_method", default="none",
                   help="with --protocol vl3: comma-separated training-free baselines (fastvid, visionzip, dycoke, "
                        "uniform; '{method}' in --output) that fit the 180-frame official input into B "
                        "(vbe/tf_compress.py). fastvid/visionzip prune the 16K-resolution input to B; dycoke runs "
                        "TTM on an input resized to B / 0.475; uniform keeps the 16K resolution and drops frames")
    p.add_argument("--no_system_prompt", action="store_true",
                   help="drop VideoLLaMA3's default system prompt (the official benchmark eval does not add it)")
    p.add_argument("--limit", type=int, default=0, help="only the first N videos of this shard (debug)")
    p.add_argument("--force_mode", default="", choices=["", "all_compressed"],
                   help="debug: all_compressed = the Phase-1 training layout")
    p.add_argument("--protocol", default="budget", choices=["budget", "vl3"],
                   help="vl3 = original VideoLLaMA3: official load_video (1 fps, <= 180 frames), every frame raw, "
                        "the official processor resize to max_visual_tokens = B (per-frame floor 16 tokens, "
                        "dropped when B // T < 16 so the video stays <= B); no compression")
    return p.parse_args()


VL3_MAX_FRAMES = 180
VL3_MIN_TOKENS = 16  # original VideoLLaMA3 preprocessor_config


def vl3_frame_cap(budget: int, n_frames: int):
    """(per-frame token cap, floor) of the official resize: simple_batched_resize gives each of the
    video's frames max_tokens * 28 * 28 // n_frames pixels."""
    floor = VL3_MIN_TOKENS if budget // n_frames >= VL3_MIN_TOKENS else 1
    return (budget * bd.FACTOR * bd.FACTOR // n_frames) / (bd.FACTOR * bd.FACTOR), floor


# ----------------------------------------------------------------------------- decoding

def decode_video(path: str, budget: int):
    """Frames at 1 fps, or -- when 1 fps would exceed max_frames = B // 64 * 8 -- at
    ~max_frames / duration fps (then evenly thinned to max_frames), so ffmpeg only
    emits about as many frames as the budget allows.
    Returns the frames, their timestamps and the per-frame token cap. Same ffmpeg pipeline
    as videollama3.mm_utils.load_video (identical output at 1 fps), but large videos are
    scaled inside ffmpeg so thousands of frames fit in memory."""
    probe = ffmpeg.probe(path)
    duration = float(probe["format"]["duration"])
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    start = float(vs.get("start_time", 0.0) or 0.0)
    max_frames = bd.max_frames_for(budget)
    fps = bd.FPS if int(duration * bd.FPS) + 1 <= max_frames else (max_frames + 1) / duration
    n_est = min(int(duration * fps) + 1, max_frames)
    tok = bd.frame_tokens(h, w, budget, n_est)
    if n_est * h * w * 3 <= NATIVE_DECODE_BYTES:
        dw, dh = w, h
    else:
        dh, dw = bd.frame_size(h, w, tok)
    stream = ffmpeg.input(path)
    stream = ffmpeg.filter(stream, "fps", fps=fps, round="down")
    if (dw, dh) != (w, h):
        stream = ffmpeg.filter(stream, "scale", dw, dh)
    stream = ffmpeg.output(stream, "pipe:", format="rawvideo", pix_fmt="rgb24")
    out, _ = ffmpeg.run(stream, capture_stdout=True, capture_stderr=True, quiet=True)
    frames = np.frombuffer(out, np.uint8).reshape([-1, dh, dw, 3]).transpose([0, 3, 1, 2])
    ts = (start + np.arange(len(frames)) / fps)
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
    def __init__(self, groups, budget, protocol="budget"):
        self.groups, self.budget, self.protocol = groups, budget, protocol

    def __len__(self):
        return len(self.groups)

    def __getitem__(self, i):
        video, samples = self.groups[i]
        try:
            if self.protocol == "vl3":
                frames, ts = load_video(video, fps=1, max_frames=VL3_MAX_FRAMES, precise_time=True)
                # (T, H, W, C): every budget re-runs the processor on these frames; channel-last makes its
                # to_pil_image a plain copy instead of a transpose (bit-identical pixel_values, ~2-3x faster)
                frames = np.stack(frames).transpose(0, 2, 3, 1)
                ts = [float(t) for t in ts]
                h, w = frames.shape[1:3]
                dur = float(ffmpeg.probe(video)["format"]["duration"])
                fps = min(bd.FPS, len(ts) / max(dur, 1e-6))
                info = dict(duration=dur, native_hw=[int(h), int(w)], n_frames=len(ts), fps=round(fps, 4),
                            fps_ok=fps >= bd.MIN_FPS - 1e-6)
                return dict(video=video, samples=samples, frames=np.ascontiguousarray(frames), ts=ts, tok=None,
                            info=info, error=None)
            frames, ts, tok, info = decode_video(video, self.budget)
            return dict(video=video, samples=samples, frames=frames, ts=ts, tok=tok, info=info, error=None)
        except Exception as e:  # unreadable video: its questions are recorded as unanswered
            return dict(video=video, samples=samples, error=repr(e))


# ----------------------------------------------------------------------------- model side

class BudgetedVideoLLM:
    def __init__(self, model_path, orig_projector_path, budget, encoder_chunk_frames, prep_threads=8,
                 raw_mode="orig", raw_max_tokens=256):
        self.model = Videollama3Qwen2ForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2").cuda().eval()
        self.inner = self.model.get_model()
        if getattr(self.inner, "token_compressor", None) is None:
            raise RuntimeError(f"{model_path} has no token_compressor")
        self.processor = Videollama3Processor.from_pretrained(model_path, trust_remote_code=False)
        self.tok = self.processor.tokenizer
        self.ip = self.processor.image_processor
        self.min_tokens = self.ip.min_tokens
        self.system_prompt = True  # VideoLLaMA3's default system prompt (training); official benchmark eval: False
        self.raw_mode, self.raw_max_tokens = raw_mode, raw_max_tokens
        if raw_mode == "orig":
            self.proj_orig = build_vision_projector(self.model.config, self.model.config.mm_hidden_size)
            self.proj_orig.load_state_dict(torch.load(orig_projector_path, map_location="cpu", weights_only=True))
            self.proj_orig = self.proj_orig.to(device="cuda", dtype=torch.bfloat16).eval()
        else:  # Phase 3 trained raw rows through the checkpoint's own projector
            from videollama3.model import hybrid_raw  # Phase-3 model code (OnlineLong_ft)
            self.hybrid_raw = hybrid_raw
            self.proj_orig = self.inner.mm_projector
        self.budget, self.chunk = budget, encoder_chunk_frames
        self.pieces_per_chunk = 8
        self.pool = ThreadPoolExecutor(max_workers=prep_threads)
        self.image_token_id = self.tok.convert_tokens_to_ids(DEFAULT_IMAGE_TOKEN)
        self.cs = self.tok.convert_ids_to_tokens(self.model.config.compression_start_token_id)
        self.ce = self.tok.convert_ids_to_tokens(self.model.config.compression_end_token_id)

    @torch.inference_mode()
    def encode_with_stats(self, frames, tok, min_tokens=None):
        """``encode`` + the last encoder layer's per-frame attention stats (vbe.tf_compress)."""
        if not hasattr(self, "stats"):
            self.stats = tfc.LastLayerStats(self.inner.get_vision_encoder().vision_encoder)
        self.stats.enabled = True
        self.stats.reset()
        try:
            feats, grid = self.encode(frames, tok, min_tokens)
        finally:
            self.stats.enabled = False
        st = self.stats
        out = dict(fastvid=torch.stack(st.fastvid), vzip=torch.stack(st.vzip), vzip_key=torch.stack(st.vzip_key))
        assert out["fastvid"].shape[:2] == (len(frames), grid[0] * grid[1]), (out["fastvid"].shape, grid)
        st.reset()
        return feats, grid, out

    @torch.inference_mode()
    def build_tf_video(self, feats, grid, ts, method, budget, stats=None):
        """Original-VL3 layout (one "Time t s:<rows>" block per frame, original projector) after a
        training-free method picked each frame's rows; frames left with no rows are dropped."""
        T, hw = len(ts), grid[0] * grid[1]
        rows = self.proj_orig(feats).view(T, hw, -1)
        if method == "fastvid":
            torch.manual_seed(0)  # FastVID adds 1e-6 noise to its density scores
            per = tfc.fastvid(rows, stats["fastvid"].to(rows.device), min(1.0, budget / (T * hw)))
        elif method == "visionzip":
            per = tfc.visionzip(rows, stats["vzip"].to(rows.device), stats["vzip_key"].to(rows.device),
                                min(hw, budget // T))
        elif method == "dycoke":
            per = tfc.dycoke_ttm(rows)
        else:  # uniform: frames already subsampled, all rows raw
            per = list(rows)
        keep = [f for f in range(T) if per[f].shape[0] > 0]
        text, out_rows, counts = [], [], []
        for j, f in enumerate(keep):
            text.append(f"Time {round(ts[f], 1)}s:{DEFAULT_IMAGE_TOKEN}" + ("\n" if j == len(keep) - 1 else ","))
            out_rows.append(per[f]); counts.append(int(per[f].shape[0]))
        stats_out = dict(mode=method, hw=hw, n_groups=len(keep), n_compressed=0, n_raw_frames=len(keep),
                         vision_tokens=sum(counts))
        return dict(rows=torch.cat(out_rows), counts=counts, text="".join(text), stats=stats_out)

    @torch.inference_mode()
    def encode(self, frames, tok, min_tokens=None):
        """Encoder tokens (T*hw, C) of every frame at <= tok merged tokens per frame.
        The image processor (CPU) runs on SUB-frame pieces in a thread pool, ahead of
        the GPU; each GPU call takes ``chunk`` frames as several pieces (the encoder
        attends within a frame, so splitting a chunk into pieces changes nothing)."""
        self.ip.max_tokens, self.ip.max_tokens_per_frame = 10 ** 9, tok
        self.ip.min_tokens = self.min_tokens if min_tokens is None else min_tokens
        enc = self.inner.get_vision_encoder()
        piece = max(1, self.chunk // self.pieces_per_chunk)
        starts = list(range(0, len(frames), piece))

        def prep(s0):
            return self.ip(images=[list(frames[s0:s0 + piece])], merge_size=2, return_tensors="pt")

        feats, grid, pending = [], None, deque()
        it = iter(starts)
        for s0 in itertools.islice(it, max(2 * self.pieces_per_chunk, self.pool._max_workers)):
            pending.append(self.pool.submit(prep, s0))
        while pending:
            xs = []
            while pending and len(xs) < self.pieces_per_chunk:
                xs.append(pending.popleft().result())
                nxt = next(it, None)
                if nxt is not None:
                    pending.append(self.pool.submit(prep, nxt))
            for x in xs:
                g = x["grid_sizes"][0].tolist()
                assert grid is None or g[1:] == grid[1:], "frame size changed between chunks"
                grid = g
            feats.append(enc(pixel_values=torch.cat([x["pixel_values"] for x in xs]).cuda().to(torch.bfloat16),
                             grid_sizes=torch.cat([x["grid_sizes"] for x in xs]).cuda(),
                             merge_sizes=torch.cat([x["merge_sizes"] for x in xs]).cuda()))
        feats = torch.cat(feats)
        hh, ww = grid[1] // 2, grid[2] // 2
        assert feats.shape[0] == len(frames) * hh * ww, (feats.shape, len(frames), hh, ww)
        return feats, (hh, ww)

    @torch.inference_mode()
    def build_video(self, frames, ts, tok, force_mode="", budget=None, enc=None, frame_idx=None):
        """Vision rows of the whole video (in LLM order) + the video's text with one
        <image> per block, ready to be spliced into any question's prompt.
        ``enc`` = precomputed ``encode(frames, tok)`` (the encoder is per-frame, so a
        frame subset's features are rows of a larger set's). ``frame_idx``: indices into
        ``frames`` of the ``ts`` frames when ``frames`` is a larger decode (only read
        when Phase-3 raw frames are re-encoded)."""
        budget = self.budget if budget is None else budget
        T = len(ts)
        feats, (hh, ww) = self.encode(frames, tok) if enc is None else enc
        hw = hh * ww
        # merged grid of a raw frame: the groups' own grid, or Phase 3's capped one
        raw_grid = ((hh, ww) if self.raw_mode == "orig"
                    else self.hybrid_raw.raw_frame_grid(hh, ww, self.raw_max_tokens))
        raw_hw = raw_grid[0] * raw_grid[1]
        per_frame = feats.view(T, hw, -1).float().mean(1)
        plan = bd.plan_groups(per_frame, hw, budget, force_mode, raw_hw=raw_hw)

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
            if raw_hw == hw:
                sel = feats.view(T, hw, -1)[torch.tensor(raw_idx, device=feats.device)].reshape(-1, feats.shape[-1])
            else:
                src = raw_idx if frame_idx is None else [frame_idx[f] for f in raw_idx]
                sel = self.reencode(frames[src], tok, raw_grid)
            raw_rows = self.proj_orig(sel)    # orig: original VL3 projector; phase3: checkpoint projector

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
                    rows.append(raw_rows[r * raw_hw:(r + 1) * raw_hw]); counts.append(raw_hw)
                    r += 1
            s += L
        stats = dict(mode=plan.mode, hw=hw, raw_hw=raw_hw, n_groups=len(plan.seg_lens), n_compressed=len(plan.compressed),
                     n_raw_frames=plan.n_raw_frames, vision_tokens=sum(counts))
        assert stats["vision_tokens"] == plan.vision_tokens
        return dict(rows=torch.cat(rows), counts=counts, text="".join(text), stats=stats)

    @torch.inference_mode()
    def reencode(self, frames, tok, grid):
        """Encoder rows of ``frames`` at the merged grid ``grid``, the way Phase-3 training
        makes a raw frame smaller (hybrid_raw.reencode_frames): processed at the groups'
        size ``tok``, then the pixel rows resized and passed through the encoder again."""
        self.ip.max_tokens, self.ip.max_tokens_per_frame = 10 ** 9, tok
        enc = self.inner.get_vision_encoder()

        def encode(pv, gs, ms):
            return enc(pixel_values=pv, grid_sizes=gs, merge_sizes=ms)

        out = []
        for s0 in range(0, len(frames), self.chunk):
            x = self.ip(images=[list(frames[s0:s0 + self.chunk])], merge_size=2, return_tensors="pt")
            pv = x["pixel_values"].cuda().to(torch.bfloat16)
            gs, ms = x["grid_sizes"].cuda(), x["merge_sizes"].cuda()
            out.append(self.hybrid_raw.reencode_frames(encode, pv, gs, ms, 0, range(int(gs[0, 0])), grid))
        return torch.cat(out)

    def prompt_embeds(self, video, prompt):
        msgs = [{"role": "user", "content": [{"type": "text", "text": VIDEO_SLOT + prompt}]}]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, add_system_prompt=self.system_prompt)
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
    def answer_many(self, video, prompts, max_new_tokens):
        """Greedy answers to several questions about one video. The shared prefix
        (system + user header + video) is prefilled once; each question then only runs
        its own text and the answer tokens on that KV cache (cropped back to the prefix
        between questions). Falls back to
        ``answer`` when the prefix does not tokenize identically inside a prompt."""
        msgs = [{"role": "user", "content": [{"type": "text", "text": VIDEO_SLOT + "Q"}]}]
        full = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, add_system_prompt=self.system_prompt)
        prefix = full[: full.index(VIDEO_SLOT)] + video["text"]
        p_exp = self._expand(self.tok(prefix, add_special_tokens=False).input_ids, video)
        n_rows = len(p_exp)  # prefix length with every <image> expanded to its rows
        outs, lm = [], super(Videollama3Qwen2ForCausalLM, self.model)
        cache = None
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        for prompt in prompts:
            ids, emb = self.prompt_embeds(video, prompt)
            if ids[:n_rows].tolist() != p_exp:
                outs.append(self.answer(video, prompt, max_new_tokens))
                continue
            if cache is None:
                cache = DynamicCache()
                lm.forward(inputs_embeds=emb[None, :n_rows], past_key_values=cache, use_cache=True, logits_to_keep=1)
            else:
                cache.crop(n_rows)
            out = lm.forward(inputs_embeds=emb[None, n_rows:], past_key_values=cache, use_cache=True, logits_to_keep=1)
            new = []
            while True:
                t = int(out.logits[0, -1].argmax())
                if t in eos:
                    break
                new.append(t)
                if len(new) == max_new_tokens:
                    break
                out = lm.forward(input_ids=torch.tensor([[t]], device="cuda"), past_key_values=cache, use_cache=True,
                                 logits_to_keep=1)
            outs.append((self.tok.decode(new, skip_special_tokens=True).strip(), emb.shape[0]))
        return outs

    def _expand(self, p_ids, video):
        """prefix token ids with every <image> repeated to its row count."""
        out, k = [], 0
        for t in p_ids:
            if t == self.image_token_id:
                out += [t] * video["counts"][k]
                k += 1
            else:
                out.append(t)
        return out

    @torch.inference_mode()
    def answer(self, video, prompt, max_new_tokens):
        _, emb = self.prompt_embeds(video, prompt)
        out = super(Videollama3Qwen2ForCausalLM, self.model).generate(
            inputs_embeds=emb[None], attention_mask=torch.ones(1, emb.shape[0], device="cuda", dtype=torch.long),
            max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return self.tok.batch_decode(out, skip_special_tokens=True)[0].strip(), emb.shape[0]


# ----------------------------------------------------------------------------- main

class _Done(Exception):
    """control flow: the training-free branch built the video"""


def _fit_cap(llm, frames, budget, enc_cache, with_stats):
    """(tok, floor) of the official resize at ``budget`` over all frames, with the vl3 protocol's
    floor drop when the 16-token floor overshoots; encodes (cached) and returns the encoding."""
    T = len(frames)
    tok, floor = vl3_frame_cap(budget, T)
    for fl in ([floor, 1] if floor != 1 else [1]):
        key = (tok, fl, with_stats)
        if not with_stats and (tok, fl, True) in enc_cache:  # same features, stats unused
            key = (tok, fl, True)
        if key not in enc_cache:
            enc_cache[key] = llm.encode_with_stats(frames, tok, fl) if with_stats else (*llm.encode(frames, tok, fl), None)
        feats, grid, stats = enc_cache[key]
        if T * grid[0] * grid[1] <= budget:
            return feats, grid, stats, fl
    return feats, grid, stats, fl


def run_tf(llm, method, frames, ts, b, enc_cache):
    """One budget of a training-free baseline; returns (video, frame indices, floor)."""
    T = len(ts)
    if method in ("fastvid", "visionzip"):
        # prune the full official input (16K budget over the 180 frames) down to b
        feats, grid, stats, fl = _fit_cap(llm, frames, 16384, enc_cache, True)
        return llm.build_tf_video(feats, grid, ts, method, b, stats), list(range(T)), fl
    if method == "uniform":
        # 16K per-frame resolution, as many evenly spaced frames as b allows
        feats, grid, _, fl = _fit_cap(llm, frames, 16384, enc_cache, False)
        hw = grid[0] * grid[1]
        n = max(1, min(T, b // hw))
        idx = np.linspace(0, T - 1, n).round().astype(int).tolist()
        sel = torch.tensor(idx, device=feats.device)
        f_b = feats.view(T, hw, -1)[sel].reshape(-1, feats.shape[-1])
        return llm.build_tf_video(f_b, grid, [ts[k] for k in idx], method, b), idx, fl
    # dycoke: resize so TTM (keeps ~0.475) lands at b; shrink the input until it fits
    target = b / tfc.DYCOKE_KEEP
    for _ in range(60):
        feats, grid, _, fl = _fit_cap(llm, frames, int(target), enc_cache, False)
        video = llm.build_tf_video(feats, grid, ts, method, b)
        if video["stats"]["vision_tokens"] <= b:
            return video, list(range(T)), fl
        target *= 0.97
    raise RuntimeError(f"dycoke could not fit {b}")


def main():
    args = parse_args()
    budgets = sorted(int(b) for b in args.budgets.split(","))
    assert len(budgets) == 1 or "{budget}" in args.output, "several budgets need '{budget}' in --output"
    methods = [m for m in args.tf_method.split(",") if m]
    assert all(m in ("none", "fastvid", "visionzip", "dycoke", "uniform") for m in methods), methods
    assert methods == ["none"] or args.protocol == "vl3", "--tf_method needs --protocol vl3"
    assert len(methods) == 1 or "{method}" in args.output, "several methods need '{method}' in --output"
    # a "budget" key below is (method, budget)
    keys = [(m, b) for m in methods for b in budgets]
    outputs = {k: args.output.replace("{budget}", str(k[1])).replace("{method}", k[0]) for k in keys}
    samples = load_benchmark(args.benchmark, args.data_root)
    done = {k: set() for k in keys}
    for k, path in outputs.items():
        if os.path.exists(path):
            done[k] = {json.loads(l)["qid"] for l in open(path) if l.strip()}
    groups = OrderedDict()
    for s in samples:
        groups.setdefault(s["video"], []).append(s)
    groups = sorted(groups.items())[args.shard_id :: args.num_shards]
    if args.limit:
        groups = groups[: args.limit]
    # per video: {budget: questions still to answer}
    todo = [(v, {k: [s for s in ss if s["qid"] not in done[k]] for k in keys}) for v, ss in groups]
    todo = [(v, {b: q for b, q in per.items() if q}) for v, per in todo]
    todo = [(v, per) for v, per in todo if per]
    n_q = sum(len(q) for _, per in todo for q in per.values())
    print(f"[{args.benchmark} shard {args.shard_id}/{args.num_shards}] methods {methods} budgets {budgets}: "
          f"{len(todo)} videos / "
          f"{n_q} (question, budget) pairs to go", flush=True)
    if not todo:
        return

    assert args.raw_mode != "orig" or args.orig_projector, "--raw_mode orig needs --orig_projector"
    llm = BudgetedVideoLLM(args.model_path, args.orig_projector, budgets[-1], args.encoder_chunk_frames,
                           args.prep_threads, raw_mode=args.raw_mode, raw_max_tokens=args.raw_max_tokens)
    llm.system_prompt = not args.no_system_prompt
    # decode once for the largest budget still needed by the video; smaller budgets take a subset
    loader = DataLoader(Videos([(v, per) for v, per in todo], budgets[-1], args.protocol), batch_size=None,
                        shuffle=False,
                        num_workers=args.num_workers, prefetch_factor=1 if args.num_workers else None)
    fouts = {}
    for k, path in outputs.items():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fouts[k] = open(path, "a")
    t0, n_done = time.time(), 0
    tm = dict(wait=0.0, encode=0.0, build=0.0, answer=0.0)  # where the time goes (GPU stages synced)
    t_last = time.time()
    for item in loader:
        tm["wait"] += time.time() - t_last
        per = item["samples"]
        frames, ts, info, err = item.get("frames"), item.get("ts"), item.get("info"), item["error"]
        enc_cache = {}  # tok -> encoder output of ALL decoded frames at that size
        for key in sorted(per):
            method, b = key
            video, err_b, binfo = None, err, None
            if err is None:
                try:
                    if method != "none":
                        t1 = time.time()
                        video, idx, floor = run_tf(llm, method, frames, ts, b, enc_cache)
                        assert video["stats"]["vision_tokens"] <= b, (video["stats"], b)
                        torch.cuda.synchronize()
                        tm["build"] += time.time() - t1
                        fps = min(bd.FPS, len(idx) / max(info["duration"], 1e-6))
                        binfo = dict(info, n_frames=len(idx), fps=round(fps, 4), fps_ok=fps >= bd.MIN_FPS - 1e-6,
                                     min_tokens=floor)
                        raise _Done
                    if args.protocol == "vl3":
                        idx = list(range(len(ts)))
                        tok, floor = vl3_frame_cap(b, len(ts))
                    else:
                        idx = bd.subsample(len(ts), b)
                        tok, floor = bd.frame_tokens(*info["native_hw"], b, len(idx)), None
                    ts_b = [ts[k] for k in idx]
                    if (tok, floor) not in enc_cache:
                        t1 = time.time()
                        enc_cache[(tok, floor)] = llm.encode(frames, tok, floor)
                        torch.cuda.synchronize()
                        tm["encode"] += time.time() - t1
                    t1 = time.time()
                    feats, grid = enc_cache[(tok, floor)]
                    hw = grid[0] * grid[1]
                    if args.protocol == "vl3" and len(idx) * hw > b and floor != 1:
                        # the 16-token floor's round-up overshot B (e.g. 456x718 at B=4096 -> 24 x 180 > 4096):
                        # drop the floor for this video so the official resize alone fits
                        floor = 1
                        if (tok, floor) not in enc_cache:
                            enc_cache[(tok, floor)] = llm.encode(frames, tok, floor)
                        feats, grid = enc_cache[(tok, floor)]
                        hw = grid[0] * grid[1]
                    sel = torch.tensor(idx, device=feats.device)
                    feats_b = feats.view(len(ts), hw, -1)[sel].reshape(-1, feats.shape[-1])
                    video = llm.build_video(frames, ts_b, tok, "raw" if args.protocol == "vl3" else args.force_mode,
                                            budget=b, enc=(feats_b, grid), frame_idx=idx)
                    assert video["stats"]["vision_tokens"] <= b, (video["stats"], b)
                    torch.cuda.synchronize()
                    tm["build"] += time.time() - t1
                    fps = min(bd.FPS, len(idx) / max(info["duration"], 1e-6))
                    binfo = dict(info, n_frames=len(idx), fps=round(fps, 4), fps_ok=fps >= bd.MIN_FPS - 1e-6)
                    if args.protocol == "vl3":
                        binfo["min_tokens"] = floor
                except _Done:
                    pass
                except torch.cuda.OutOfMemoryError as e:
                    err_b = repr(e)
                    if os.environ.get("VBE_DEBUG"):
                        import traceback; traceback.print_exc()
                    torch.cuda.empty_cache()
            answers = []
            if err_b is None:
                try:
                    t1 = time.time()
                    answers = llm.answer_many(video, [build_prompt(args.benchmark, s) for s in per[key]],
                                              args.max_new_tokens)
                    tm["answer"] += time.time() - t1
                except torch.cuda.OutOfMemoryError as e:
                    if os.environ.get("VBE_DEBUG"):
                        import traceback; traceback.print_exc()
                    torch.cuda.empty_cache()
                    err_b = repr(e)
            for k, s in enumerate(per[key]):
                rec = dict(qid=s["qid"], video=os.path.basename(s["video"]), answer=s["answer"], meta=s["meta"])
                if err_b is not None:
                    rec.update(pred="", response="", error=err_b)
                else:
                    resp, seq_len = answers[k]
                    rec.update(pred=extract_choice(args.benchmark, resp, s["options"]), response=resp,
                               seq_len=seq_len, video_info=binfo, **video["stats"])
                fouts[key].write(json.dumps(rec, ensure_ascii=False) + "\n")
                fouts[key].flush()
                n_done += 1
            del video
        del enc_cache
        torch.cuda.empty_cache()
        if n_done and n_done % 50 < sum(len(q) for q in per.values()):
            el = time.time() - t0
            print(f"  {n_done}/{n_q} questions, {el / n_done:.2f}s/q, eta {(n_q - n_done) * el / n_done / 60:.0f} min "
                  f"| " + " ".join(f"{k} {v / el:.0%}" for k, v in tm.items()), flush=True)
        t_last = time.time()
    for f in fouts.values():
        f.close()
    print(f"[{args.benchmark} shard {args.shard_id}] done: {n_done} questions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
