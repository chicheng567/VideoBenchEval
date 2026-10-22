"""The budgeted pipeline, forced into the Phase-1 layout (every adaptive group
compressed), must feed the LLM exactly what the training code path does
(Videollama3Qwen2ForCausalLM.prepare_inputs_labels_for_multimodal on a whole-video
compression part). Run on one GPU:

    PYTHONPATH=third_party/OnlineLong:. PATH=.venv/bin:$PATH .venv/bin/python tests/test_equivalence.py
"""
import os
import sys

import torch

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.mm_utils import load_video
from videollama3.train.data.global_compressor import build_range_ts_info

from vbe import budget as bd
from vbe.benchmarks import build_prompt, load_benchmark
from vbe.infer import BudgetedVideoLLM, decode_video

MODEL = os.environ.get("MODEL", "checkpoints/phase1_qbase_internvid")
B = 53820


def training_path(llm, frames, ts, prompt):
    """Inputs exactly as videollama3/train/data/global_compressor.py builds them."""
    n = len(frames)
    ip = llm.ip
    ip.max_tokens, ip.max_tokens_per_frame = B, bd.MAX_FRAME_TOKENS
    content = {"type": "video", "num_frames": n, "timestamps": ts}
    msgs = [{"role": "user", "content": [content, {"type": "text", "text": prompt}]}]
    x = llm.processor(images=[frames], text=msgs, merge_size=2, return_tensors="pt")
    n_vis = int((x["input_ids"] == llm.image_token_id).sum())
    m = llm.model
    with torch.inference_mode():
        out = m.prepare_inputs_labels_for_multimodal(
            input_ids=x["input_ids"].cuda(), attention_mask=x["attention_mask"].cuda(),
            position_ids=torch.arange(x["input_ids"].shape[1], device="cuda"), past_key_values=None, labels=None,
            pixel_values=x["pixel_values"].cuda().to(torch.bfloat16), grid_sizes=x["grid_sizes"].cuda(),
            merge_sizes=x["merge_sizes"].cuda(), modals=["video"],
            compression_parts=[[0, n_vis]], compression_ts_info=build_range_ts_info(content, llm.tok),
            compression_frame_sec=[[int(round(float(t))) for t in ts]], compression_is_image=[False])
    return out[4][0]  # inputs_embeds (1, L, D) -> (L, D)


def vl3_raw_path(llm, frames, ts, prompt):
    """Plain VideoLLaMA3 (no compression) with the ORIGINAL projector swapped in."""
    n = len(frames)
    ip = llm.ip
    ip.max_tokens, ip.max_tokens_per_frame = B, bd.MAX_FRAME_TOKENS
    msgs = [{"role": "user", "content": [{"type": "video", "num_frames": n, "timestamps": ts},
                                         {"type": "text", "text": prompt}]}]
    x = llm.processor(images=[frames], text=msgs, merge_size=2, return_tensors="pt")
    m, inner = llm.model, llm.inner
    saved = inner.mm_projector
    inner.mm_projector = llm.proj_orig
    try:
        with torch.inference_mode():
            out = m.prepare_inputs_labels_for_multimodal(
                input_ids=x["input_ids"].cuda(), attention_mask=x["attention_mask"].cuda(),
                position_ids=torch.arange(x["input_ids"].shape[1], device="cuda"), past_key_values=None,
                labels=None, pixel_values=x["pixel_values"].cuda().to(torch.bfloat16),
                grid_sizes=x["grid_sizes"].cuda(), merge_sizes=x["merge_sizes"].cuda(), modals=["video"])
    finally:
        inner.mm_projector = saved
    return out[4][0]


def main():
    llm = BudgetedVideoLLM(MODEL, "checkpoints/vl3_orig_projector.pt", B, 128)
    samples = [s for s in load_benchmark("videomme") if s["meta"]["duration"] == "short"][:9:3]
    ok = True
    for s in samples:
        prompt = build_prompt("videomme", s)
        frames, ts, tok, info = decode_video(s["video"], B)
        video = llm.build_video(frames, ts, tok)
        if video["stats"]["mode"] != "raw":
            continue
        emb_a = vl3_raw_path(llm, frames, ts, prompt)
        _, emb_b = llm.prompt_embeds(video, prompt)
        same_len = emb_a.shape == emb_b.shape
        diff = (emb_a.float() - emb_b.float()).abs().max().item() if same_len else float("nan")
        print(f"[raw] {s['qid']}: {len(frames)} frames, {video['stats']}, len {tuple(emb_a.shape)} vs "
              f"{tuple(emb_b.shape)}, max|diff| {diff:.3e}")
        ok &= same_len and diff == 0.0
    for s in samples:
        prompt = build_prompt("videomme", s)
        frames_a, ts_a = load_video(s["video"], fps=1, max_frames=180)
        if len(frames_a) >= 180:
            continue
        emb_a = training_path(llm, frames_a, [float(t) for t in ts_a], prompt)
        frames_b, ts_b, tok, info = decode_video(s["video"], B)
        assert len(frames_b) == len(frames_a) and ts_b == [float(t) for t in ts_a], "decode differs"
        video = llm.build_video(frames_b, ts_b, tok, force_mode="all_compressed")
        _, emb_b = llm.prompt_embeds(video, prompt)
        same_len = emb_a.shape == emb_b.shape
        diff = (emb_a.float() - emb_b.float()).abs().max().item() if same_len else float("nan")
        print(f"{s['qid']}: {len(frames_a)} frames, {video['stats']}, len {tuple(emb_a.shape)} vs "
              f"{tuple(emb_b.shape)}, max|diff| {diff:.3e}")
        ok &= same_len and diff == 0.0
    print("EQUIVALENT" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
