"""Budgeted raw / compressed layout of one video under a vision-token budget B.

Rules (all counts are LLM-side vision tokens = merged 2x2 patches):

1. Frames: 1 fps, at most ``max_frames = B // K * 8`` (longer videos are sampled
   uniformly down to max_frames, i.e. fps < 1; fps must stay >= 0.2).
2. Resolution: one size for every frame, ``min(native, 2048, max(256, B // T))`` tokens
   per frame. If ``T * hw <= B`` every frame goes to the LLM raw -> mode "raw".
3. Otherwise cut the video with the Phase-1 adaptive segmenter (N = T//4 + 1 groups
   of 1-8 frames, cuts at the largest consecutive-frame cosine distances). A group's
   variation is the MEAN of the cosine distances between its consecutive frames' mean
   encoder features. A 1-frame group has no internal distance; it sits between two of
   the largest cuts (a scene change) and saves the least when compressed, so it gets
   variation +inf (compressed last). Compress groups (each -> K tokens) from the lowest
   variation up until the total fits -> mode "hybrid".
4. If even compressing every adaptive group does not fit (N * K > B), cut uniformly
   into 8-frame groups and compress all of them -> mode "forced8"
   (T <= max_frames guarantees ceil(T / 8) * K <= B).
"""
import math
from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple

import torch

from videollama3.model.compressor import adaptive_segment_count, adaptive_segment_lengths

K = 64                  # qbase queries per compressed group
FORCE_EVERY = 8         # max frames per group / forced-group size
TARGET_FRAMES = 4       # adaptive segmenter: N = T // 4 + 1
MIN_FRAME_TOKENS = 256  # resolution floor when shrinking to fit the budget
MAX_FRAME_TOKENS = 2048 # per-frame cap (checkpoint's preprocessor_config)
MIN_TOKENS = 64         # processor floor (checkpoint's preprocessor_config)
FPS = 1.0
MIN_FPS = 0.2
FACTOR = 28             # patch 14 x merge 2


def max_frames_for(budget: int) -> int:
    return budget // K * FORCE_EVERY


def subsample(n_frames: int, budget: int) -> List[int]:
    """Frame indices (into a 1 fps or denser decode of n_frames) that ``budget`` keeps:
    all of them if they fit max_frames, else max_frames evenly spaced ones."""
    m = max_frames_for(budget)
    if n_frames <= m:
        return list(range(n_frames))
    import numpy as np
    return np.linspace(0, n_frames - 1, m, dtype=int).tolist()


def frame_size(h: int, w: int, tokens: int, factor: int = FACTOR, min_tokens: int = MIN_TOKENS) -> Tuple[int, int]:
    """(h, w) a frame is resized to under a per-frame cap of ``tokens`` merged tokens --
    the same arithmetic as the processor's ``simple_batched_resize``, so frames
    pre-shrunk to this size pass through the processor unchanged."""
    max_pixels = tokens * factor * factor
    min_pixels = min_tokens * factor * factor
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return h_bar, w_bar


def frame_tokens(h: int, w: int, budget: int, n_frames: int) -> int:
    """Per-frame token cap for ``n_frames`` frames of an (h, w) video (rule 2)."""
    nh, nw = frame_size(h, w, MAX_FRAME_TOKENS)
    native = (nh // FACTOR) * (nw // FACTOR)
    return min(native, max(MIN_FRAME_TOKENS, budget // max(1, n_frames)))


@dataclass
class Plan:
    mode: str                         # "raw" | "hybrid" | "forced8" | "all_compressed" (debug)
    seg_lens: List[int]               # frames per group, sums to T (raw mode: all 1)
    compressed: Set[int] = field(default_factory=set)   # indices into seg_lens
    vision_tokens: int = 0

    @property
    def n_raw_frames(self) -> int:
        return sum(L for j, L in enumerate(self.seg_lens) if j not in self.compressed)


def plan_groups(per_frame: torch.Tensor, hw: int, budget: int, force_mode: str = "",
                raw_hw: Optional[int] = None) -> Plan:
    """Rules 2-4 given the per-frame mean encoder features (T, C) at the chosen size.
    ``raw_hw``: LLM rows of one raw frame when raw frames are sent at a lower resolution
    than the encoder features the groups are cut from (Phase-3 raw frames, <= 256 rows);
    defaults to ``hw``."""
    T = int(per_frame.shape[0])
    hw = hw if raw_hw is None else int(raw_hw)
    if force_mode == "all_compressed":  # Phase-1 training layout, for equivalence tests
        lens = adaptive_segment_lengths(per_frame, TARGET_FRAMES, FORCE_EVERY, 0.0, None,
                                        n_segments=adaptive_segment_count(T, TARGET_FRAMES, FORCE_EVERY))
        return Plan("all_compressed", lens, set(range(len(lens))), len(lens) * K)
    if force_mode == "raw" or T * hw <= budget:  # "raw": original VideoLLaMA3 (resize-only) protocol
        return Plan("raw", [1] * T, set(), T * hw)

    lens = adaptive_segment_lengths(per_frame, TARGET_FRAMES, FORCE_EVERY, 0.0, None,
                                    n_segments=adaptive_segment_count(T, TARGET_FRAMES, FORCE_EVERY))
    if len(lens) * K <= budget:
        f = torch.nn.functional.normalize(per_frame.float(), dim=-1)
        diff = (1.0 - (f[1:] * f[:-1]).sum(-1)).tolist()     # diff[j]: frame j -> j+1
        var, s = [], 0
        for L in lens:
            var.append(sum(diff[s:s + L - 1]) / (L - 1) if L > 1 else math.inf)
            s += L
        # lowest variation first; ties -> the bigger saving first
        order = sorted(range(len(lens)), key=lambda j: (var[j], -lens[j], j))
        cost, comp = T * hw, set()
        for j in order:
            if cost <= budget:
                break
            comp.add(j)
            cost -= lens[j] * hw - K
        assert cost <= budget, (cost, budget)
        return Plan("hybrid", lens, comp, cost)

    lens = [FORCE_EVERY] * (T // FORCE_EVERY) + ([T % FORCE_EVERY] if T % FORCE_EVERY else [])
    assert len(lens) * K <= budget, f"{T} frames do not fit {budget} even as 8-frame groups"
    return Plan("forced8", lens, set(range(len(lens))), len(lens) * K)
