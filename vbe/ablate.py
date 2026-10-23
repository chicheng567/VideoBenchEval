"""Layout ablations of one checkpoint, every variant in one pass over the videos.

Each video is decoded and encoded once, exactly as vbe/infer.py does (same frames, same
encoder rows); the variants only change which frames are kept, how they are grouped,
and what a compressed group becomes:

  coverage (long videos)   a video with more 1 fps frames than the variant's cap keeps
                           ``cap`` evenly spaced frames; every other video keeps the
                           baseline layout
      rho<R>    cap B // R, laid out by the baseline's hybrid rule: fewer frames, more
                of them raw. rho16 caps at the most frames whose adaptive tf-4 groups
                all fit (4 * B/64 - 1), i.e. pure qbase at 16 tokens/frame. rho32-128
                then fill the budget: the greedy rule can overshoot by one 8-frame group
                (up to ~30% of B unused at 4096), so compressed groups go back to raw,
                most-changing first, while they still fit.
      rho32nf   rho32 without that fill step (does the fill matter?)
      qtf<k>    cap k * B/64 - 1 (k = 1: B/64), every frame compressed in adaptive
                tf-k groups: lighter compression (64 / k tokens per frame), no raw
      low64     cap B // 64, every frame raw at <= 64 rows (re-encoded)
  compressor (baseline plan, every compressed group swapped for ONE raw frame of
  <= 64 rows, the qbase's own K, in the raw-frame text format)
      sel_resize   the group's middle frame re-encoded at the small grid
      sel_merge    the group's middle frame's encoder rows area-pooled to that grid
      mean_merge   the group's mean encoder rows area-pooled to that grid
  grouping (uniform cuts, same count N = T//tf + 1, instead of the adaptive segmenter)
      uni8      baseline coverage
      uni16     rho16 coverage
  compression order
      hivar     the hybrid rule compresses the MOST-changing groups first (the baseline:
                the least-changing), change measured as Phase-3 training ranks raw
                segments (compressor._hybrid_raw_segments: a 1-frame group's change is
                its step from the previous frame); training sent the least-changing
                segments raw, so this is the order that matches it

A (question, budget, variant) whose LLM input is identical to the baseline's (same
frames, plan and raw grid, qbase rows) is not re-run: the baseline record is copied with
``reused: true``. Outputs: <out_root>/<variant>_b<B>/<benchmark>/shard*.jsonl, scored with
``python -m vbe.score``.
"""
import argparse
import glob
import json
import math
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from videollama3.constants import DEFAULT_IMAGE_TOKEN
from videollama3.model.compressor import adaptive_segment_count, adaptive_segment_lengths

from vbe import budget as bd
from vbe.benchmarks import DEFAULT_ROOT, build_prompt, extract_choice, load_benchmark
from vbe.infer import BudgetedVideoLLM, Videos

SMALL = 64  # rows of a selected / merged frame: the qbase's K

BASE = dict(cap=None, layout="hybrid", tf=4, seg="adaptive", comp="qbase", raw_tokens=None)
VARIANTS = {
    "rho16": dict(BASE, cap=("tf", 4)),
    "rho32": dict(BASE, cap=("rho", 32), fill=True),
    "rho32nf": dict(BASE, cap=("rho", 32)),
    "rho64": dict(BASE, cap=("rho", 64), fill=True),
    "rho128": dict(BASE, cap=("rho", 128), fill=True),
    "rho256": dict(BASE, cap=("rho", 256)),
    "qtf2": dict(BASE, cap=("tf", 2), layout="qbase", tf=2),
    "qtf1": dict(BASE, cap=("tf", 1), layout="qbase", tf=1),
    "low64": dict(BASE, cap=("rho", 64), raw_tokens=SMALL),
    "sel_resize": dict(BASE, comp="sel_resize"),
    "sel_merge": dict(BASE, comp="sel_merge"),
    "mean_merge": dict(BASE, comp="mean_merge"),
    "uni8": dict(BASE, seg="uniform"),
    "uni16": dict(BASE, cap=("tf", 4), seg="uniform"),
    "hivar": dict(BASE, order="hivar"),
    "base": BASE,  # the baseline itself (equivalence check, with --no_reuse)
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--raw_max_tokens", type=int, default=256, help="baseline raw-frame cap (Phase-3 protocol)")
    p.add_argument("--benchmark", required=True, choices=["videomme", "mlvu", "longvideobench"])
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--out_root", required=True, help="results go to <out_root>/<variant>_b<B>/<benchmark>/")
    p.add_argument("--baseline_run", required=True,
                   help="results/<run> prefix of the baseline (<run>_b<B>/<benchmark>/*.jsonl), for reuse")
    p.add_argument("--jobs", required=True,
                   help="variant:budget[:budget...] separated by commas, e.g. rho32:4096:16384,sel_merge:1024")
    p.add_argument("--no_reuse", action="store_true", help="run every job, even when identical to the baseline")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--encoder_chunk_frames", type=int, default=128)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--limit", type=int, default=0, help="only the first N videos of this shard (debug)")
    return p.parse_args()


# ----------------------------------------------------------------------------- planning

def frame_cap(v, budget):
    """Most frames a long video keeps under variant ``v`` (None: the baseline's B/8)."""
    if v["cap"] is None:
        return None
    kind, x = v["cap"]
    m = budget // bd.K
    if kind == "rho":
        return budget // x
    return x * m - 1 if x > 1 else m  # most T with adaptive_segment_count(T, x) <= B/K


def frame_indices(n_decoded, budget, v):
    """(indices into the decoded frames, capped?) for variant ``v`` at ``budget``."""
    cap = frame_cap(v, budget)
    if cap is None or n_decoded <= cap:
        return bd.subsample(n_decoded, budget), False
    return np.linspace(0, n_decoded - 1, cap, dtype=int).tolist(), True


def segment(per_frame, tf, seg):
    T = int(per_frame.shape[0])
    n = adaptive_segment_count(T, tf, bd.FORCE_EVERY)
    if seg == "adaptive":
        return adaptive_segment_lengths(per_frame, tf, bd.FORCE_EVERY, 0.0, None, n_segments=n)
    bounds = np.round(np.linspace(0, T, n + 1)).astype(int)
    return [int(x) for x in np.diff(bounds)]


def group_variation(per_frame, lens):
    """bd.plan_groups' per-group variation (1-frame groups: +inf)."""
    f = F.normalize(per_frame.float(), dim=-1)
    diff = (1.0 - (f[1:] * f[:-1]).sum(-1)).tolist()
    var, s = [], 0
    for L in lens:
        var.append(sum(diff[s:s + L - 1]) / (L - 1) if L > 1 else math.inf)
        s += L
    return var


def training_change(per_frame, lens):
    """Per-group change as Phase-3 training ranks raw segments
    (compressor.TransformerDecoderFlatCompressor._hybrid_raw_segments)."""
    f = F.normalize(per_frame.float(), dim=-1)
    diff = (1.0 - (f[1:] * f[:-1]).sum(-1)).tolist()
    change, s = [], 0
    for L in lens:
        change.append(sum(diff[s:s + L - 1]) / (L - 1) if L >= 2 else (diff[s - 1] if s >= 1 else diff[0]))
        s += L
    return change


def fill(plan, per_frame, raw_hw, budget):
    """Spend what the greedy rule left of the budget on raw frames, most-changing
    compressed group first: a whole group goes back raw when it fits, else its first k
    frames do (k raw frames + a compressed group of the rest)."""
    if plan.mode != "hybrid":
        return plan
    var = group_variation(per_frame, plan.seg_lens)
    raw_k = {}                                   # compressed group -> frames sent raw
    cost = plan.vision_tokens
    for j in sorted(plan.compressed, key=lambda j: (-var[j], plan.seg_lens[j], j)):
        L = plan.seg_lens[j]
        if cost + L * raw_hw - bd.K <= budget:
            raw_k[j] = L
            cost += L * raw_hw - bd.K
        elif L > 1 and (budget - cost) // raw_hw >= 1:
            raw_k[j] = min(L - 1, (budget - cost) // raw_hw)
            cost += raw_k[j] * raw_hw
    lens, comp = [], set()
    for j, L in enumerate(plan.seg_lens):
        k = raw_k.get(j, 0) if j in plan.compressed else L
        if k:
            lens.append(k)
        if L - k:
            comp.add(len(lens))
            lens.append(L - k)
    assert cost <= budget and sum(lens) == sum(plan.seg_lens)
    return bd.Plan("hybrid", lens, comp, cost)


def plan_variant(per_frame, hw, raw_hw, budget, v):
    """bd.plan_groups with the variant's layout / cut; identical to it for the baseline."""
    plan = _plan_variant(per_frame, hw, raw_hw, budget, v)
    return fill(plan, per_frame, raw_hw, budget) if v.get("fill") else plan


def _plan_variant(per_frame, hw, raw_hw, budget, v):
    T = int(per_frame.shape[0])
    if v["layout"] == "qbase":
        lens = segment(per_frame, v["tf"], v["seg"])
        assert len(lens) * bd.K <= budget, (len(lens), budget)
        return bd.Plan("qbase", lens, set(range(len(lens))), len(lens) * bd.K)
    if v["seg"] == "adaptive" and v.get("order") is None:
        return bd.plan_groups(per_frame, hw, budget, raw_hw=raw_hw)
    # the hybrid rule of bd.plan_groups (rules 2-4) on the variant's cut / order
    if T * raw_hw <= budget:
        return bd.Plan("raw", [1] * T, set(), T * raw_hw)
    lens = segment(per_frame, v["tf"], v["seg"])
    if len(lens) * bd.K > budget:
        return bd.plan_groups(per_frame, hw, budget, raw_hw=raw_hw)   # forced8: already uniform
    if v.get("order") == "hivar":
        change = training_change(per_frame, lens)
        order = sorted(range(len(lens)), key=lambda j: (-change[j], -lens[j], j))
    else:
        var = group_variation(per_frame, lens)
        order = sorted(range(len(lens)), key=lambda j: (var[j], -lens[j], j))
    cost, comp = T * raw_hw, set()
    for j in order:
        if cost <= budget:
            break
        comp.add(j)
        cost -= lens[j] * raw_hw - bd.K
    assert cost <= budget, (cost, budget)
    return bd.Plan("hybrid", lens, comp, cost)


# ----------------------------------------------------------------------------- LLM input

@torch.inference_mode()
def build(llm, frames, ts, tok, enc, frame_idx, plan, raw_grid, comp):
    """vbe.infer.BudgetedVideoLLM.build_video for a given plan; ``comp`` says what a
    compressed group becomes (qbase: K rows in compression tokens, else one small raw
    frame). Bit-identical to build_video for the baseline (checked by --no_reuse runs)."""
    T = len(ts)
    feats, (hh, ww) = enc
    hw = hh * ww
    raw_hw = raw_grid[0] * raw_grid[1]
    C = feats.shape[-1]
    starts = np.cumsum([0] + plan.seg_lens[:-1]).tolist()

    group_rows = {}                                   # compressed group j -> its LLM rows
    small = llm.hybrid_raw.raw_frame_grid(hh, ww, SMALL)
    if plan.compressed and comp == "qbase":
        qb = llm.inner.token_compressor
        cu = torch.tensor([0] + list(np.cumsum([L * hw for L in plan.seg_lens])), device=feats.device)
        saved = qb.adaptive_segmentation
        qb.adaptive_segmentation = False
        try:
            rows = qb(feats, cu, [(hh, ww)] * len(plan.seg_lens))
        finally:
            qb.adaptive_segmentation = saved
        rows = llm.inner.mm_projector(rows)
        group_rows = {j: rows[j * bd.K:(j + 1) * bd.K] for j in plan.compressed}
    elif plan.compressed:
        comp_js = sorted(plan.compressed)
        mid = [starts[j] + (plan.seg_lens[j] - 1) // 2 for j in comp_js]
        if comp == "sel_resize":
            src = [frame_idx[f] for f in mid] if frame_idx is not None else mid
            x = llm.reencode(frames[src], tok, small)
        else:
            grid = feats.view(T, hh, ww, C)
            if comp == "sel_merge":
                x = grid[torch.tensor(mid, device=feats.device)].float()
            else:  # mean_merge
                x = torch.stack([grid[starts[j]:starts[j] + plan.seg_lens[j]].float().mean(0) for j in comp_js])
            x = F.adaptive_avg_pool2d(x.permute(0, 3, 1, 2), small).permute(0, 2, 3, 1)
            x = x.reshape(-1, C).to(feats.dtype)
        x = llm.proj_orig(x)
        n = small[0] * small[1]
        group_rows = {j: x[i * n:(i + 1) * n] for i, j in enumerate(comp_js)}

    raw_idx = [f for j, (s0, L) in enumerate(zip(starts, plan.seg_lens)) if j not in plan.compressed
               for f in range(s0, s0 + L)]
    raw_rows = None
    if raw_idx:
        if raw_grid == (hh, ww):
            sel = feats.view(T, hw, -1)[torch.tensor(raw_idx, device=feats.device)].reshape(-1, C)
        else:
            src = raw_idx if frame_idx is None else [frame_idx[f] for f in raw_idx]
            sel = llm.reencode(frames[src], tok, raw_grid)
        raw_rows = llm.proj_orig(sel)

    sec = [int(round(t)) for t in ts]
    rows, counts, text, r = [], [], [], 0
    n_blocks = sum(1 if j in plan.compressed else L for j, L in enumerate(plan.seg_lens))
    b = 0
    for j, (s, L) in enumerate(zip(starts, plan.seg_lens)):
        if j in plan.compressed:
            b += 1
            end = "\n" if b == n_blocks else ""
            if comp == "qbase":
                text.append(f"Time:{sec[s]}s-{sec[s + L - 1]}s:{llm.cs}{DEFAULT_IMAGE_TOKEN}{llm.ce}" + end)
            else:
                m = s + (L - 1) // 2
                text.append(f"Time {round(ts[m], 1)}s:{DEFAULT_IMAGE_TOKEN}" + (end or ","))
            rows.append(group_rows[j]); counts.append(group_rows[j].shape[0])
        else:
            for f in range(s, s + L):
                b += 1
                text.append(f"Time {round(ts[f], 1)}s:{DEFAULT_IMAGE_TOKEN}" + ("\n" if b == n_blocks else ","))
                rows.append(raw_rows[r * raw_hw:(r + 1) * raw_hw]); counts.append(raw_hw)
                r += 1
    stats = dict(mode=plan.mode, hw=hw, raw_hw=raw_hw, n_groups=len(plan.seg_lens),
                 n_compressed=len(plan.compressed), n_raw_frames=plan.n_raw_frames, vision_tokens=sum(counts))
    return dict(rows=torch.cat(rows), counts=counts, text="".join(text), stats=stats)


# ----------------------------------------------------------------------------- main

def parse_jobs(s):
    jobs = []
    for item in s.split(","):
        name, *budgets = item.split(":")
        assert name in VARIANTS, f"unknown variant {name}"
        jobs += [(name, int(b)) for b in budgets]
    return jobs


def load_baseline(prefix, bench, budgets):
    out = {}
    for b in budgets:
        recs = {}
        for f in sorted(glob.glob(f"{prefix}_b{b}/{bench}/*.jsonl")):
            for line in open(f):
                if line.strip():
                    r = json.loads(line)
                    recs[r["qid"]] = r
        out[b] = recs
    return out


def main():
    args = parse_args()
    jobs = parse_jobs(args.jobs)
    budgets = sorted({b for _, b in jobs})
    outputs = {(v, b): os.path.join(args.out_root, f"{v}_b{b}", args.benchmark,
                                    f"shard{args.shard_id}of{args.num_shards}.jsonl") for v, b in jobs}
    base_recs = load_baseline(args.baseline_run, args.benchmark, budgets)
    samples = load_benchmark(args.benchmark, args.data_root)
    done = {}
    for job, path in outputs.items():
        done[job] = {json.loads(l)["qid"] for l in open(path) if l.strip()} if os.path.exists(path) else set()
    groups = OrderedDict()
    for s in samples:
        groups.setdefault(s["video"], []).append(s)
    groups = sorted(groups.items())[args.shard_id::args.num_shards]
    if args.limit:
        groups = groups[: args.limit]
    todo = [(vid, {job: [s for s in ss if s["qid"] not in done[job]] for job in jobs}) for vid, ss in groups]
    todo = [(vid, {job: q for job, q in per.items() if q}) for vid, per in todo]
    todo = [(vid, per) for vid, per in todo if per]
    n_q = sum(len(q) for _, per in todo for q in per.values())
    print(f"[ablate {args.benchmark} shard {args.shard_id}/{args.num_shards}] {len(jobs)} jobs: {len(todo)} videos / "
          f"{n_q} (question, job) pairs to go", flush=True)
    if not todo:
        return

    llm = BudgetedVideoLLM(args.model_path, "", budgets[-1], args.encoder_chunk_frames,
                           raw_mode="phase3", raw_max_tokens=args.raw_max_tokens)
    # the baseline's decode: frames for the largest budget (B/8 of it); every job takes a subset
    loader = DataLoader(Videos(todo, 16384), batch_size=None, shuffle=False,
                        num_workers=args.num_workers, prefetch_factor=1 if args.num_workers else None)
    fouts = {}
    for job, path in outputs.items():
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fouts[job] = open(path, "a")
    t0, n_done, n_reused = time.time(), 0, 0
    tm = dict(wait=0.0, encode=0.0, build=0.0, answer=0.0)
    t_last = time.time()
    for item in loader:
        tm["wait"] += time.time() - t_last
        per = item["samples"]
        frames, ts, info, err = item.get("frames"), item.get("ts"), item.get("info"), item["error"]
        enc_cache = {}

        def features(idx, b):
            tok = bd.frame_tokens(*info["native_hw"], b, len(idx))
            if tok not in enc_cache:
                t1 = time.time()
                enc_cache[tok] = llm.encode(frames, tok)
                torch.cuda.synchronize()
                tm["encode"] += time.time() - t1
            feats, grid = enc_cache[tok]
            hw = grid[0] * grid[1]
            sel = torch.tensor(idx, device=feats.device)
            return tok, (feats.view(len(ts), hw, -1)[sel].reshape(-1, feats.shape[-1]), grid)

        for job in sorted(per, key=lambda j: (j[1], j[0])):
            vname, b = job
            v = VARIANTS[vname]
            qs = per[job]
            video, err_b, binfo, reuse = None, err, None, False
            if err is None:
                try:
                    idx, capped = frame_indices(len(ts), b, v)
                    eff = v if (capped or v["cap"] is None) else BASE   # short videos keep the baseline layout
                    reuse = eff is BASE and not args.no_reuse
                    if not reuse:
                        tok, enc = features(idx, b)
                        t1 = time.time()
                        grid = enc[1]
                        raw_grid = llm.hybrid_raw.raw_frame_grid(*grid, eff["raw_tokens"] or args.raw_max_tokens)
                        base_grid = llm.hybrid_raw.raw_frame_grid(*grid, args.raw_max_tokens)
                        per_frame = enc[0].view(len(idx), grid[0] * grid[1], -1).float().mean(1)
                        hw = grid[0] * grid[1]
                        plan = plan_variant(per_frame, hw, raw_grid[0] * raw_grid[1], b, eff)
                        if not args.no_reuse and not capped and raw_grid == base_grid:
                            bplan = bd.plan_groups(per_frame, hw, b, raw_hw=base_grid[0] * base_grid[1])
                            reuse = (plan.seg_lens == bplan.seg_lens and plan.compressed == bplan.compressed
                                     and (eff["comp"] == "qbase" or not plan.compressed))
                        if not reuse:
                            video = build(llm, frames, [ts[k] for k in idx], tok, enc, idx, plan, raw_grid,
                                          eff["comp"])
                            torch.cuda.synchronize()
                            fps = min(bd.FPS, len(idx) / max(info["duration"], 1e-6))
                            binfo = dict(info, n_frames=len(idx), fps=round(fps, 4),
                                         fps_ok=fps >= bd.MIN_FPS - 1e-6, capped=capped)
                        tm["build"] += time.time() - t1
                except torch.cuda.OutOfMemoryError as e:
                    err_b = repr(e)
                    torch.cuda.empty_cache()
            if reuse and all(s["qid"] in base_recs[b] for s in qs):
                for s in qs:
                    rec = dict(base_recs[b][s["qid"]], variant=vname, reused=True)
                    fouts[job].write(json.dumps(rec, ensure_ascii=False) + "\n")
                fouts[job].flush()
                n_done += len(qs); n_reused += len(qs)
                continue
            assert not reuse, f"baseline records missing for {item['video']} at {b}"
            answers = []
            if err_b is None:
                try:
                    t1 = time.time()
                    answers = llm.answer_many(video, [build_prompt(args.benchmark, s) for s in qs],
                                              args.max_new_tokens)
                    tm["answer"] += time.time() - t1
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    err_b = repr(e)
            for k, s in enumerate(qs):
                rec = dict(qid=s["qid"], video=os.path.basename(s["video"]), answer=s["answer"], meta=s["meta"],
                           variant=vname, reused=False)
                if err_b is not None:
                    rec.update(pred="", response="", error=err_b)
                else:
                    resp, seq_len = answers[k]
                    rec.update(pred=extract_choice(args.benchmark, resp, s["options"]), response=resp,
                               seq_len=seq_len, video_info=binfo, **video["stats"])
                fouts[job].write(json.dumps(rec, ensure_ascii=False) + "\n")
            fouts[job].flush()
            n_done += len(qs)
            del video
        del enc_cache
        torch.cuda.empty_cache()
        el = time.time() - t0
        print(f"  {n_done}/{n_q} pairs ({n_reused} reused), {el / 60:.1f} min, "
              f"eta {(n_q - n_done) * el / max(n_done, 1) / 60:.0f} min | "
              + " ".join(f"{k} {v / el:.0%}" for k, v in tm.items()), flush=True)
        t_last = time.time()
    for f in fouts.values():
        f.close()
    print(f"[ablate {args.benchmark} shard {args.shard_id}] done: {n_done} pairs ({n_reused} reused) "
          f"in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
