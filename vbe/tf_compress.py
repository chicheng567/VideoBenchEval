"""Training-free video-token compression baselines on top of the original VideoLLaMA3.

Each method takes the LLM-space rows of every frame (original VL3 projector output,
(T, hw, D)) plus the per-frame vision-encoder statistics it needs, and returns the rows
each frame keeps (a list of T tensors, possibly empty). Ported from the official code:

* FastVID (NeurIPS'25, github.com/LunarShen/FastVID, Qwen2.5-VL port
  ``fastvid_qwen25vl/.../modeling_qwen2_5_vl.py``): DySeg + STPrune (salient tokens by
  the last encoder layer's frame-mean-query attention, context tokens by density peaks)
  + DTM merging. Defaults are the official eval.sh values; ``retention`` replaces
  ``fastvid_retention_ratio``.
* VisionZip (CVPR'25, github.com/dvlab-research/VisionZip, Qwen2.5-VL port
  ``Qwen2_5_VL/qwen2_5vl_visionzip.py``): dominant tokens = most attended in the last
  encoder layer, contextual tokens = uniformly spaced targets that absorb the rest by key
  similarity. Applied per frame (the encoder attends within a frame); the per-frame
  dominant : contextual split is the paper's 54 : 10.
* DyCoke (CVPR'25, github.com/KD-TAO/DyCoke, ``dycole_ttm`` in ``llava/model/llava_arch.py``):
  temporal token merging (TTM) over 4-frame windows, k = 0.7 (README). Its second stage
  prunes the KV cache during decoding, which does nothing for 1-token MCQ answers and is
  left out.

Encoder statistics come from ``LastLayerStats``, which wraps the last encoder layer's
attention and records, per frame and per merged (2x2) token, what FastVID and VisionZip
read from Qwen2.5-VL's last vision block: the patch order inside VL3's encoder is the
same merge-unit-major order (h/2, w/2, 2, 2), so every 4 consecutive patches are one
merged token, as in Qwen2.5-VL.
"""
import math
from typing import List

import torch
import torch.nn.functional as F

# ignore: the Qwen2.5-VL port's extra step (0.95) folds runs of near-identical frames into one
# frame capped at frm_len, which on 1-fps per-frame encoders throws most of the budget away on static
# videos; the LLaVA-Video / LLaVA-OneVision ports (per-frame encoders, like VL3) have no such step,
# and ignore > 1 reproduces them.
FASTVID = dict(c=8, tau=0.84, ignore=1.01, d=0.4, p=4, beta=0.6)
VISIONZIP_DOMINANT, VISIONZIP_CONTEXTUAL = 54, 10
DYCOKE_K = 0.7
DYCOKE_KEEP = (1 + 3 * (1 - DYCOKE_K)) / 4  # fraction of tokens TTM keeps (0.475)


class LastLayerStats:
    """Records, for every frame passing through the encoder's last layer:
    ``fastvid``  (hw,)    softmax(mean-query . merged-key) averaged over heads
    ``vzip``     (hw,)    attention each merged token receives (sum over queries, head-mean)
    ``vzip_key`` (hw, d)  head-mean key of each merged token
    Frames are appended in encoder order; ``reset()`` before each video."""

    def __init__(self, encoder):
        """encoder: the Videollama3VisionEncoderModel (``get_vision_encoder().vision_encoder``)."""
        self.attn = encoder.encoder.layers[-1].self_attn
        self.enabled = True
        self._forward = self.attn.forward
        self.attn.forward = self._wrapped
        self.reset()

    def reset(self):
        self.fastvid, self.vzip, self.vzip_key = [], [], []

    def _wrapped(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
        from videollama3.model.videollama3_encoder.modeling_videollama3_encoder import \
            apply_rotary_pos_emb_vision_fused
        a = self.attn
        if not self.enabled:
            return self._forward(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)
        with torch.no_grad():
            L = hidden_states.shape[0]
            q = a.q_proj(hidden_states).view(L, a.num_heads, a.head_dim)
            k = a.k_proj(hidden_states).view(L, a.num_heads, a.head_dim)
            q = apply_rotary_pos_emb_vision_fused(q, rotary_pos_emb)
            k = apply_rotary_pos_emb_vision_fused(k, rotary_pos_emb)
            bounds = cu_seqlens.tolist()
            for s, e in zip(bounds[:-1], bounds[1:]):
                qf = q[s:e].transpose(0, 1).float()            # (H, P, d)
                kf = k[s:e].transpose(0, 1).float()
                H, P, d = qf.shape
                km = kf.view(H, P // 4, 4, d).mean(2)          # merged-token keys
                # FastVID: one frame-mean query against the merged keys
                w = torch.softmax(qf.mean(1, keepdim=True) @ km.transpose(1, 2) / math.sqrt(d), -1)
                self.fastvid.append(w.mean(1).mean(0))         # (P/4,)
                # VisionZip: full softmax attention, head-mean, summed over queries, merged-token mean
                att = torch.softmax(qf @ kf.transpose(1, 2) / math.sqrt(d), -1).mean(0).sum(0)
                self.vzip.append(att.view(P // 4, 4).mean(-1))
                self.vzip_key.append(km.mean(0))
        return self._forward(hidden_states, cu_seqlens=cu_seqlens, rotary_pos_emb=rotary_pos_emb)


# ----------------------------------------------------------------------------- FastVID

def fastvid(rows: torch.Tensor, attn: torch.Tensor, retention: float, c=FASTVID["c"], tau=FASTVID["tau"],
            ignore=FASTVID["ignore"], d=FASTVID["d"], p=FASTVID["p"], beta=FASTVID["beta"]) -> List[torch.Tensor]:
    """rows (T, N, D) LLM-space tokens, attn (T, N) frame-query attention. Line-by-line the
    official per-segment loop; returns each frame's kept rows (salient + merged context, in
    token order)."""
    T, frm_len, _ = rows.shape
    device = rows.device
    g = rows.float().mean(1)
    g = g / g.norm(dim=1, keepdim=True)
    sim = (g[:-1] * g[1:]).sum(1)
    if T > 1:
        cut = torch.topk(sim, min(c - 1, T - 1), largest=False).indices
        cut = torch.unique(torch.cat([cut, torch.nonzero(sim < tau).squeeze(1)])).sort().values
        sizes = [cut[0].item() + 1] + [cut[i].item() - cut[i - 1].item() for i in range(1, len(cut))] \
            + [T - cut[-1].item() - 1]
        sizes = [s for s in sizes if s > 0]
    else:
        sizes = [1]
    frame_retain = int(frm_len * retention)
    out = [rows.new_zeros(0, rows.shape[-1]) for _ in range(T)]
    f0 = 0
    for seg_len in sizes:
        hs, aw, gf = rows[f0:f0 + seg_len], attn[f0:f0 + seg_len], g[f0:f0 + seg_len]
        seg_retain = frame_retain * seg_len
        seg_context = max(int(seg_retain * d), 1)
        seg_salient = seg_retain - seg_context
        if seg_len == 1:
            frm_sal, frm_ctx = [seg_salient], [seg_context]
        else:
            s2 = (gf[:-1] * gf[1:]).sum(1)
            cc = torch.nonzero(s2 < ignore).squeeze(1).sort().values
            if cc.shape[0] > 0:
                csz = [cc[0].item() + 1] + [cc[i].item() - cc[i - 1].item() for i in range(1, cc.shape[0])] \
                    + [seg_len - cc[-1].item() - 1]
            else:
                csz = [seg_len]
            vlen = len(csz)
            ch, rem = divmod(seg_salient, vlen)
            sal = [ch + (1 if i < rem else 0) for i in range(vlen)]
            tnum = (vlen + p - 1) // p
            ch, rem = divmod(seg_context, tnum)
            tctx = [ch + (1 if i < rem else 0) for i in range(tnum)]
            ti, ctx = 0, []
            for i in range(vlen):
                if i % p == 0:
                    ctx.append(min(tctx[ti], frm_len // 2))
                    if tctx[ti] + ctx[-1] > frm_len:
                        tctx[ti] = frm_len - ctx[-1]
                    ti += 1
                else:
                    ctx.append(0)
            ctx.reverse()
            frm_sal, frm_ctx = [], []
            for i in range(vlen):
                frm_sal.extend([0] * (csz[i] - 1))
                frm_ctx.extend([0] * (csz[i] - 1))
                if ctx[i] == 0:
                    frm_sal.append(min(sal[i], frm_len)); frm_ctx.append(0)
                elif ctx[i] > 0:
                    frm_ctx.append(min(ctx[i], frm_len // 2))
                    frm_sal.append(min(sal[i], frm_len - frm_ctx[-1]))
                else:
                    frm_sal.append(frm_len - int(frm_len * d)); frm_ctx.append(int(frm_len * d))
        sal_idx, ctx_idx = [], []
        for fi in range(seg_len):
            top = torch.topk(aw[fi], frm_sal[fi]).indices if frm_sal[fi] > 0 else None
            if frm_ctx[fi] > 0:
                allf = torch.arange(frm_len, device=device)
                remain = allf[~torch.isin(allf, top)] if top is not None else allf
                x = hs[fi][remain].unsqueeze(0).float()
                D = x.shape[-1]
                dist = torch.cdist(x, x) / (D ** 0.5)
                dn, _ = torch.topk(dist, k=min(4, x.shape[1]), dim=-1, largest=False)
                dens = (-(dn ** 2).mean(-1)).exp()
                dens = dens + torch.rand(dens.shape, device=device, dtype=dens.dtype) * 1e-6
                mask = (dens[:, None, :] > dens[:, :, None]).to(x.dtype)
                dmax = dist.flatten(1).max(-1)[0][:, None, None]
                dd, _ = (dist * mask + dmax * (1 - mask)).min(-1)
                _, si = torch.topk(dd * dens, k=frm_ctx[fi], dim=-1)
                cidx = remain[si[0].sort().values]
            else:
                cidx = None
            if top is not None:
                sal_idx.append(top + fi * frm_len)
            if cidx is not None:
                ctx_idx.append(cidx + fi * frm_len)
        flat = hs.reshape(seg_len * frm_len, -1)
        sal_idx = torch.cat(sal_idx) if sal_idx else torch.zeros(0, dtype=torch.long, device=device)
        all_idx = torch.arange(seg_len * frm_len, device=device)
        norm = flat / flat.norm(dim=-1, keepdim=True)
        ctx_rows = []
        for ci in ctx_idx:
            merge = all_idx[~torch.isin(all_idx, torch.cat([sal_idx, ci]))]
            simm = norm[merge] @ norm[ci].T
            one = torch.zeros(merge.shape[0], ci.shape[0], dtype=flat.dtype, device=device)
            one.scatter_(1, simm.argmax(1).unsqueeze(-1), 1)
            avg = (1 / (one.sum(0).unsqueeze(-1) + 1)).clamp(min=beta)
            cnt = one.sum(0).clamp(min=1).unsqueeze(-1)
            agg = (one.T @ flat[merge]) / cnt
            ctx_rows.append(avg * flat[ci] + (1 - avg) * agg)
        idx = torch.cat([sal_idx] + ctx_idx)
        hid = torch.cat([flat[sal_idx]] + ctx_rows, 0)
        order = torch.argsort(idx)
        idx, hid = idx[order], hid[order]
        fr = torch.div(idx, frm_len, rounding_mode="floor")
        for fi in range(seg_len):
            out[f0 + fi] = hid[fr == fi]
        f0 += seg_len
    return out


# ----------------------------------------------------------------------------- VisionZip

def visionzip(rows: torch.Tensor, attn: torch.Tensor, keys: torch.Tensor, per_frame: int) -> List[torch.Tensor]:
    """Per frame: ``dominant`` most-attended tokens + ``contextual`` uniformly spaced targets
    that sum in (official ``target + aggregated``) the remaining tokens they are most similar
    to by key. rows (T, N, D), attn (T, N), keys (T, N, d)."""
    T, N, _ = rows.shape
    k = min(per_frame, N)
    ctx_n = max(1, round(k * VISIONZIP_CONTEXTUAL / (VISIONZIP_DOMINANT + VISIONZIP_CONTEXTUAL))) if k > 1 else 0
    dom_n = k - ctx_n
    out = []
    for f in range(T):
        top = torch.topk(attn[f], dom_n).indices
        keep = torch.zeros(N, dtype=torch.bool, device=rows.device)
        keep[top] = True
        rest = (~keep).nonzero(as_tuple=True)[0]
        if ctx_n == 0 or rest.numel() == 0:
            out.append(rows[f][keep]); continue
        m = keys[f][rest]
        m = m / m.norm(dim=-1, keepdim=True)
        step = max(1, m.shape[0] // ctx_n)
        tgt = torch.arange(0, m.shape[0], step, device=rows.device)[:ctx_n]
        others = ~torch.isin(torch.arange(m.shape[0], device=rows.device), tgt)
        simm = m[others] @ m[tgt].T
        one = torch.zeros(int(others.sum()), tgt.shape[0], dtype=rows.dtype, device=rows.device)
        one.scatter_(1, simm.argmax(1).unsqueeze(-1), 1)
        cnt = one.sum(0).clamp(min=1).unsqueeze(-1)
        hr = rows[f][rest]
        ctx = hr[tgt] + (one.T @ hr[others]) / cnt
        # official order: dominant tokens and contextual targets stay at their positions
        pos = torch.cat([top, rest[tgt]])
        val = torch.cat([rows[f][top], ctx])
        o = torch.argsort(pos)
        out.append(val[o])
    return out


# ----------------------------------------------------------------------------- DyCoke TTM

def dycoke_ttm(rows: torch.Tensor, k: float = DYCOKE_K) -> List[torch.Tensor]:
    """``dycole_ttm`` with frames kept apart: in every pair (i, i+1) frame i+1 keeps its
    int((1-k) N) tokens least similar to frame i; in every 4-window, frame i+2 is re-pruned
    against frame i and frame i is restored to full."""
    T, N, _ = rows.shape
    keep_n = int((1 - k) * N)
    out = [None] * T
    for i in range(0, T - 1, 2):
        s = F.cosine_similarity(F.normalize(rows[i], dim=1), F.normalize(rows[i + 1], dim=1), dim=1)
        out[i] = rows[i]
        out[i + 1] = rows[i + 1][s.topk(keep_n, largest=False).indices]
    if T % 2:  # official: the pair loop never emits the last frame of an odd-length video
        out[T - 1] = rows.new_zeros(0, rows.shape[-1])
    for i in range(0, T - 4, 4):
        s = F.cosine_similarity(rows[i], rows[i + 2], dim=1)
        out[i] = rows[i]
        out[i + 2] = rows[i + 2][s.topk(keep_n, largest=False).indices]
    return out
