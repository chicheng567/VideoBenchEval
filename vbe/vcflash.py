"""VideoChat-Flash (arXiv 2501.00574) on the same benchmarks, prompts and scoring as
vbe/infer.py, at matched LLM visual-token budgets.

The model spends a fixed 16 tokens per frame (4-frame clips -> 64 tokens, HiCo), so a
budget B means at most B // 16 frames, sampled by the model's own loader (1 fps in
8-frame steps, at least 64 frames, uniform when capped). Its in-LLM token dropping
(mm_llm_compress) stays off, so the LLM sees exactly those tokens. Needs the
checkpoint's transformers (4.40.1, see scripts/run_vcflash.sh).

    python -m vbe.vcflash --model_path /root/models/VideoChat-Flash-Qwen2-7B_res448 \
        --benchmarks videomme,mlvu --budgets 1024,4096 --output results/vcflash7b_b{budget}/{bench}/shard0of1.jsonl

Work goes budget by budget (all benchmarks at the smallest budget first), so a run cut
short leaves whole budgets done. Questions already in any jsonl of an output directory
are skipped, so a run can be resumed with a different number of shards.
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

from vbe.benchmarks import DEFAULT_ROOT, build_prompt, extract_choice, load_benchmark

TOKENS_PER_FRAME = 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--benchmarks", required=True, help="comma-separated: videomme,mlvu,longvideobench")
    p.add_argument("--data_root", default=DEFAULT_ROOT)
    p.add_argument("--budgets", required=True, help="comma-separated visual-token budgets")
    p.add_argument("--output", required=True, help="jsonl path with '{budget}' and '{bench}'")
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=16)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only the first N videos of this shard (debug)")
    return p.parse_args()


class Clips(Dataset):
    """(benchmark, video, budget, questions) -> the model's frames for that budget, preprocessed."""

    def __init__(self, items, load_video, processor):
        self.items, self.load_video, self.processor = items, load_video, processor

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        bench, video, b, samples = self.items[i]
        out = dict(bench=bench, video=video, budget=b, samples=samples, error=None)
        try:
            frames, msg = self.load_video(video, max_num_frames=b // TOKENS_PER_FRAME)
            px = self.processor.preprocess(frames, return_tensors="pt")["pixel_values"]
            out.update(pixels=px.to(torch.bfloat16), msg=msg, hw=list(frames[0].shape[:2]), n_frames=len(frames))
        except Exception as e:  # unreadable video: every question of it gets an error record
            out["error"] = repr(e)
        return out


def main():
    args = parse_args()
    budgets = sorted(int(b) for b in args.budgets.split(","))
    benches = args.benchmarks.split(",")
    outputs = {(bench, b): args.output.replace("{budget}", str(b)).replace("{bench}", bench)
               for bench in benches for b in budgets}
    done = {}
    for key, path in outputs.items():
        done[key] = {json.loads(l)["qid"] for f in glob.glob(os.path.join(os.path.dirname(path), "*.jsonl"))
                     for l in open(f) if l.strip()}
    shard = {}
    for bench in benches:
        groups = {}
        for s in load_benchmark(bench, args.data_root):
            groups.setdefault(s["video"], []).append(s)
        shard[bench] = sorted(groups.items())[args.shard_id :: args.num_shards]
        if args.limit:
            shard[bench] = shard[bench][: args.limit]
    items = [(bench, v, b, [s for s in ss if s["qid"] not in done[(bench, b)]])
             for b in budgets for bench in benches for v, ss in shard[bench]]
    items = [it for it in items if it[3]]
    n_q = sum(len(it[3]) for it in items)
    print(f"[vcflash {args.benchmarks} shard {args.shard_id}/{args.num_shards}] budgets {budgets}: "
          f"{len(items)} (video, budget) / {n_q} questions to go", flush=True)
    if not items:
        return

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = 151643  # as the checkpoint's chat()
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16)
    model = model.cuda().eval()
    model.config.mm_llm_compress = False
    rc = sys.modules[type(model).__module__]  # the checkpoint's own helpers
    processor = model.get_vision_tower().image_processor

    # answer every question of a (video, budget) from one encoder pass
    cache, encode = {}, model.encode_video_image

    def encode_once(images_list, video_idx_in_batch):
        if "out" not in cache:
            cache["out"] = encode(images_list, video_idx_in_batch)
        return cache["out"]

    model.encode_video_image = encode_once

    loader = DataLoader(Clips(items, rc.load_video, processor), batch_size=None, shuffle=False,
                        num_workers=args.num_workers, prefetch_factor=1 if args.num_workers else None)
    fouts = {}
    for key, path in outputs.items():
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        fouts[key] = open(path, "a")
    t0, n_done = time.time(), 0
    for item in loader:
        bench, b, err = item["bench"], item["budget"], item["error"]
        cache.clear()
        px = item["pixels"].cuda() if err is None else None
        for s in item["samples"]:
            rec = dict(qid=s["qid"], video=os.path.basename(s["video"]), answer=s["answer"], meta=s["meta"])
            if err is None:
                try:
                    conv = rc.conv_templates["qwen_2"].copy()
                    conv.append_message(conv.roles[0], f"{rc.DEFAULT_IMAGE_TOKEN}\n{item['msg'].strip()} "
                                                       f"{build_prompt(bench, s)}")
                    conv.append_message(conv.roles[1], None)
                    ids = rc.tokenizer_image_token(conv.get_prompt(), tok, rc.IMAGE_TOKEN_INDEX,
                                                   return_tensors="pt").unsqueeze(0).cuda()
                    sep = conv.sep if conv.sep_style != rc.SeparatorStyle.TWO else conv.sep2
                    stop = rc.KeywordsStoppingCriteria([sep], tok, ids)
                    with torch.inference_mode():
                        out = model.generate(inputs=ids, images=[px], attention_mask=ids.ne(tok.pad_token_id).long(),
                                             modalities=["video"], image_sizes=[item["hw"]], use_cache=True,
                                             stopping_criteria=[stop], do_sample=False, num_beams=1,
                                             max_new_tokens=args.max_new_tokens)
                    resp = tok.batch_decode(out, skip_special_tokens=True)[0].strip()
                    if resp.endswith(sep):
                        resp = resp[: -len(sep)].strip()
                    n_vis = int(cache["out"][0].shape[:-1].numel())
                    dur = re.search(r"lasts for ([\d.]+) seconds", item["msg"])
                    rec.update(pred=extract_choice(bench, resp, s["options"]), response=resp,
                               seq_len=ids.shape[1] - 1 + n_vis, mode="vcflash", vision_tokens=n_vis,
                               video_info=dict(n_frames=item["n_frames"], native_hw=item["hw"], duration=float(dur.group(1)) if dur else None,
                                               msg=item["msg"].strip()))
                except torch.cuda.OutOfMemoryError as e:
                    torch.cuda.empty_cache()
                    rec.update(pred="", response="", error=repr(e))
            else:
                rec.update(pred="", response="", error=err)
            fouts[(bench, b)].write(json.dumps(rec, ensure_ascii=False) + "\n")
            fouts[(bench, b)].flush()
            n_done += 1
        del px
        cache.clear()
        if n_done and n_done % 50 < len(item["samples"]):
            el = time.time() - t0
            print(f"  {n_done}/{n_q} questions, {el / n_done:.2f}s/q, eta {(n_q - n_done) * el / n_done / 60:.0f} min",
                  flush=True)
    for f in fouts.values():
        f.close()
    print(f"[vcflash shard {args.shard_id}] done: {n_done} questions in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
