"""Save the ORIGINAL VideoLLaMA3-7B mm_projector weights to a small file.

Phase-1 fine-tuned mm_projector (lr 5e-6) on compressed tokens only; raw (uncompressed)
frames are projected with the original VideoLLaMA3 projector instead (vbe/infer.py).
The vision encoder and the LLM are bit-identical between the two checkpoints.

    python scripts/extract_orig_projector.py \
        /workspace/chicheng/OnlineLong/pretrained_models/videollama3_7b_local checkpoints/vl3_orig_projector.pt
"""
import os
import sys

import torch

src, dst = sys.argv[1], sys.argv[2]
path = os.path.join(src, "pytorch_model.bin")
if os.path.exists(path):
    sd = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
else:
    from safetensors.torch import load_file
    import json
    index = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
    files = {f for k, f in index.items() if k.startswith("model.mm_projector.")}
    sd = {}
    for f in files:
        sd.update(load_file(os.path.join(src, f)))
proj = {k[len("model.mm_projector."):]: v.clone() for k, v in sd.items() if k.startswith("model.mm_projector.")}
assert proj, f"no model.mm_projector.* weights in {src}"
os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
torch.save(proj, dst)
print(f"saved {sorted(proj)} -> {dst}")
