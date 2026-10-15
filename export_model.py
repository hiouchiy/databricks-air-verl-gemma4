"""Turn verl's saved HF model into a compact bf16 Hugging Face model (CPU only).

verl (save_contents=['hf_model']) writes <ckpt>/global_step_N/actor/huggingface/ with the
FSDP master weights in fp32, although config.json says bfloat16 -- about twice the size
needed to serve the model. This script copies that folder to --dst with every floating
tensor cast to bf16, reading one tensor at a time and writing ~--shard-gb shards, so it
fits a small serverless CPU.

  python export_model.py --src <ckpt>/global_step_3/actor/huggingface --dst /tmp/ckpt/model_bf16

run_grpo*.sh call it on the training node, on its local disk, right after training
(save_model: true). Reading a 100 GB checkpoint back from a Volume on a serverless CPU
was far too slow (~14 min per 5 GB), so keep both folders on fast local disk.
"""
import argparse
import json
import os
import shutil
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="verl's .../actor/huggingface folder")
    p.add_argument("--dst", required=True, help="output folder (created)")
    p.add_argument("--shard-gb", type=float, default=5.0)
    a = p.parse_args()
    t0 = time.time()
    os.makedirs(a.dst, exist_ok=True)

    index = json.load(open(os.path.join(a.src, "model.safetensors.index.json")))
    by_file = {}
    for name, f in index["weight_map"].items():
        by_file.setdefault(f, []).append(name)

    # everything except the weights (config, tokenizer, chat template, generation config)
    for f in os.listdir(a.src):
        if not f.endswith(".safetensors") and f != "model.safetensors.index.json":
            shutil.copy(os.path.join(a.src, f), os.path.join(a.dst, f))

    limit = int(a.shard_gb * 1024**3)
    shards, cur, cur_bytes, total = [], {}, 0, 0

    def flush():
        nonlocal cur, cur_bytes
        if cur:
            name = f"part-{len(shards) + 1:05d}.safetensors"
            save_file(cur, os.path.join(a.dst, name), metadata={"format": "pt"})
            print(f"wrote {name} ({cur_bytes / 1e9:.1f} GB, {len(cur)} tensors, {time.time() - t0:.0f}s)", flush=True)
            shards.append(list(cur))
            cur, cur_bytes = {}, 0

    for f in sorted(by_file):
        with safe_open(os.path.join(a.src, f), framework="pt") as r:
            for name in by_file[f]:
                t = r.get_tensor(name)
                if t.is_floating_point():
                    t = t.to(torch.bfloat16)
                t = t.contiguous()
                n = t.numel() * t.element_size()
                if cur and cur_bytes + n > limit:
                    flush()
                cur[name] = t
                cur_bytes += n
                total += n
    flush()

    # rename part-xxxxx to the usual model-0000i-of-0000n names and write the index
    n = len(shards)
    weight_map = {}
    for i, names in enumerate(shards, 1):
        final = f"model-{i:05d}-of-{n:05d}.safetensors"
        os.replace(os.path.join(a.dst, f"part-{i:05d}.safetensors"), os.path.join(a.dst, final))
        weight_map.update({k: final for k in names})
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(a.dst, "model.safetensors.index.json"), "w"), indent=2)
    print(f"done: {total / 1e9:.1f} GB in {n} shards -> {a.dst} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
