"""Create SAMPLE "customer-style" raw data to try the custom-dataset flow.

Real customers already have their own files; this script only fabricates
look-alike raw data from public datasets so you can rehearse CUSTOM_DATA.md:

  --kind text  : <out>/data.jsonl               lines: {"question": ..., "answer": ...}
                 (from openai/gsm8k; answer = final number only)
  --kind image : <out>/data.jsonl + <out>/images/*.png
                 lines: {"question": ..., "answer": ..., "image": "images/0000.png"}
                 (from hiyouga/geometry3k; "<image>" removed from the question,
                  as raw data usually has no placeholder)

CPU only. Runs as a serverless CPU job (custom_sample_raw_job.json).
"""
import argparse
import json
import os
import re
import tempfile

# Private HF cache per process (serverless presets a shared one; see prep_data.py).
_HF_TMP = tempfile.mkdtemp(prefix="hf-")
os.environ["HF_HOME"] = _HF_TMP
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_TMP, "datasets")

import datasets  # noqa: E402

N = 72  # 64 train + 8 test after prep_custom.py


def text(out: str):
    d = datasets.load_dataset("openai/gsm8k", "main")["train"].select(range(N))
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "data.jsonl"), "w") as f:
        for ex in d:
            m = re.search(r"#### (\-?[0-9\.\,]+)", ex["answer"])
            ans = m.group(1).replace(",", "").strip() if m else ex["answer"]
            f.write(json.dumps({"question": ex["question"], "answer": ans}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(d)} rows to {out}/data.jsonl")


def image(out: str):
    d = datasets.load_dataset("hiyouga/geometry3k")["train"].select(range(N))
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    with open(os.path.join(out, "data.jsonl"), "w") as f:
        for i, ex in enumerate(d):
            rel = f"images/{i:04d}.png"
            ex["images"][0].save(os.path.join(out, rel))
            q = ex["problem"].replace("<image>", "").strip()
            f.write(json.dumps({"question": q, "answer": ex["answer"], "image": rel}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(d)} rows + images to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kind", choices=["text", "image"], required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    {"text": text, "image": image}[a.kind](a.out)
