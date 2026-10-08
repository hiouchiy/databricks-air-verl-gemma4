"""Prepare the tiny hands-on datasets (verl format) and write them to a UC Volume.

CPU only — no GPU, no custom image. Runs as a Databricks serverless (CPU) job task
(see prep_data_job.json / setup.md §4-2), or anywhere with `datasets` + `pyarrow`
and access to the Volume path.

  python prep_data.py --dataset gsm8k --out /Volumes/<catalog>/<schema>/<volume>/gsm8k
  python prep_data.py --dataset geo3k --out /Volumes/<catalog>/<schema>/<volume>/geo3k

This is a "does it run" smoke dataset: 64 train / 8 test samples.
  - gsm8k (text): mirrors verl's examples/data_preprocess/gsm8k.py format.
  - geo3k (image): verl geo3k format with an `images` column (for setup.md §6).
"""
import argparse
import os
import re
import tempfile

# Use a private HF cache per process. Serverless presets a shared HF_DATASETS_CACHE
# (/tmp/.hf.data.cache); two tasks on the same node then collide on its lock files
# (PermissionError). Must be set before `datasets` is imported.
_HF_TMP = tempfile.mkdtemp(prefix="hf-")
os.environ["HF_HOME"] = _HF_TMP
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_TMP, "datasets")

import datasets  # noqa: E402

MAXN = 64
MAXN_TEST = 8


def gsm8k(out: str):
    instr = r"Let's think step by step and output the final answer after \"####\"."
    ds = "openai/gsm8k"

    def extract_solution(sol: str) -> str:
        # gsm8k answers end with "#### <number>"
        m = re.search(r"#### (\-?[0-9\.\,]+)", sol)
        return m.group(1).replace(",", "").strip() if m else sol.strip()

    def mk(split):
        def fn(ex, idx):
            q = ex.pop("question")
            ans = ex.pop("answer")
            gt = extract_solution(ans)
            return {"data_source": "openai/gsm8k",
                    "prompt": [{"role": "user", "content": q + " " + instr}],
                    "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": gt},
                    "extra_info": {"split": split, "index": idx, "answer": ans, "question": q}}
        return fn

    d = datasets.load_dataset(ds, "main")
    tr = d["train"].select(range(min(MAXN, len(d["train"])))).map(mk("train"), with_indices=True)
    te = d["test"].select(range(min(MAXN_TEST, len(d["test"])))).map(mk("test"), with_indices=True)
    return tr, te


def geo3k(out: str):
    instr = (r"You FIRST think about the reasoning process as an internal monologue and then "
             r"provide the final answer. The reasoning process MUST BE enclosed within "
             r"<think> </think> tags. The final answer MUST BE put in \boxed{}.")
    ds = "hiyouga/geometry3k"

    def mk(split):
        def fn(ex, idx):
            problem = ex.pop("problem"); answer = ex.pop("answer"); images = ex.pop("images")
            return {"data_source": ds, "prompt": [{"role": "user", "content": problem + " " + instr}],
                    "images": images, "ability": "math",
                    "reward_model": {"style": "rule", "ground_truth": answer},
                    "extra_info": {"split": split, "index": idx, "answer": answer, "question": problem}}
        return fn

    d = datasets.load_dataset(ds)
    tr = d["train"].select(range(min(MAXN, len(d["train"])))).map(mk("train"), with_indices=True)
    te = d["test"].select(range(min(MAXN_TEST, len(d["test"])))).map(mk("test"), with_indices=True)
    return tr, te


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["gsm8k", "geo3k"], required=True)
    p.add_argument("--out", required=True, help="output directory, e.g. /Volumes/<c>/<s>/<v>/gsm8k")
    a = p.parse_args()
    tr, te = {"gsm8k": gsm8k, "geo3k": geo3k}[a.dataset](a.out)
    os.makedirs(a.out, exist_ok=True)
    tr.to_parquet(os.path.join(a.out, "train.parquet"))
    te.to_parquet(os.path.join(a.out, "test.parquet"))
    print(f"Wrote {len(tr)} train / {len(te)} test to {a.out}")


if __name__ == "__main__":
    main()
