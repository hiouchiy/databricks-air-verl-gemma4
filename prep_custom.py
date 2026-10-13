"""Convert your own Q&A data (CSV / JSONL, optionally with images) into verl's
RL dataset format (parquet) for GRPO. CPU only — runs as a serverless CPU job
(custom_prep_job.json) or anywhere with pandas + pyarrow.

Input (one row = one problem):
  - CSV or JSONL with at least a question column and an answer column.
  - For image problems, an image column with a file path (relative to --image-dir,
    or absolute such as /Volumes/...). Several images: separate paths with "|".

Output (written to --out): train.parquet / test.parquet with these columns
  data_source  : str   your dataset name; selects the reward (scoring) function
  prompt       : [{"role": "user", "content": "<image>question + instruction"}]
  images       : [{"bytes": <image bytes>, "path": ""}]   (image data only)
  ability      : str   free label
  reward_model : {"style": "rule", "ground_truth": "<answer>"}
  extra_info   : {"split", "index", "question", "answer"}

Example:
  python prep_custom.py --input /Volumes/c/s/v/raw/text/questions.csv \
      --data-source acme/math-qa --out /Volumes/c/s/v/custom/text
  python prep_custom.py --input /Volumes/c/s/v/raw/image/data.jsonl \
      --image-dir /Volumes/c/s/v/raw/image --data-source acme/geometry \
      --out /Volumes/c/s/v/custom/image
See CUSTOM_DATA.md.
"""
import argparse
import os
import random

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_INSTRUCTION = ("Think step by step, then give the final answer in \\boxed{} "
                       "(for example \\boxed{42}).")

SCHEMA_FIELDS = [
    ("data_source", pa.string()),
    ("prompt", pa.list_(pa.struct([("content", pa.string()), ("role", pa.string())]))),
    ("ability", pa.string()),
    ("reward_model", pa.struct([("ground_truth", pa.string()), ("style", pa.string())])),
    ("extra_info", pa.struct([("answer", pa.string()), ("index", pa.int64()),
                              ("question", pa.string()), ("split", pa.string())])),
]
IMAGES_FIELD = ("images", pa.list_(pa.struct([("bytes", pa.binary()), ("path", pa.string())])))


def read_table(path: str) -> pd.DataFrame:
    if path.endswith(".jsonl") or path.endswith(".json"):
        return pd.read_json(path, lines=path.endswith(".jsonl"), dtype=False)
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, help="CSV or JSONL file")
    p.add_argument("--data-source", required=True, help="your dataset name, e.g. acme/math-qa")
    p.add_argument("--out", required=True, help="output directory (train.parquet / test.parquet)")
    p.add_argument("--question-col", default="question")
    p.add_argument("--answer-col", default="answer")
    p.add_argument("--image-col", default="image", help="image path column (image data only)")
    p.add_argument("--image-dir", default=None, help="base directory for relative image paths")
    p.add_argument("--instruction", default=DEFAULT_INSTRUCTION, help="appended to every question")
    p.add_argument("--ability", default="custom")
    p.add_argument("--n-test", type=int, default=8, help="rows held out for validation")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    df = read_table(a.input)
    for col in (a.question_col, a.answer_col):
        if col not in df.columns:
            raise SystemExit(f"column '{col}' not found in {a.input} (columns: {list(df.columns)})")
    has_images = a.image_col in df.columns
    image_dir = a.image_dir or os.path.dirname(os.path.abspath(a.input))

    rows = []
    for i, r in enumerate(df.to_dict("records")):
        question = str(r[a.question_col]).strip()
        answer = str(r[a.answer_col]).strip()
        content = f"{question} {a.instruction}".strip()
        row = {"data_source": a.data_source, "ability": a.ability,
               "reward_model": {"ground_truth": answer, "style": "rule"},
               "extra_info": {"answer": answer, "index": i, "question": question, "split": ""}}
        if has_images:
            paths = [x for x in str(r.get(a.image_col) or "").split("|") if x]
            images = []
            for path in paths:
                full = path if os.path.isabs(path) else os.path.join(image_dir, path)
                with open(full, "rb") as f:
                    images.append({"bytes": f.read(), "path": ""})
            # verl inserts images where "<image>" appears, in order; without the
            # placeholder the image is silently ignored. Put one per image up front.
            missing = len(images) - content.count("<image>")
            if missing > 0:
                content = "<image>" * missing + content
            row["images"] = images
        row["prompt"] = [{"content": content, "role": "user"}]
        rows.append(row)

    if len(rows) <= a.n_test:
        raise SystemExit(f"need more than --n-test={a.n_test} rows, got {len(rows)}")
    random.Random(a.seed).shuffle(rows)
    test, train = rows[:a.n_test], rows[a.n_test:]
    schema = pa.schema(SCHEMA_FIELDS + ([IMAGES_FIELD] if has_images else []))
    os.makedirs(a.out, exist_ok=True)
    for name, part in (("train", train), ("test", test)):
        for j, row in enumerate(part):
            row["extra_info"]["split"] = name
            row["extra_info"]["index"] = j
        pq.write_table(pa.Table.from_pylist(part, schema=schema), os.path.join(a.out, f"{name}.parquet"))
    print(f"Wrote {len(train)} train / {len(test)} test to {a.out} "
          f"(data_source={a.data_source}, images={'yes' if has_images else 'no'})")


if __name__ == "__main__":
    main()
