"""Reward (scoring) function template for your own dataset.

GRPO learns only from this score: for each question the model writes several
answers, this function scores each one, and the model is pushed toward the
higher-scoring answers. verl picks a built-in scorer from `data_source` only for
known public datasets (e.g. openai/gsm8k, hiyouga/geometry3k); for your own
data_source you must point verl at this file (see CUSTOM_DATA.md):

  reward.custom_reward_function.path=<this file>
  reward.custom_reward_function.name=compute_score

This template: take the last \\boxed{...} in the response (fallback: text after
"####", then the last number), normalize it, and compare with ground_truth.
Returns 1.0 for correct, 0.0 otherwise. Edit `extract_answer` / `is_correct`
to match how answers look in your data. Pure Python, no extra packages.
"""
import re


def _last_boxed(text: str):
    """Content of the last \\boxed{...}, handling nested braces."""
    start = text.rfind("\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\boxed{"), 1, []
    while i < len(text):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return "".join(out)
        out.append(c)
        i += 1
    return None


def extract_answer(response: str):
    boxed = _last_boxed(response)
    if boxed is not None:
        return boxed
    m = re.search(r"####\s*(.+)", response)
    if m:
        return m.group(1).strip().splitlines()[0]
    nums = re.findall(r"-?\d[\d,]*\.?\d*", response)
    return nums[-1] if nums else None


def normalize(s: str) -> str:
    s = s.strip().strip("$").strip()
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = s.replace("\\,", "").replace(",", "").replace(" ", "").rstrip(".")
    return s.lower()


def is_correct(pred: str, truth: str) -> bool:
    p, t = normalize(pred), normalize(truth)
    if p == t:
        return True
    try:  # numeric answers: 3 == 3.0 == 3.00
        return abs(float(p) - float(t)) < 1e-6
    except ValueError:
        return False


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    pred = extract_answer(solution_str or "")
    if pred is None:
        return 0.0
    return 1.0 if is_correct(pred, str(ground_truth)) else 0.0
