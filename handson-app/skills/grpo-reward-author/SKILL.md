---
name: grpo-reward-author
description: Write, test, and hand over a rule-based reward (scoring) function `compute_score` for GRPO training with verl on a custom dataset whose answers are short (numbers, choices, short words/expressions). Use when a user needs a custom_reward.py for their own data_source, wants to change how answers are graded, or asks why rewards look wrong.
---

# Writing a GRPO reward function for verl

GRPO learns **only** from this function: for each question the model writes several answers,
`compute_score` grades each one, and the model is pushed toward higher-scoring answers. A grader
that is too lenient rewards wrong or unfinished answers; one that is too strict punishes correct
answers written in a slightly different form. Both quietly ruin training, so every function you
write must be tested against concrete cases before a human approves it.

## Hard requirements (never violate)
1. Signature — verl calls exactly this:
   ```python
   def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
   ```
   `solution_str` is the model's whole response; `ground_truth` is the dataset's answer string.
2. Return a float in `[0.0, 1.0]`. Use `1.0` / `0.0` unless the user explicitly asks for partial credit.
3. **Python standard library only.** No network, no file or environment access, no randomness,
   deterministic, fast (well under 10 ms per call).
4. **Never raise.** Any unexpected input (None, empty, odd types) returns `0.0`.
5. **Grade only the final answer** the prompt asked for (by default the last `\boxed{...}`;
   `#### <answer>` as a fallback). If the response never states a final answer (for example it was
   cut off at the length limit), return `0.0`. Do NOT fall back to "the last number in the text":
   a cut-off response then gets credit for a number it merely mentioned on the way.
6. The instruction appended to each question and the extraction rule must match. The default
   instruction is: *"Think step by step, then give the final answer in \boxed{} (for example \boxed{42})."*

## Procedure
1. **Understand the answers.** Read the sample rows you are given (question + ground_truth) and the
   user's grading criteria. Decide the answer type:
   - numeric (integers, decimals, fractions, with or without units / currency / %),
   - choice (A/B/C/D, yes/no, a fixed label set),
   - short text / expression (a word, a name, a formula such as `6\sqrt{3}`).
   If ground truths are long free-form text (sentences, explanations), STOP: a rule-based grader
   is not suitable. Say so and recommend an LLM-as-a-judge reward instead (not covered by this skill).
2. **Ask only what you cannot infer** (keep it to a few questions), e.g.: Should `3` and `3.0`
   match? Are units/currency (`°`, `%`, `$`, `km`) ignorable? Is case ignorable for labels?
   Should a numeric answer be accepted within a tolerance (absolute or relative)?
   Is partial credit wanted?
3. **Start from `reward_template.py`** in this skill. It already handles `\boxed{}` with nested braces,
   `####`, commas, spaces, `$` / `\$`, degrees (`^\circ`, `°`, `\degree`), `%`, `\text{}`, and numeric
   equality. Change only what the data and the criteria require — extraction (`extract_answer`),
   normalization (`normalize`) and comparison (`is_correct`).
4. **Write test cases** as a JSON list of `{"response", "ground_truth", "expected", "why"}`.
   For several real ground truths from the samples, include at least:
   - the correct answer in the expected format → `1.0`
   - the correct answer in each accepted variant (commas, units, currency, LaTeX, `.0`) → `1.0`
   - a wrong answer in the expected format → `0.0`
   - no final answer at all → `0.0`
   - a cut-off response that mentions the correct value but has no final answer → `0.0`
   - several `\boxed{}` where only the last is correct / only the first is correct
   - anything the user's criteria call out (tolerance edges, case, label synonyms)
5. **Run the tests in the sandbox** with `test_harness.py`:
   ```bash
   python3 test_harness.py custom_reward.py tests.json
   ```
   It prints a JSON report (`passed`, `failed`, per-case results) and exits non-zero on failure.
   Fix the function (not the tests, unless a test contradicts the user's criteria) and rerun until all pass.
6. **Hand over** for human approval: the final `custom_reward.py`, the test table (case, expected,
   got), and a short list of the assumptions you made. Do not claim the function is correct beyond
   what the tests show.

## After training starts
The custom YAMLs write every answer with its score to `$VOL/rollouts/<experiment>/<step>.jsonl`
(`input`, `output`, `gts`, `score`). Reading 20–50 of those lines is the best check: look for
correct answers scored 0 and unfinished/wrong answers scored 1. See `pitfalls.md` for real examples.
