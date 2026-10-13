# Real grading mistakes (found by reading 408 dumped answers from Gemma 4 GRPO runs)

Each was a bug in an earlier version of `reward_template.py`; the current template handles all three.
Use them as test cases whenever the data has similar answers.

| Ground truth | Model's final answer | Old score | Correct score | Cause / fix |
|---|---|---|---|---|
| `90` | `\boxed{90^\circ}` | 0 | 1 | Degree marker not stripped. Strip `^\circ`, `^{\circ}`, `°`, `\degree` before comparing. |
| `1200` | `\boxed{\$1,200}` | 0 | 1 | Escaped dollar `\$` not stripped (only a leading `$` was). Remove `\$` and `$` anywhere. |
| `90` | *(no `\boxed{}`; response cut off at the length limit while still reasoning, mentioning 90 on the way)* | 1 | 0 | "Last number in the text" fallback gave credit to an unfinished answer. Grade only the final answer; no final answer → 0. |

Other observations from the same runs:
- Text (gsm8k-like): 240/240 responses used `\boxed{}`; genuinely wrong answers (e.g. 29 vs 59) were scored 0 correctly.
- Image (geometry): 16/48 responses were cut off before `\boxed{}` at MAX_RESPONSE_LEN=512. If many
  answers are cut off, raise MAX_RESPONSE_LEN rather than loosening the grader.
