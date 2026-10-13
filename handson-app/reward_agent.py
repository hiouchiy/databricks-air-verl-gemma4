"""Reward-function agent: UC skill + Claude (Foundation Model API) + Databricks Sandbox.

author (Claude writes custom_reward.py + tests, following the UC skill)
  -> test (run the tests in a Sandbox)
  -> back to author with the failures, up to MAX_ATTEMPTS times.

Also: `trial_score` — Claude answers a few real questions from the dataset, and the
reward function scores those answers in the Sandbox, so a human can judge whether the
scores make sense before approving the function.

Runs as the app's service principal (needs READ VOLUME on the skill's schema and
CAN_QUERY on the LLM endpoint; the Sandbox API needs no extra grant).
"""
import base64
import json
import os
import re
import time
from typing import Callable, TypedDict

import requests
from databricks.sdk.core import Config
from langgraph.graph import END, StateGraph
from openai import OpenAI

cfg = Config()
HOST = cfg.host.rstrip("/")
LLM = os.environ.get("AIR_LLM_ENDPOINT", "databricks-claude-sonnet-5-5")
MAX_ATTEMPTS = 3
HERE = os.path.dirname(os.path.abspath(__file__))


def _headers():
    return cfg.authenticate()


def _api(method, path, timeout=120, **kw):
    r = requests.request(method, f"{HOST}{path}", headers=_headers(), timeout=timeout, **kw)
    ctype = r.headers.get("content-type", "")
    return r.status_code, (r.json() if "json" in ctype and r.text else r.text)


# ---- UC skill -------------------------------------------------------------------
def skill_files(skill: str) -> dict:
    """Read every file of a UC skill (catalog.schema.skill) via the Files API."""
    cat, sch, name = skill.split(".")
    code, listing = _api("GET", f"/api/2.0/fs/directories/Skills/{cat}/{sch}/{name}/")
    if code != 200:
        raise RuntimeError(f"cannot list UC skill {skill}: {code} {str(listing)[:300]}")
    out = {}
    for e in listing.get("contents", []):
        rel = e["path"].split(f"/{name}/", 1)[1]
        r = requests.get(f"{HOST}/api/2.0/fs/files{e['path']}", headers=_headers(), timeout=60)
        if r.status_code == 200:
            out[rel] = r.text
    if "SKILL.md" not in out:
        raise RuntimeError(f"UC skill {skill} has no SKILL.md")
    return out


# ---- Sandbox --------------------------------------------------------------------
class Sandbox:
    def __enter__(self):
        self.id = f"reward-agent-{int(time.time() * 1000)}"
        code, body = _api("POST", f"/api/2.0/sandboxes?sandbox_id={self.id}",
                          json={"display_name": "reward agent", "spec": {"compute": {"inactivity_timeout": "900s"}}})
        if code != 200:
            raise RuntimeError(f"cannot create sandbox: {code} {str(body)[:300]}")
        for _ in range(100):
            _, s = _api("GET", f"/api/2.0/sandboxes/{self.id}")
            if isinstance(s, dict) and s.get("status", {}).get("state") == "SANDBOX_STATE_RUNNING":
                return self
            time.sleep(3)
        raise RuntimeError("sandbox did not start")

    def run(self, files: dict, command: str) -> dict:
        script = "set -e; rm -rf /home/sandbox-agent/job; mkdir -p /home/sandbox-agent/job; cd /home/sandbox-agent/job\n"
        for name, body in files.items():
            script += f"echo {base64.b64encode(body.encode()).decode()} | base64 -d > {name}\n"
        script += command
        code, body = _api("POST", f"/api/2.0/sandbox-exec/sandboxes/{self.id}/exec-sync",
                          json={"cmd": "/bin/bash", "args": ["-c", script]}, timeout=300)
        if code != 200:
            raise RuntimeError(f"sandbox exec failed: {code} {str(body)[:300]}")
        return body

    def __exit__(self, *exc):
        _api("DELETE", f"/api/2.0/sandboxes/{self.id}")


# ---- LLM ------------------------------------------------------------------------------
def llm(messages, max_tokens=16000) -> str:
    token = _headers()["Authorization"].split(" ", 1)[1]
    client = OpenAI(api_key=token, base_url=f"{HOST}/serving-endpoints")
    # Claude on FMAPI: no `temperature`; content can be a list of parts.
    r = client.chat.completions.create(model=LLM, messages=messages, max_tokens=max_tokens)
    content = r.choices[0].message.content
    if isinstance(content, list):
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                          if not isinstance(p, dict) or p.get("type") in (None, "text"))
    return content or ""


def _block(text: str, lang: str):
    m = re.findall(rf"```{lang}\s*\n(.*?)```", text, re.S | re.I)
    return m[-1] if m else None


def _valid_tests(text) -> bool:
    try:
        t = json.loads(text)
        return isinstance(t, list) and len(t) > 0 and all({"response", "ground_truth", "expected"} <= set(x) for x in t)
    except Exception:
        return False


# ---- agent graph -----------------------------------------------------------------------
class State(TypedDict, total=False):
    skill: dict
    samples: list
    criteria: str
    code: str
    tests: str
    report: dict
    attempts: int
    sandbox: Sandbox
    progress: Callable[[str], None]


def _author(state: State) -> State:
    say = state["progress"]
    n = state.get("attempts", 0) + 1
    say(f"Claude が採点関数とテストを書いています（{n} 回目）")
    sk = state["skill"]
    system = ("You follow this skill exactly.\n\n# SKILL.md\n" + sk["SKILL.md"] +
              "".join(f"\n\n# {k}\n```\n{v}\n```" for k, v in sk.items() if k not in ("SKILL.md", "test_harness.py")))
    user = (f"Sample rows from the user's dataset (question, ground_truth):\n"
            f"{json.dumps(state['samples'], ensure_ascii=False, indent=1)}\n\n"
            f"User's grading criteria (may be Japanese): {state['criteria'] or '(none given; use the skill defaults)'}\n\n"
            "Return exactly two fenced blocks: ```python with the complete custom_reward.py, and ```json with the "
            "test-case list described in the skill. Keep explanations short.")
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    if state.get("report"):
        failed = [r for r in state["report"]["results"] if not r.get("ok")]
        msgs += [{"role": "assistant", "content": f"```python\n{state['code']}```\n```json\n{state['tests']}```"},
                 {"role": "user", "content": "These tests failed in the sandbox. Fix the function (change a test only if "
                  "it contradicts the criteria) and return both blocks again:\n" + json.dumps(failed, ensure_ascii=False)}]
    out = llm(msgs)
    code, tests = _block(out, "python"), _block(out, "json")
    if code is None:
        raise RuntimeError("Claude が ```python ブロックを返しませんでした")
    if not (tests and _valid_tests(tests)):
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": "Now return ONLY the test cases as one ```json block (a list of "
                  '{"response","ground_truth","expected","why"} objects).'}]
        tests = _block(llm(msgs), "json")
        if not (tests and _valid_tests(tests)):
            raise RuntimeError("Claude が有効なテストケースを返しませんでした")
    return {"code": code, "tests": tests, "attempts": n}


def _test(state: State) -> State:
    state["progress"]("Sandbox でテストを実行しています")
    res = state["sandbox"].run({"custom_reward.py": state["code"], "test_harness.py": state["skill"]["test_harness.py"],
                                "tests.json": state["tests"]}, "python3 test_harness.py custom_reward.py tests.json")
    out = res.get("stdout", "")
    try:
        report = json.loads(out)
    except Exception:
        report = {"passed": 0, "failed": 1, "total": 1,
                  "results": [{"ok": False, "error": (res.get("stderr") or out)[-2000:], "why": "harness error"}]}
    state["progress"](f"テスト結果: {report['passed']} / {report['total']} 件合格")
    return {"report": report}


def _route(state: State):
    return END if state["report"]["failed"] == 0 or state["attempts"] >= MAX_ATTEMPTS else "author"


_g = StateGraph(State)
_g.add_node("author", _author)
_g.add_node("test", _test)
_g.set_entry_point("author")
_g.add_edge("author", "test")
_g.add_conditional_edges("test", _route)
GRAPH = _g.compile()


def write_reward(skill: str, samples: list, criteria: str, progress: Callable[[str], None]) -> dict:
    progress("UC のスキルを読み込んでいます")
    files = skill_files(skill)
    progress("Sandbox を起動しています")
    with Sandbox() as sb:
        final = GRAPH.invoke({"skill": files, "samples": samples, "criteria": criteria,
                              "sandbox": sb, "attempts": 0, "progress": progress})
    rep = final["report"]
    return {"code": final["code"], "tests": json.loads(final["tests"]), "report": rep,
            "attempts": final["attempts"], "all_passed": rep["failed"] == 0}


def run_tests(skill: str, code: str, tests: list) -> dict:
    files = skill_files(skill)
    with Sandbox() as sb:
        res = sb.run({"custom_reward.py": code, "test_harness.py": files["test_harness.py"],
                      "tests.json": json.dumps(tests, ensure_ascii=False)},
                     "python3 test_harness.py custom_reward.py tests.json || true")
    return json.loads(res.get("stdout") or '{"passed":0,"failed":0,"total":0,"results":[]}')


TRIAL_SCRIPT = r'''
import importlib.util, json
spec = importlib.util.spec_from_file_location("r", "custom_reward.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
rows = json.load(open("answers.json"))
for r in rows:
    try:
        r["score"] = float(m.compute_score("custom", r["response"], r["ground_truth"]))
    except Exception as e:
        r["score"] = None; r["error"] = f"{type(e).__name__}: {e}"
print(json.dumps(rows, ensure_ascii=False))
'''


def _image_part(data: bytes, path: str) -> dict:
    ext = path.rsplit(".", 1)[-1].lower()
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp", "gif": "gif"}.get(ext, "png")
    return {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{base64.b64encode(data).decode()}"}}


def trial_score(code: str, samples: list, instruction: str, read_file: Callable[[str], bytes],
                progress: Callable[[str], None]) -> list:
    """Claude answers each sample question (with its images, if any), then the reward function
    scores the answers in the Sandbox — so a human can judge whether the scores make sense."""
    answers = []
    for i, s in enumerate(samples):
        progress(f"Claude が問題を解いています（{i + 1} / {len(samples)}）")
        text = f"{s['question']} {instruction}".strip()
        imgs = s.get("images") or []
        content = [*(_image_part(read_file(p), p) for p in imgs), {"type": "text", "text": text}] if imgs else text
        reply = llm([{"role": "user", "content": content}], max_tokens=2000)
        answers.append({"question": s["question"], "ground_truth": s["ground_truth"], "response": reply,
                        "n_images": len(imgs)})
    progress("Sandbox で採点しています")
    with Sandbox() as sb:
        res = sb.run({"custom_reward.py": code, "trial.py": TRIAL_SCRIPT,
                      "answers.json": json.dumps(answers, ensure_ascii=False)}, "python3 trial.py")
    return json.loads(res.get("stdout") or "[]")
