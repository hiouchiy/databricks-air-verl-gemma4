"""Reward-function agent: UC skill + Claude (Foundation Model API) + Databricks Sandbox.

author (Claude writes custom_reward.py + tests, following the UC skill)
  -> test (run the tests in a Sandbox)
  -> back to author with the failures, up to MAX_ATTEMPTS times.

Also: `trial_score` — Claude answers a few real questions from the dataset, and the
reward function scores those answers in the Sandbox, so a human can judge whether the
scores make sense before approving the function.

Runs as the app's service principal (needs READ VOLUME on the skill's schema and
CAN_QUERY on the LLM endpoint; the Sandbox API needs no extra grant).

Sandboxes are created WITHOUT a Databricks token (by default the creator's token is put
inside, so code run there could act as the app). One warm sandbox is reused across
requests and deleted after SANDBOX_IDLE_SECONDS unused.

`progress(text, **event)` reports structured events to the UI: stage / state
("active" | "done" | "error") / detail, plus terminal output for sandbox commands.
"""
import base64
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from typing import Callable, TypedDict

import requests
from databricks.sdk.core import Config
from langgraph.graph import END, StateGraph
from openai import OpenAI

cfg = Config()
HOST = cfg.host.rstrip("/")
LLM = os.environ.get("AIR_LLM_ENDPOINT", "databricks-claude-sonnet-5-5")
MAX_ATTEMPTS = 3
SANDBOX_IDLE_SECONDS = int(os.environ.get("AIR_SANDBOX_IDLE_SECONDS", "600"))
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
def _tail(text: str, n: int = 12) -> list:
    lines = (text or "").rstrip("\n").splitlines()
    return lines[-n:]


class Sandbox:
    def __init__(self):
        self.id = f"reward-agent-{int(time.time() * 1000)}"
        self.created = None
        self.uses = 0

    def start(self, progress):
        t0 = time.time()
        progress("Sandbox を作成しています", stage="sb-create", state="active", detail=f"ID: {self.id}")
        code, body = _api("POST", f"/api/2.0/sandboxes?sandbox_id={self.id}",
                          json={"display_name": "reward agent",
                                "spec": {"compute": {"inactivity_timeout": "900s"},
                                         "identity": {"databricks_access_token_included": False}}})
        if code != 200:
            progress("Sandbox を作成できませんでした", stage="sb-create", state="error", detail=f"{code} {str(body)[:200]}")
            raise RuntimeError(f"cannot create sandbox: {code} {str(body)[:300]}")
        seen = []
        for _ in range(200):
            _, st = _api("GET", f"/api/2.0/sandboxes/{self.id}")
            state = (st.get("status", {}).get("state") or "?").replace("SANDBOX_STATE_", "") if isinstance(st, dict) else "?"
            if not seen or seen[-1] != state:
                seen.append(state)
                progress("Sandbox を作成しています", stage="sb-create", state="active",
                         detail=f"ID: {self.id} ／ 状態: {' → '.join(seen)}（{time.time() - t0:.1f} 秒）")
            if state == "RUNNING":
                self.created = time.time()
                progress("Sandbox を作成しました", stage="sb-create", state="done",
                         detail=f"ID: {self.id} ／ {' → '.join(seen)}（{self.created - t0:.1f} 秒）／ Databricks のトークンなし")
                return self
            time.sleep(0.5)
        progress("Sandbox が起動しませんでした", stage="sb-create", state="error", detail=self.id)
        raise RuntimeError("sandbox did not start")

    def alive(self) -> bool:
        code, st = _api("GET", f"/api/2.0/sandboxes/{self.id}")
        return code == 200 and isinstance(st, dict) and st.get("status", {}).get("state") == "SANDBOX_STATE_RUNNING"

    def run(self, files: dict, command: str, progress=None, stage="sb-exec", label="Sandbox で実行しています") -> dict:
        say = progress or (lambda *a, **k: None)
        sent = [{"name": k, "bytes": len(v.encode())} for k, v in files.items()]
        say(label, stage=stage, state="active", cmd=command, files=sent, sandbox=self.id)
        script = "set -e; rm -rf /home/sandbox-agent/job; mkdir -p /home/sandbox-agent/job; cd /home/sandbox-agent/job\n"
        for name, body in files.items():
            script += f"echo {base64.b64encode(body.encode()).decode()} | base64 -d > {name}\n"
        script += command
        t0 = time.time()
        code, body = _api("POST", f"/api/2.0/sandbox-exec/sandboxes/{self.id}/exec-sync",
                          json={"cmd": "/bin/bash", "args": ["-c", script]}, timeout=300)
        ms = int((time.time() - t0) * 1000)
        if code != 200 or not isinstance(body, dict):
            say(label, stage=stage, state="error", cmd=command, files=sent, ms=ms, stderr=[str(body)[:300]])
            raise RuntimeError(f"sandbox exec failed: {code} {str(body)[:300]}")
        self.uses += 1
        out = body.get("stdout") or ""
        say(label.replace("しています", "しました"), stage=stage, state="done", cmd=command, files=sent, ms=ms,
            exit_code=body.get("exit_code"), stdout=_tail(out) if not out.lstrip().startswith(("{", "[")) else
            [f"（結果の JSON {len(out.encode()):,} バイト）"], stderr=_tail(body.get("stderr") or "", 20))
        return body

    def delete(self):
        _api("DELETE", f"/api/2.0/sandboxes/{self.id}")


class SandboxPool:
    """One warm sandbox shared by all requests (one command at a time); deleted when idle."""
    def __init__(self):
        self.lock = threading.Lock()
        self.sb = None
        self.last_used = 0.0
        self.deleted_at = None
        threading.Thread(target=self._reaper, daemon=True).start()

    @contextmanager
    def use(self, progress):
        with self.lock:
            if self.sb and not self.sb.alive():
                self.sb = None
            if self.sb:
                progress("起動済みの Sandbox を使います", stage="sb-create", state="done",
                         detail=f"ID: {self.sb.id} ／ {int(time.time() - self.sb.created) // 60} 分前に起動・"
                                f"これまで {self.sb.uses} 回実行 ／ Databricks のトークンなし", reused=True)
            else:
                self.sb = Sandbox().start(progress)
            try:
                yield self.sb
            finally:
                self.last_used = time.time()

    def status(self) -> dict:
        sb = self.sb
        if not sb:
            return {"running": False, "deleted_at": self.deleted_at, "idle_limit": SANDBOX_IDLE_SECONDS}
        return {"running": True, "id": sb.id, "age": int(time.time() - sb.created), "uses": sb.uses,
                "idle": int(time.time() - self.last_used), "idle_limit": SANDBOX_IDLE_SECONDS, "busy": self.lock.locked()}

    def delete(self):
        with self.lock:
            if self.sb:
                self.sb.delete()
                self.sb, self.deleted_at = None, time.time()

    def _reaper(self):
        while True:
            time.sleep(20)
            if self.sb and not self.lock.locked() and time.time() - self.last_used > SANDBOX_IDLE_SECONDS:
                try:
                    self.delete()
                except Exception:
                    pass


POOL = SandboxPool()


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
    say(f"Claude が採点関数とテストを書いています（{n} 回目）", stage=f"author-{n}", state="active",
        detail="UC のスキルの手順とひな形に沿って書いています" if n == 1 else "Sandbox で失敗したテストを渡して、直してもらっています")
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
    say(f"Claude が採点関数とテストを書きました（{n} 回目）", stage=f"author-{n}", state="done",
        detail=f"採点関数 {len(code.splitlines())} 行 ／ テスト {len(json.loads(tests))} 件")
    return {"code": code, "tests": tests, "attempts": n}


def _test(state: State) -> State:
    n = state["attempts"]
    res = state["sandbox"].run({"custom_reward.py": state["code"], "test_harness.py": state["skill"]["test_harness.py"],
                                "tests.json": state["tests"]}, "timeout 30 python3 test_harness.py custom_reward.py tests.json",
                               state["progress"], stage=f"test-{n}", label=f"Sandbox でテストを実行しています（{n} 回目）")
    out = res.get("stdout", "")
    try:
        report = json.loads(out)
    except Exception:
        report = {"passed": 0, "failed": 1, "total": 1,
                  "results": [{"ok": False, "error": (res.get("stderr") or out)[-2000:], "why": "harness error"}]}
    state["progress"](f"テスト結果: {report['passed']} / {report['total']} 件合格", stage=f"result-{n}",
                      state="done" if report["failed"] == 0 else "error",
                      detail="すべて合格" if report["failed"] == 0 else
                      ("Claude に直してもらいます" if n < MAX_ATTEMPTS else f"{MAX_ATTEMPTS} 回で打ち切り。内容を確かめてください"))
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
    progress("UC のスキルを読み込んでいます", stage="skill", state="active", detail=skill)
    files = skill_files(skill)
    progress("UC のスキルを読み込みました", stage="skill", state="done", detail=f"{skill} ／ {' · '.join(sorted(files))}")
    with POOL.use(progress) as sb:
        final = GRAPH.invoke({"skill": files, "samples": samples, "criteria": criteria,
                              "sandbox": sb, "attempts": 0, "progress": progress})
    rep = final["report"]
    return {"code": final["code"], "tests": json.loads(final["tests"]), "report": rep,
            "attempts": final["attempts"], "all_passed": rep["failed"] == 0}


def run_tests(skill: str, code: str, tests: list) -> dict:
    files = skill_files(skill)
    with POOL.use(lambda *a, **k: None) as sb:
        res = sb.run({"custom_reward.py": code, "test_harness.py": files["test_harness.py"],
                      "tests.json": json.dumps(tests, ensure_ascii=False)},
                     "timeout 30 python3 test_harness.py custom_reward.py tests.json || true")
    return json.loads(res.get("stdout") or '{"passed":0,"failed":0,"total":0,"results":[]}')


TRIAL_SCRIPT = r'''
import importlib.util, json
spec = importlib.util.spec_from_file_location("r", "custom_reward.py")
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
import sys
rows = json.load(open("answers.json"))
for i, r in enumerate(rows, 1):
    try:
        r["score"] = float(m.compute_score("custom", r["response"], r["ground_truth"]))
    except Exception as e:
        r["score"] = None; r["error"] = f"{type(e).__name__}: {e}"
    print(f"[{i}/{len(rows)}] 正解 {r['ground_truth']!s:>10} → 点数 {r['score']}", file=sys.stderr)
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
        progress("Claude が問題を解いています", stage="llm", state="active", done=i, total=len(samples),
                 detail=f"{i + 1} / {len(samples)} 問目" + ("（画像も渡しています）" if s.get("images") else ""))
        text = f"{s['question']} {instruction}".strip()
        imgs = s.get("images") or []
        content = [*(_image_part(read_file(p), p) for p in imgs), {"type": "text", "text": text}] if imgs else text
        reply = llm([{"role": "user", "content": content}], max_tokens=2000)
        answers.append({"question": s["question"], "ground_truth": s["ground_truth"], "response": reply,
                        "n_images": len(imgs)})
    progress("Claude が問題を解きました", stage="llm", state="done", done=len(samples), total=len(samples),
             detail=f"{len(samples)} 問")
    with POOL.use(progress) as sb:
        res = sb.run({"custom_reward.py": code, "trial.py": TRIAL_SCRIPT,
                      "answers.json": json.dumps(answers, ensure_ascii=False)}, "timeout 30 python3 trial.py",
                     progress, label="Sandbox で採点しています")
    return json.loads(res.get("stdout") or "[]")


# ---- run edited code (editor's test button) ----------------------------------------------
RUN_SCRIPT = r'''
import importlib.util, json, sys, traceback
inp = json.load(open("input.json"))
out = {}
def where(tb):
    lines = [f.lineno for f in traceback.extract_tb(tb) if f.filename.endswith("custom_reward.py")]
    return lines[-1] if lines else None
try:
    src = open("custom_reward.py").read()
    compile(src, "custom_reward.py", "exec")
    out["syntax"] = {"ok": True}
    print("構文チェック: OK", file=sys.stderr)
except SyntaxError as e:
    out["syntax"] = {"ok": False, "line": e.lineno, "msg": f"{e.msg}"}
    print(f"構文エラー: {e.lineno} 行目: {e.msg}", file=sys.stderr)
    print(json.dumps(out, ensure_ascii=False)); sys.exit(0)
try:
    spec = importlib.util.spec_from_file_location("r", "custom_reward.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    fn = m.compute_score
except Exception as e:
    out["load"] = {"ok": False, "line": where(e.__traceback__), "msg": f"{type(e).__name__}: {e}"}
    print(f"読み込みエラー: {out['load']['msg']}", file=sys.stderr)
    print(json.dumps(out, ensure_ascii=False)); sys.exit(0)
def score(resp, gt):
    try:
        v = fn("custom", resp, gt, extra_info=None)
        if not isinstance(v, (int, float)) or not 0.0 <= float(v) <= 1.0:
            return None, f"0〜1 の数値を返していません（{v!r}）", None
        return float(v), None, None
    except Exception as e:
        return None, f"{type(e).__name__}: {e}", where(e.__traceback__)
out["answers"] = []
for i, a in enumerate(inp.get("answers") or [], 1):
    v, err, line = score(a["response"], a["ground_truth"])
    out["answers"].append({"score": v, "error": err, "line": line})
    print(f"確認テストの回答 [{i}] 正解 {a['ground_truth']!s:>10} → 点数 {v if err is None else 'エラー'}", file=sys.stderr)
out["tests"] = []
for c in inp.get("tests") or []:
    v, err, line = score(c["response"], c["ground_truth"])
    ok = err is None and abs(v - float(c["expected"])) < 1e-6
    out["tests"].append({"ok": ok, "got": v, "error": err, "line": line, "expected": c["expected"],
                         "ground_truth": c["ground_truth"], "response_tail": str(c["response"])[-120:], "why": c.get("why", "")})
if out["tests"]:
    print(f"AI が作ったテスト: {sum(t['ok'] for t in out['tests'])} / {len(out['tests'])} 件合格", file=sys.stderr)
case = inp.get("case")
if case:
    v, err, line = score(case["response"], case["ground_truth"])
    out["case"] = {"score": v, "error": err, "line": line}
    print(f"自分で試した1件 → 点数 {v if err is None else err}", file=sys.stderr)
print(json.dumps(out, ensure_ascii=False))
'''


def run_code(code: str, answers: list, tests: list, case: dict | None, progress: Callable[..., None]) -> dict:
    """Syntax-check the edited function, re-score the confirmation answers (no new LLM call),
    run the agent's tests and the user's own case — all in the warm sandbox."""
    inp = {"answers": [{"response": a.get("response", ""), "ground_truth": a.get("ground_truth", "")} for a in answers],
           "tests": tests or [], "case": case}
    with POOL.use(progress) as sb:
        res = sb.run({"custom_reward.py": code, "run.py": RUN_SCRIPT, "input.json": json.dumps(inp, ensure_ascii=False)},
                     "timeout 20 python3 run.py", progress, label="Sandbox で直したコードを実行しています")
    if res.get("exit_code") == 124:
        return {"timeout": True}
    return json.loads(res.get("stdout") or "{}")
