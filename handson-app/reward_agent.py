"""Reward-function agent: UC skill + Claude Code in a Databricks Sandbox.

write_reward: the UC skill is copied to ~/.claude/skills/ in a sandbox, and Claude Code
(preinstalled in the sandbox, talking to the workspace's AI Gateway) writes
custom_reward.py + tests.json and runs the skill's test harness itself. Its steps are
streamed to the UI. The result is then re-tested in the token-less sandbox.

Also: `trial_score` — Claude (Foundation Model API) answers a few real questions from the
dataset, and the reward function scores those answers in the Sandbox, so a human can judge
whether the scores make sense before approving the function.

Runs as the app's service principal (needs READ VOLUME on the skill's schema, CAN_QUERY on
the LLM endpoint, and access to the AI Gateway models; the Sandbox API needs no extra grant).

Two kinds of sandbox:
- the Claude Code sandbox carries the app's Databricks token (Claude Code needs it for the
  AI Gateway); it is created per request and deleted as soon as Claude Code is done.
- everything else (tests, confirmation test, the editor's test button) runs in a sandbox
  created WITHOUT a token, so code run there cannot act as the app. That one is reused and
  deleted after SANDBOX_IDLE_SECONDS unused.

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
from typing import Callable

import requests
from databricks.sdk.core import Config
from openai import OpenAI

cfg = Config()
HOST = cfg.host.rstrip("/")
LLM = os.environ.get("AIR_LLM_ENDPOINT", "databricks-claude-sonnet-5-5")
CODING_MODEL = os.environ.get("AIR_CODING_MODEL", "system.ai.claude-sonnet-5-5")   # AI Gateway model for Claude Code
CLAUDE_TIMEOUT = 600
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
    def __init__(self, with_token: bool = False):
        self.with_token = with_token
        self.id = f"reward-{'coder' if with_token else 'agent'}-{int(time.time() * 1000)}"
        self.created = None
        self.uses = 0

    def start(self, progress):
        t0 = time.time()
        stage, tok = ("cc-create", "アプリの SP のトークン入り（作業後すぐ削除）") if self.with_token else ("sb-create", "Databricks のトークンなし")
        progress("Sandbox を作成しています", stage=stage, state="active", detail=f"ID: {self.id}")
        code, body = _api("POST", f"/api/2.0/sandboxes?sandbox_id={self.id}",
                          json={"display_name": "reward coder" if self.with_token else "reward agent",
                                "spec": {"compute": {"inactivity_timeout": "900s"},
                                         "identity": {"databricks_access_token_included": self.with_token}}})
        if code != 200:
            progress("Sandbox を作成できませんでした", stage=stage, state="error", detail=f"{code} {str(body)[:200]}")
            raise RuntimeError(f"cannot create sandbox: {code} {str(body)[:300]}")
        seen = []
        for _ in range(200):
            _, st = _api("GET", f"/api/2.0/sandboxes/{self.id}")
            state = (st.get("status", {}).get("state") or "?").replace("SANDBOX_STATE_", "") if isinstance(st, dict) else "?"
            if not seen or seen[-1] != state:
                seen.append(state)
                progress("Sandbox を作成しています", stage=stage, state="active",
                         detail=f"ID: {self.id} ／ 状態: {' → '.join(seen)}（{time.time() - t0:.1f} 秒）")
            if state == "RUNNING":
                self.created = time.time()
                progress("Sandbox を作成しました", stage=stage, state="done",
                         detail=f"ID: {self.id} ／ {' → '.join(seen)}（{self.created - t0:.1f} 秒）／ {tok}")
                return self
            time.sleep(0.5)
        progress("Sandbox が起動しませんでした", stage=stage, state="error", detail=self.id)
        raise RuntimeError("sandbox did not start")

    def alive(self) -> bool:
        code, st = _api("GET", f"/api/2.0/sandboxes/{self.id}")
        return code == 200 and isinstance(st, dict) and st.get("status", {}).get("state") == "SANDBOX_STATE_RUNNING"

    def exec(self, script: str, timeout: int = 300) -> dict:
        code, body = _api("POST", f"/api/2.0/sandbox-exec/sandboxes/{self.id}/exec-sync",
                          json={"cmd": "/bin/bash", "args": ["-c", script]}, timeout=timeout)
        if code != 200 or not isinstance(body, dict):
            raise RuntimeError(f"sandbox exec failed: {code} {str(body)[:300]}")
        return body

    def run(self, files: dict, command: str, progress=None, stage="sb-exec", label="Sandbox で実行しています") -> dict:
        say = progress or (lambda *a, **k: None)
        sent = [{"name": k, "bytes": len(v.encode())} for k, v in files.items()]
        say(label, stage=stage, state="active", cmd=command, files=sent, sandbox=self.id)
        script = "set -e; rm -rf /home/sandbox-agent/job; mkdir -p /home/sandbox-agent/job; cd /home/sandbox-agent/job\n"
        for name, body in files.items():
            script += f"mkdir -p \"$(dirname '{name}')\"; echo {base64.b64encode(body.encode()).decode()} | base64 -d > '{name}'\n"
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

    def delete(self) -> bool:
        """Delete, then confirm the sandbox is gone (GET returns 404)."""
        _api("DELETE", f"/api/2.0/sandboxes/{self.id}")
        for _ in range(20):
            if _api("GET", f"/api/2.0/sandboxes/{self.id}")[0] == 404:
                return True
            time.sleep(0.5)
        return False


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


# ---- Claude Code in a sandbox ------------------------------------------------------------
CLAUDE = "/usr/local/libexec/claude"   # the real binary; /usr/local/bin/claude is the ucode wrapper
# The sandbox pre-configures Claude Code for the AI Gateway, but its default model
# (databricks-claude-opus-4-6) is rejected with 403 there and Claude Code then waits
# silently, so the model is always given explicitly.
CC_PROMPT = (
    "Use the {skill} skill. Sample rows of the user's dataset (question, ground_truth) are in samples.json. "
    "User's grading criteria (may be Japanese): {criteria} "
    "Write custom_reward.py and tests.json in the current directory, run the skill's test_harness.py on them "
    "(python3 ~/.claude/skills/{skill}/test_harness.py custom_reward.py tests.json), fix until all tests pass, "
    "then reply in Japanese with a short summary (how it grades, assumptions). Do not use the network.")


def _cc_event(ev: dict, progress, pending: dict, counter: list):
    """Turn one stream-json line from Claude Code into UI events."""
    t = ev.get("type")
    if t == "system" and ev.get("subtype") == "init":
        sk = [x for x in (ev.get("skills") or []) if "reward" in x]
        progress("Claude Code が起動しました", stage="cc-init", state="done",
                 detail=f"モデル {ev.get('model')} ／ スキル {', '.join(sk) or '（なし）'} を認識 ／ ツール {len(ev.get('tools') or [])} 個")
    elif t == "assistant":
        for c in ev["message"].get("content", []):
            if c.get("type") == "tool_use":
                counter[0] += 1
                key, name, inp = f"cc-{counter[0]}", c["name"], c.get("input") or {}
                if name == "Skill":
                    label, detail, cmd = f"スキル「{inp.get('skill')}」を読み込んでいます", "UC から渡したスキルの手順に沿って進めます", None
                elif name == "Bash":
                    cmd = inp.get("command", "")
                    label, detail = "コマンドを実行しています", inp.get("description") or cmd.splitlines()[0][:120]
                elif name in ("Write", "Edit", "MultiEdit"):
                    f = os.path.basename(inp.get("file_path", ""))
                    n = len((inp.get("content") or inp.get("new_string") or "").splitlines())
                    label, detail, cmd = f"{f} を{'書いています' if name == 'Write' else '直しています'}", f"{n} 行", None
                elif name == "Read":
                    label, detail, cmd = f"{os.path.basename(inp.get('file_path', ''))} を読んでいます", "", None
                else:
                    label, detail, cmd = f"{name} を使っています", json.dumps(inp, ensure_ascii=False)[:120], None
                pending[c["id"]] = (key, label, detail, cmd, time.time())
                progress(label, stage=key, state="active", detail=detail, tool=name)
            elif c.get("type") == "text" and c.get("text", "").strip():
                progress("Claude Code のメッセージ", stage=f"cc-msg-{counter[0]}", state="done", detail=c["text"].strip()[:300])
    elif t == "user":
        for c in (ev.get("message") or {}).get("content", []) if isinstance((ev.get("message") or {}).get("content"), list) else []:
            if c.get("type") == "tool_result" and c.get("tool_use_id") in pending:
                key, label, detail, cmd, t0 = pending.pop(c["tool_use_id"])
                body = c.get("content")
                text = "".join(x.get("text", "") for x in body if isinstance(x, dict)) if isinstance(body, list) else str(body or "")
                done = label.replace("しています", "しました").replace("でいます", "ました").replace("ています", "ました")
                extra = {"cmd": cmd, "stdout": _tail(text, 10), "exit_code": 1 if c.get("is_error") else 0,
                         "ms": int((time.time() - t0) * 1000), "files": [], "tool": "Bash"} if cmd else {}
                progress(done, stage=key, state="error" if c.get("is_error") else "done", detail=detail, **extra)
    elif t == "result":
        progress("Claude Code が作業を終えました", stage="cc-result", state="done" if ev.get("subtype") == "success" else "error",
                 detail=f"{ev.get('num_turns')} ターン ／ {ev.get('duration_ms', 0) / 1000:.0f} 秒 ／ 約 ${ev.get('total_cost_usd') or 0:.2f}")


def write_reward(skill: str, samples: list, criteria: str, progress: Callable[..., None]) -> dict:
    progress("UC のスキルを読み込んでいます", stage="skill", state="active", detail=skill)
    files = skill_files(skill)
    progress("UC のスキルを読み込みました", stage="skill", state="done", detail=f"{skill} ／ {' · '.join(sorted(files))}")
    name = skill.split(".")[-1]
    sb = Sandbox(with_token=True).start(progress)
    try:
        home = "/home/sandbox-agent"
        setup = f"set -e; export HOME={home}; rm -rf ~/job; mkdir -p ~/.claude/skills/{name} ~/job\n"
        for rel, body in files.items():
            setup += f"echo {base64.b64encode(body.encode()).decode()} | base64 -d > ~/.claude/skills/{name}/{rel}\n"
        setup += f"echo {base64.b64encode(json.dumps(samples, ensure_ascii=False).encode()).decode()} | base64 -d > ~/job/samples.json\n"
        prompt = CC_PROMPT.format(skill=name, criteria=criteria or "(none given; use the skill defaults)")
        run_sh = (f"cd {home}/job\nexport HOME={home} ANTHROPIC_MODEL={CODING_MODEL} ANTHROPIC_SMALL_FAST_MODEL={CODING_MODEL}\n"
                  f"timeout {CLAUDE_TIMEOUT} {CLAUDE} -p {json.dumps(prompt)} --model {CODING_MODEL} --output-format stream-json "
                  f"--verbose --dangerously-skip-permissions --max-turns 30 < /dev/null > stream.jsonl 2> err.txt\necho $? > done\n")
        setup += f"echo {base64.b64encode(run_sh.encode()).decode()} | base64 -d > ~/job/run.sh\n"
        setup += "cd ~/job; setsid nohup bash run.sh > /dev/null 2>&1 < /dev/null &\necho started\n"
        sb.exec(setup, timeout=60)
        progress("Claude Code を起動しています", stage="cc-init", state="active",
                 detail=f"{CLAUDE} -p … --model {CODING_MODEL}（スキルを ~/.claude/skills/{name}/ に置きました）")
        seen, pending, counter, t0 = 0, {}, [0], time.time()
        while True:
            r = sb.exec(f"cd {home}/job; tail -n +{seen + 1} stream.jsonl 2>/dev/null; echo '@@DONE:'$(cat done 2>/dev/null)", timeout=60)
            out = r.get("stdout", "")
            body, _, flag = out.rpartition("@@DONE:")
            lines = body.split("\n")
            if not flag.strip():
                lines = lines[:-1]          # the last line may still be being written: read it next time
            for line in lines:
                if not line.strip():
                    continue
                try:
                    ev = json.loads(line)
                except json.JSONDecodeError:
                    break                   # partial line: re-read from here
                seen += 1
                _cc_event(ev, progress, pending, counter)
            if flag.strip():
                break
            if time.time() - t0 > CLAUDE_TIMEOUT + 30:
                raise RuntimeError("Claude Code が時間内に終わりませんでした")
            time.sleep(1.5)
        got = sb.exec(f"cd {home}/job; python3 -c \"import json,os;print(json.dumps({{f: open(f).read() for f in ['custom_reward.py','tests.json','err.txt'] if os.path.exists(f)}}))\"")
        res = json.loads(got.get("stdout") or "{}")
    finally:
        gone = sb.delete()
        progress("Claude Code 用の Sandbox を削除しました" if gone else "Claude Code 用の Sandbox の削除を確認できませんでした",
                 stage="cc-delete", state="done" if gone else "error",
                 detail=f"ID: {sb.id} ／ " + ("削除後に参照して、なくなったことを確認しました" if gone else "ワークスペースで確認してください"))
    if "custom_reward.py" not in res or "tests.json" not in res:
        raise RuntimeError("Claude Code が custom_reward.py / tests.json を作りませんでした。" + (res.get("err.txt") or "")[-500:])
    tests = json.loads(res["tests.json"])
    # Do not trust the agent's own run: test again in the token-less sandbox.
    with POOL.use(progress) as tsb:
        out = tsb.run({"custom_reward.py": res["custom_reward.py"], "test_harness.py": files["test_harness.py"],
                       "tests.json": res["tests.json"]}, "timeout 30 python3 test_harness.py custom_reward.py tests.json || true",
                      progress, stage="sb-verify", label="トークンなしの Sandbox でテストし直しています")
    report = json.loads(out.get("stdout") or '{"passed":0,"failed":1,"total":1,"results":[]}')
    progress(f"テスト結果: {report['passed']} / {report['total']} 件合格", stage="verify-result",
             state="done" if report["failed"] == 0 else "error",
             detail="Claude Code の報告ではなく、アプリが改めて実行した結果です")
    return {"code": res["custom_reward.py"], "tests": tests, "report": report, "attempts": counter[0],
            "all_passed": report["failed"] == 0}


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
