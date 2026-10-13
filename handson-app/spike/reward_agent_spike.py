"""Spike: can a Databricks App (as its SP) run an agent that writes a GRPO reward function?

  - reads a Unity Catalog skill (SKILL.md + files) via the Files API
  - asks Claude (Foundation Model API) to write custom_reward.py + tests, following the skill
  - runs the tests in a Databricks Sandbox; on failure, feeds the report back (LangGraph loop)
"""
import base64
import json
import os
import re
import time
from typing import TypedDict

import requests
from databricks.sdk.core import Config
from fastapi import FastAPI
from langgraph.graph import END, StateGraph
from openai import OpenAI

cfg = Config()
HOST = cfg.host.rstrip("/")
SKILL = os.environ["SKILL_NAME"]                      # catalog.schema.skill
LLM = os.environ.get("LLM_ENDPOINT", "databricks-claude-sonnet-5-5")
HARNESS = open(os.path.join(os.path.dirname(__file__), "test_harness.py")).read()
app = FastAPI()


def headers():
    return cfg.authenticate()


def api(method, path, **kw):
    r = requests.request(method, f"{HOST}{path}", headers=headers(), timeout=kw.pop("timeout", 120), **kw)
    return r.status_code, (r.json() if "json" in r.headers.get("content-type", "") else r.text)


# ---- UC skill -----------------------------------------------------------------
def skill_files() -> dict:
    cat, sch, name = SKILL.split(".")
    base = f"/api/2.0/fs/directories/Skills/{cat}/{sch}/{name}/"
    code, listing = api("GET", base)
    if code != 200:
        raise RuntimeError(f"list skill files: {code} {listing}")
    out = {}
    for e in listing.get("contents", []):
        rel = e["path"].split(f"/{name}/", 1)[1]
        r = requests.get(f"{HOST}/api/2.0/fs/files{e['path']}", headers=headers(), timeout=60)
        out[rel] = r.text if r.status_code == 200 else f"<error {r.status_code}>"
    return out


# ---- Sandbox --------------------------------------------------------------------
def sandbox_create() -> str:
    sid = f"reward-agent-{int(time.time())}"
    code, body = api("POST", f"/api/2.0/sandboxes?sandbox_id={sid}",
                     json={"display_name": "reward agent", "spec": {"compute": {"inactivity_timeout": "900s"}}})
    if code != 200:
        raise RuntimeError(f"create sandbox: {code} {body}")
    for _ in range(60):
        _, s = api("GET", f"/api/2.0/sandboxes/{sid}")
        if s.get("status", {}).get("state") == "SANDBOX_STATE_RUNNING":
            return sid
        time.sleep(3)
    raise RuntimeError(f"sandbox not running: {s}")


def sandbox_exec(sid: str, files: dict, command: str) -> dict:
    script = "set -e; mkdir -p /home/sandbox-agent/job && cd /home/sandbox-agent/job\n"
    for name, body in files.items():
        script += f"echo {base64.b64encode(body.encode()).decode()} | base64 -d > {name}\n"
    script += command
    code, body = api("POST", f"/api/2.0/sandbox-exec/sandboxes/{sid}/exec-sync",
                     json={"cmd": "/bin/bash", "args": ["-c", script]}, timeout=300)
    if code != 200:
        raise RuntimeError(f"exec: {code} {body}")
    return body


def sandbox_delete(sid: str):
    api("DELETE", f"/api/2.0/sandboxes/{sid}")


# ---- LLM ----------------------------------------------------------------------------
def llm(messages) -> str:
    token = headers()["Authorization"].split(" ", 1)[1]
    client = OpenAI(api_key=token, base_url=f"{HOST}/serving-endpoints")
    r = client.chat.completions.create(model=LLM, messages=messages, max_tokens=16000)
    print(f"[llm] finish_reason={r.choices[0].finish_reason}", flush=True)
    content = r.choices[0].message.content
    if isinstance(content, list):  # Claude may return content parts (e.g. reasoning + text)
        content = "".join(p.get("text", "") if isinstance(p, dict) else str(p) for p in content
                          if not isinstance(p, dict) or p.get("type") in (None, "text"))
    return content or ""


def extract_block(text: str, lang: str):
    m = re.findall(rf"```{lang}\s*\n(.*?)```", text, re.S | re.I)
    return m[-1] if m else None


# ---- LangGraph: author -> test -> (fix) ---------------------------------------------
class S(TypedDict, total=False):
    skill: dict
    samples: list
    criteria: str
    code: str
    tests: str
    report: dict
    attempts: int
    sandbox: str
    log: list


def _valid_tests(text: str) -> bool:
    try:
        t = json.loads(text)
        return isinstance(t, list) and all({"response", "ground_truth", "expected"} <= set(x) for x in t)
    except Exception:
        return False


def author(state: S) -> S:
    skill = state["skill"]
    sys_prompt = ("You follow this skill exactly.\n\n# SKILL.md\n" + skill.get("SKILL.md", "") +
                  "\n\n# pitfalls.md\n" + skill.get("pitfalls.md", "") +
                  "\n\n# reward_template.py\n```python\n" + skill.get("reward_template.py", "") + "\n```")
    user = (f"Sample rows (question, ground_truth):\n{json.dumps(state['samples'], ensure_ascii=False, indent=1)}\n\n"
            f"User's grading criteria: {state['criteria']}\n\n"
            "Return exactly two fenced blocks: ```python (the complete custom_reward.py) and ```json (the test "
            "cases list as described in the skill).")
    msgs = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user}]
    if state.get("report"):
        failed = [r for r in state["report"]["results"] if not r["ok"]]
        msgs += [{"role": "assistant", "content": f"```python\n{state['code']}```\n```json\n{state['tests']}```"},
                 {"role": "user", "content": "These tests failed in the sandbox; fix the function (only change a test "
                  f"if it contradicts the criteria) and return both blocks again:\n{json.dumps(failed, ensure_ascii=False)}"}]
    out = llm(msgs)
    code, tests = extract_block(out, "python"), extract_block(out, "json")
    print(f"[author] got python={code is not None} json={tests is not None} len={len(out)} tail={out[-300:]!r}", flush=True)
    if code is None:
        raise ValueError("model did not return a ```python block")
    if tests is None or not _valid_tests(tests):  # ask once more for just the test cases
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": "Now return ONLY the test cases as one ```json block: a list of "
                  '{"response", "ground_truth", "expected", "why"} objects, as described in the skill.'}]
        tests = extract_block(llm(msgs), "json")
        if tests is None or not _valid_tests(tests):
            raise ValueError("model did not return valid ```json test cases")
    n = state.get("attempts", 0) + 1
    return {"code": code, "tests": tests, "attempts": n, "log": state.get("log", []) + [f"author attempt {n}"]}


def test(state: S) -> S:
    res = sandbox_exec(state["sandbox"], {"custom_reward.py": state["code"], "test_harness.py": HARNESS,
                                          "tests.json": state["tests"]},
                       "python3 test_harness.py custom_reward.py tests.json")
    report = json.loads(res["stdout"]) if res.get("stdout", "").strip().startswith("{") else \
        {"passed": 0, "failed": 1, "total": 1, "results": [{"ok": False, "error": res.get("stderr", "")[-2000:]}]}
    return {"report": report, "log": state["log"] + [f"sandbox: {report['passed']}/{report['total']} passed"]}


def route(state: S):
    return END if state["report"]["failed"] == 0 or state["attempts"] >= 3 else "author"


graph = StateGraph(S)
graph.add_node("author", author)
graph.add_node("test", test)
graph.set_entry_point("author")
graph.add_edge("author", "test")
graph.add_conditional_edges("test", route)
AGENT = graph.compile()


# ---- spike endpoints ------------------------------------------------------------
@app.get("/spike/skill")
def get_skill():
    files = skill_files()
    return {k: len(v) for k, v in files.items()} | {"SKILL.md_head": files.get("SKILL.md", "")[:200]}


@app.get("/spike/sandbox")
def try_sandbox():
    t0 = time.time()
    sid = sandbox_create()
    try:
        res = sandbox_exec(sid, {}, "python3 --version; id -un")
    finally:
        sandbox_delete(sid)
    return {"seconds": round(time.time() - t0, 1), "stdout": res.get("stdout"), "exit_code": res.get("exit_code")}


@app.get("/spike/llm")
def try_llm():
    return {"reply": llm([{"role": "user", "content": "Reply with exactly: OK"}])}


@app.post("/spike/agent")
def run_agent(body: dict):
    t0 = time.time()
    sid = sandbox_create()
    try:
        final = AGENT.invoke({"skill": skill_files(), "samples": body["samples"], "criteria": body["criteria"],
                              "sandbox": sid, "attempts": 0, "log": []})
    finally:
        sandbox_delete(sid)
    return {"seconds": round(time.time() - t0, 1), "log": final["log"], "attempts": final["attempts"],
            "passed": final["report"]["passed"], "total": final["report"]["total"],
            "failed_cases": [r for r in final["report"]["results"] if not r["ok"]],
            "code": final["code"], "tests": json.loads(final["tests"])}
