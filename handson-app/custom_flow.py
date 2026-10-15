"""「自分のデータで学習」: your own Q&A data -> verl parquet -> reward function -> GRPO.

Flow (see CUSTOM_DATA.md for the CLI version of the same steps):
  1. inspect   : read the head of a CSV / JSONL already on a UC Volume, list its columns
  2. preview   : show how 3 rows become verl prompts (<image> placeholders, ground truth)
  3. convert   : prep_custom.py as a serverless CPU job -> <VOL>/custom/<name>/{train,test}.parquet
  4. reward    : a preset, or the reward agent (UC skill + Claude + Sandbox), then a trial scoring
                 on real questions, then human approval -> <VOL>/custom/<name>/custom_reward.py
  5. train     : the custom GRPO YAML (1 or 2 nodes, text or image) with the approved function
  6. rollouts  : read <VOL>/rollouts/<experiment>/<step>.jsonl (answer, ground truth, score)
"""
import csv
import io
import json
import os
import re
import shutil
import tempfile
import threading
import time
import uuid

import requests
from databricks.sdk.core import Config
from fastapi import APIRouter, HTTPException, Request

import app as core
import reward_agent

router = APIRouter(prefix="/api/custom")
_cfg = Config()
_HOST = _cfg.host.rstrip("/")
SKILL = os.environ.get("AIR_SKILL", f"{core.CATALOG}.{core.SCHEMA}.grpo-reward-author")
CUSTOM_ROOT = f"{core.VOL}/custom"
ROLLOUT_ROOT = f"{core.VOL}/rollouts"
DEFAULT_INSTRUCTION = "Think step by step, then give the final answer in \\boxed{} (for example \\boxed{42})."
IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp")
TASKS: dict[str, dict] = {}   # background reward-agent / trial tasks (in memory)

TRAIN_TEMPLATES = {  # (has_images, nodes) -> repo template
    (False, 1): "grpo_gemma4_custom.yaml", (False, 2): "grpo_gemma4_custom_multinode.yaml",
    (True, 1): "grpo_gemma4_mm_custom.yaml", (True, 2): "grpo_gemma4_mm_custom_multinode.yaml",
}


# ---- Files API (UC Volumes) --------------------------------------------------------
def _h():
    return _cfg.authenticate()


def vol_read(path: str, max_bytes: int | None = None) -> bytes:
    h = dict(_h())
    if max_bytes:
        h["Range"] = f"bytes=0-{max_bytes - 1}"
    r = requests.get(f"{_HOST}/api/2.0/fs/files{path}", headers=h, timeout=120)
    if r.status_code not in (200, 206):
        raise HTTPException(400, detail=f"{path} を読めません（{r.status_code}）: {r.text[:200]}")
    return r.content


def vol_write(path: str, data: bytes):
    r = requests.put(f"{_HOST}/api/2.0/fs/files{path}?overwrite=true", headers=_h(), data=data, timeout=120)
    if r.status_code not in (200, 201, 204):
        raise HTTPException(502, detail=f"{path} に書き込めません（{r.status_code}）: {r.text[:200]}")


def vol_exists(path: str) -> bool:
    return requests.head(f"{_HOST}/api/2.0/fs/files{path}", headers=_h(), timeout=60).status_code == 200


def vol_ls(path: str) -> list[dict]:
    r = requests.get(f"{_HOST}/api/2.0/fs/directories{path.rstrip('/')}/", headers=_h(), timeout=60)
    return r.json().get("contents", []) if r.status_code == 200 else []


# ---- helpers ----------------------------------------------------------------------------
def _slug(s: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")
    return s[:40] or "dataset"


def _parse_head(path: str, max_rows: int = 200, stats: dict | None = None) -> tuple[list[str], list[dict]]:
    if not path.startswith("/Volumes/"):
        raise HTTPException(400, detail="パスは /Volumes/ から始めてください")
    raw = vol_read(path, max_bytes=2_000_000).decode("utf-8", errors="ignore")
    if stats is not None:  # whole file read? then the row count is exact
        stats["complete"] = len(raw) < 2_000_000
        stats["lines"] = sum(1 for l in raw.splitlines() if l.strip())
    if path.endswith((".jsonl", ".json")):
        rows = []
        for line in raw.splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    break  # last partial line of the 2 MB window
            if len(rows) >= max_rows:
                break
        cols = list(dict.fromkeys(k for r in rows for k in r))
    else:
        lines = raw.splitlines()
        if len(raw) >= 2_000_000 and lines:
            lines = lines[:-1]  # drop the partial last line
        reader = csv.DictReader(io.StringIO("\n".join(lines)))
        rows = [r for _, r in zip(range(max_rows), reader)]
        cols = reader.fieldnames or []
    return cols, rows


def _guess(cols, rows):
    def pick(names):
        return next((c for c in cols if c.lower() in names), None)
    q = pick({"question", "problem", "prompt", "input", "query", "質問", "問題"})
    a = pick({"answer", "ground_truth", "label", "target", "output", "正解", "答え"})
    img = next((c for c in cols if any(str(r.get(c, "")).lower().split("|")[0].endswith(IMG_EXT) for r in rows[:20])), None)
    return {"question_col": q, "answer_col": a, "image_col": img}


def _samples(meta: dict, n: int = 8, with_images: bool = False) -> list[dict]:
    _, rows = _parse_head(meta["input"], max_rows=max(n, 50))
    out = []
    for r in rows:
        q, a = str(r.get(meta["question_col"], "")).strip(), str(r.get(meta["answer_col"], "")).strip()
        if q and a:
            s = {"question": q, "ground_truth": a}
            if with_images and meta.get("image_col"):
                paths = [p for p in str(r.get(meta["image_col"]) or "").split("|") if p]
                s["images"] = [p if p.startswith("/") else f"{meta['image_dir'].rstrip('/')}/{p}" for p in paths]
            out.append(s)
        if len(out) >= n:
            break
    return out


def _meta(name: str) -> dict:
    return json.loads(vol_read(f"{CUSTOM_ROOT}/{name}/meta.json"))


def _task(fn, *args):
    tid = uuid.uuid4().hex[:12]
    TASKS[tid] = {"status": "running", "log": [], "result": None, "error": None, "started": time.time()}

    def progress(msg, **event):  # event: stage / state / detail / cmd / stdout ... (see reward_agent)
        TASKS[tid]["log"].append({"t": time.time(), "text": msg, **event})

    def run():
        try:
            TASKS[tid]["result"] = fn(*args, progress)
            TASKS[tid]["status"] = "done"
        except Exception as e:  # surfaced to the UI
            TASKS[tid]["status"], TASKS[tid]["error"] = "error", f"{type(e).__name__}: {e}"
    threading.Thread(target=run, daemon=True).start()
    return tid


# ---- 1-3: data -----------------------------------------------------------------------------
@router.post("/sample-raw")
def sample_raw(request: Request, body: dict | None = None):
    """Fabricate customer-style raw data (data.jsonl [+ images/]) under <VOL>/raw/<kind> to try the flow."""
    kind = (body or {}).get("kind", "image")
    spec = json.loads(core.render(open(os.path.join(core.TEMPLATES, "custom_sample_raw_job.json")).read()))
    spec["tasks"] = [t for t in spec["tasks"] if t["task_key"] == kind]
    if not spec["tasks"]:
        raise HTTPException(400, detail="kind は text か image を指定してください")
    for t in spec["tasks"]:
        t["spark_python_task"]["python_file"] = f"{core.EXPERIMENT_DIR}/make_sample_raw.py"
    spec["run_name"] = f"custom-sample-raw-{kind}"
    return _submit_jobs(spec, "custom_sample_raw", core.user_email(request)) | {"path": f"{core.VOL}/raw/{kind}/data.jsonl"}


@router.post("/inspect")
def inspect(body: dict):
    stats = {}
    cols, rows = _parse_head(body["path"], max_rows=100000, stats=stats)
    if not cols:
        raise HTTPException(400, detail="データが読めません。1行に1つの JSON（{\"question\": ..., \"answer\": ...}）が書かれた JSONL か確認してください")
    n = len(rows) if stats.get("complete") else None
    return {"columns": cols, "rows": rows[:5], "n_rows": n, "n_rows_at_least": len(rows), "guess": _guess(cols, rows)}


@router.post("/preview")
def preview(body: dict):
    cols, rows = _parse_head(body["path"], max_rows=20)
    qc, ac, ic = body["question_col"], body["answer_col"], body.get("image_col") or None
    instr = body.get("instruction") or DEFAULT_INSTRUCTION
    base = body.get("image_dir") or os.path.dirname(body["path"])
    out = []
    for r in rows[:3]:
        paths = [p for p in str(r.get(ic, "") if ic else "").split("|") if p]
        content = f"{str(r.get(qc, '')).strip()} {instr}".strip()
        missing = len(paths) - content.count("<image>")
        if missing > 0:
            content = "<image>" * missing + content
        full = [p if p.startswith("/") else f"{base.rstrip('/')}/{p}" for p in paths]
        out.append({"prompt": content, "ground_truth": str(r.get(ac, "")).strip(), "images": full,
                    "images_found": [vol_exists(p) for p in full]})
    return {"rows": out, "has_images": bool(ic)}


@router.post("/convert")
def convert(body: dict, request: Request):
    name = _slug(body.get("name") or os.path.splitext(os.path.basename(body["path"]))[0])
    out_dir = f"{CUSTOM_ROOT}/{name}"
    ic = body.get("image_col") or None
    meta = {"name": name, "input": body["path"], "question_col": body["question_col"], "answer_col": body["answer_col"],
            "image_col": ic, "image_dir": body.get("image_dir") or os.path.dirname(body["path"]),
            "instruction": body.get("instruction") or DEFAULT_INSTRUCTION, "data_source": f"custom/{name}",
            "has_images": bool(ic), "out": out_dir}
    args = ["--input", meta["input"], "--data-source", meta["data_source"], "--out", out_dir,
            "--question-col", meta["question_col"], "--answer-col", meta["answer_col"], "--instruction", meta["instruction"]]
    if ic:
        args += ["--image-col", ic, "--image-dir", meta["image_dir"]]
    spec = {"run_name": f"custom-prep-{name}",
            "tasks": [{"task_key": "convert", "environment_key": "cpu", "max_retries": 0,
                       "spark_python_task": {"python_file": f"{core.EXPERIMENT_DIR}/prep_custom.py", "parameters": args}}],
            "environments": [{"environment_key": "cpu", "spec": {"environment_version": "4", "dependencies": ["pyarrow>=19.0.0"]}}]}
    vol_write(f"{out_dir}/meta.json", json.dumps(meta, ensure_ascii=False, indent=1).encode())
    return _submit_jobs(spec, "custom_prep", core.user_email(request)) | {"name": name}


def _submit_jobs(spec: dict, kind: str, email: str):
    if email:
        spec["access_control_list"] = [{"user_name": email, "permission_level": "CAN_MANAGE"}]
    wd = tempfile.mkdtemp()
    path = os.path.join(wd, "job.json")
    json.dump(spec, open(path, "w"))
    try:
        rc, out, err = core.run_cli(["jobs", "submit", "--json", f"@{path}", "--no-wait", "-o", "json"], timeout=180)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    m = re.search(r'"run_id":\s*(\d+)', out)
    if rc != 0 or not m:
        raise HTTPException(502, detail=(err or out).strip()[-2000:])
    core.SUBMITTED[m.group(1)] = {"kind": kind, "submitted_by": email, "submitted_at": time.time()}
    return {"run_id": m.group(1)}


@router.get("/datasets")
def datasets():
    out = []
    for e in vol_ls(CUSTOM_ROOT):
        if not e.get("is_directory"):
            continue
        name = e["path"].rstrip("/").split("/")[-1]
        try:
            meta = _meta(name)
        except HTTPException:
            continue
        files = {x["path"].split("/")[-1] for x in vol_ls(e["path"])}
        out.append({**meta, "converted": "train.parquet" in files, "reward_approved": "custom_reward.py" in files})
    return out


# ---- 4: reward ---------------------------------------------------------------------------------
PRESETS = {
    "numeric": "数値で比べる（標準）",
    "exact": "文字が完全に同じなら正解",
}
EXACT_CODE = '''"""Exact-match reward: the last \\\\boxed{...} must equal ground_truth (case/whitespace-insensitive)."""


def _last_boxed(text):
    start = text.rfind("\\\\boxed{")
    if start < 0:
        return None
    i, depth, out = start + len("\\\\boxed{"), 1, []
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


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float:
    try:
        pred = _last_boxed(solution_str or "")
        if pred is None:
            return 0.0
        return 1.0 if pred.strip().lower() == str(ground_truth).strip().lower() else 0.0
    except Exception:
        return 0.0
'''


@router.get("/reward/presets")
def presets():
    return [{"id": k, "label": v} for k, v in PRESETS.items()]


@router.get("/reward/preset/{pid}")
def preset_code(pid: str):
    if pid in ("numeric", "standard"):
        return {"code": open(os.path.join(core.TEMPLATES, "custom_reward.py")).read()}
    if pid == "exact":
        return {"code": EXACT_CODE}
    raise HTTPException(404, detail="unknown preset")


@router.get("/{name}/samples")
def samples(name: str, n: int = 8):
    return _samples(_meta(name), n)


@router.post("/{name}/reward/agent")
def start_agent(name: str, body: dict):
    s = _samples(_meta(name), 8)
    return {"task_id": _task(reward_agent.write_reward, SKILL, s, body.get("criteria", ""))}


@router.post("/{name}/reward/test")
def test_reward(name: str, body: dict):
    return reward_agent.run_tests(SKILL, body["code"], body["tests"])


@router.post("/{name}/reward/trial")
def start_trial(name: str, body: dict):
    meta = _meta(name)
    s = _samples(meta, int(body.get("n", 5)), with_images=True)
    return {"task_id": _task(reward_agent.trial_score, body["code"], s, meta["instruction"], vol_read)}


@router.post("/{name}/reward/run")
def start_run(name: str, body: dict):
    """Editor's test button: run the edited code in the warm Sandbox."""
    case = body.get("case")
    if case and not (str(case.get("response", "")).strip() or str(case.get("ground_truth", "")).strip()):
        case = None
    return {"task_id": _task(reward_agent.run_code, body["code"], body.get("answers") or [], body.get("tests") or [], case)}


@router.get("/sandbox")
def sandbox_status():
    return reward_agent.POOL.status()


@router.delete("/sandbox")
def sandbox_delete():
    reward_agent.POOL.delete()
    return reward_agent.POOL.status()


@router.get("/task/{tid}")
def task(tid: str):
    if tid not in TASKS:
        raise HTTPException(404, detail="task not found (the app may have restarted)")
    return TASKS[tid]


@router.post("/{name}/reward/approve")
def approve(name: str, body: dict, request: Request):
    code = body["code"]
    if "def compute_score" not in code:
        raise HTTPException(400, detail="compute_score が定義されていません")
    vol_write(f"{CUSTOM_ROOT}/{name}/custom_reward.py", code.encode())
    vol_write(f"{CUSTOM_ROOT}/{name}/custom_reward.approved.json",
              json.dumps({"by": core.user_email(request), "at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}).encode())
    return {"ok": True}


@router.get("/{name}/reward")
def get_reward(name: str):
    p = f"{CUSTOM_ROOT}/{name}/custom_reward.py"
    return {"code": vol_read(p).decode() if vol_exists(p) else None}


# ---- 5: train -------------------------------------------------------------------------------------
@router.post("/{name}/train")
def train(name: str, body: dict, request: Request):
    meta = _meta(name)
    nodes = int(body.get("nodes", 1))
    rp = f"{CUSTOM_ROOT}/{name}/custom_reward.py"
    if not vol_exists(f"{meta['out']}/train.parquet"):
        raise HTTPException(400, detail="変換がまだ終わっていません")
    if not vol_exists(rp):
        raise HTTPException(400, detail="採点関数がまだ承認されていません")
    tmpl = TRAIN_TEMPLATES[(meta["has_images"], nodes)]
    exp = f"custom-{name}-{'image' if meta['has_images'] else 'text'}-{nodes}node"
    sub = "image" if meta["has_images"] else "text"
    y = core.render(open(os.path.join(core.TEMPLATES, tmpl)).read())
    y = y.replace(f"{core.VOL}/custom/{sub}/", f"{meta['out']}/")
    y = re.sub(r"(?m)^experiment_name: .*$", f"experiment_name: {exp}", y)
    y = re.sub(r"(?m)^mlflow_experiment_directory: .*$", f"mlflow_experiment_directory: {core.EXPERIMENT_DIR}/{exp}", y)
    roll = f"{exp}-{time.strftime('%Y%m%d-%H%M%S', time.gmtime())}"  # one folder per run: never show an older run's answers
    y = re.sub(r"trainer\.rollout_data_dir=\S+", f"trainer.rollout_data_dir={ROLLOUT_ROOT}/{roll}", y)
    y = re.sub(r"(?m)^(\s*output_dir: ).*$", rf"\g<1>{core.VOL}/ckpt/{exp}", y)
    y = core.with_uc_model(y)
    email = core.user_email(request)
    if email:
        y += f"\npermissions:\n  - user_name: {email}\n    level: CAN_MANAGE\n"
    wd = tempfile.mkdtemp(prefix="custom-train-")
    try:
        for s in core.SCRIPTS:
            open(os.path.join(wd, s), "w").write(core.render(open(os.path.join(core.TEMPLATES, s)).read()))
        open(os.path.join(wd, "custom_reward.py"), "wb").write(vol_read(rp))
        open(os.path.join(wd, tmpl), "w").write(y)
        rc, out, err = core.run_cli(["air", "run", "-f", tmpl, "-o", "json"], timeout=600, cwd=wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    m = re.search(r'"run_id":\s*"(\d+)"', out)
    if rc != 0 or not m:
        raise HTTPException(502, detail=(err or out).strip()[-2000:])
    core.SUBMITTED[m.group(1)] = {"kind": "custom_train", "label": f"独自データ GRPO: {name}（{sub}・{nodes}ノード）",
                                  "submitted_by": email, "submitted_at": time.time(), "experiment": roll}
    return {"run_id": m.group(1), "experiment": roll}


# ---- 6: rollouts -----------------------------------------------------------------------------------
@router.get("/rollouts/{experiment}")
def rollouts(experiment: str, step: int | None = None, limit: int = 60):
    if not re.search(r"-\d{8}-\d{6}$", experiment):  # bare experiment name (app restarted): its latest run
        runs = sorted(n for n in (e["path"].rstrip("/").split("/")[-1] for e in vol_ls(ROLLOUT_ROOT) if e.get("is_directory"))
                      if re.fullmatch(re.escape(experiment) + r"-\d{8}-\d{6}", n))
        experiment = runs[-1] if runs else experiment
    files = sorted((e["path"] for e in vol_ls(f"{ROLLOUT_ROOT}/{experiment}") if e["path"].endswith(".jsonl")),
                   key=lambda p: int(re.sub(r"\D", "", p.split("/")[-1]) or 0))
    if not files:
        return {"steps": [], "rows": []}
    steps = [int(re.sub(r"\D", "", p.split("/")[-1])) for p in files]
    pick = files[steps.index(step)] if step in steps else files[-1]
    rows = [json.loads(l) for l in vol_read(pick).decode().splitlines() if l.strip()]
    out = [{"score": r.get("score"), "ground_truth": r.get("gts"), "answer_tail": (r.get("output") or "")[-400:],
            "answer_len": len(r.get("output") or ""), "question": (r.get("input") or "")[-300:]} for r in rows[:limit]]
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    return {"steps": steps, "step": int(re.sub(r"\D", "", pick.split("/")[-1])), "n": len(rows),
            "mean": sum(scores) / len(scores) if scores else None, "rows": out}
