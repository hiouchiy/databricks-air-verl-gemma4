"""Hands-on UI for the verl + Gemma4 GRPO sample on Databricks AI Runtime.

The app drives the same `databricks air` CLI the hands-on uses, running as the app's
service principal. Databricks Apps on-behalf-of-user scopes do not cover Jobs / MLflow /
AI Runtime APIs, so jobs are submitted as the SP and the logged-in user is granted
CAN_MANAGE on each run (YAML `permissions`).

Templates (*.yaml / run_grpo*.sh) are copied from the repo root into ./templates by
deploy.sh, and rendered here exactly like quickstart.sh does (__IMAGE__ / __VOL__ /
__WS_EMAIL__), except that MLflow experiments go to /Workspace/Shared/air-handson so
everyone can open them. Data prep runs as a serverless CPU job (no GPU needed).
"""
import io
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import urllib.request
import zipfile

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(HERE, "templates")

CLI_VERSION = os.environ.get("AIR_CLI_VERSION", "1.19.0")
CLI_URL = (f"https://github.com/databricks/cli/releases/download/v{CLI_VERSION}/"
           f"databricks_cli_{CLI_VERSION}_linux_amd64.zip")
CLI_DIR = "/tmp/dbcli"
CLI = os.path.join(CLI_DIR, "databricks")

CATALOG = os.environ["AIR_CATALOG"]
SCHEMA = os.environ["AIR_SCHEMA"]
VOLUME = os.environ.get("AIR_VOLUME", "air_handson")
IMAGE_NAME = os.environ.get("AIR_IMAGE_NAME", "verl-gemma4")
IMAGE_TAG = os.environ.get("AIR_IMAGE_TAG", "v1")
EXPERIMENT_DIR = os.environ.get("AIR_EXPERIMENT_DIR", "/Workspace/Shared/air-handson")

UC_IMAGE = f"{CATALOG}.{SCHEMA}.{IMAGE_NAME}:{IMAGE_TAG}"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}"

# Each button maps to one repo template. nodes = GPU nodes (8xH100) the run occupies.
JOBS = {
    # Data prep needs no GPU: a serverless CPU job (Jobs API), so it works even when GPUs are unavailable.
    "prep_data":      {"file": "prep_data_job.json",            "label": "データ準備: gsm8k + geo3k（CPU）", "nodes": 0, "engine": "jobs"},
    "smoke":          {"file": "smoke_test.yaml",               "label": "疎通確認 (1xA10)",              "nodes": 0},
    "grpo_text_1":    {"file": "grpo_gemma4.yaml",              "label": "GRPO テキスト 単一ノード",        "nodes": 1},
    "grpo_text_2":    {"file": "grpo_gemma4_multinode.yaml",    "label": "GRPO テキスト 2ノード",           "nodes": 2},
    "grpo_mm_1":      {"file": "grpo_gemma4_mm.yaml",           "label": "GRPO 画像 単一ノード",            "nodes": 1},
    "grpo_mm_2":      {"file": "grpo_gemma4_mm_multinode.yaml", "label": "GRPO 画像 2ノード",               "nodes": 2},
}
SCRIPTS = ["run_grpo.sh", "run_grpo_multinode.sh"]
PREP_SCRIPT = f"{EXPERIMENT_DIR}/prep_data.py"   # uploaded by deploy.sh
PREP_RUN_PREFIX = "prep-data"
TERMINAL = {"SUCCESS", "FAILED", "CANCELED", "CANCELLED", "TIMEDOUT", "TIMED_OUT", "INTERNAL_ERROR", "SKIPPED"}

# run_id -> {"kind", "submitted_by", "submitted_at"}; in-memory, lost on app restart.
SUBMITTED: dict[str, dict] = {}

app = FastAPI()


# ---- CLI helpers -------------------------------------------------------------
def ensure_cli() -> str:
    if not os.path.exists(CLI):
        os.makedirs(CLI_DIR, exist_ok=True)
        data = urllib.request.urlopen(CLI_URL, timeout=180).read()
        zipfile.ZipFile(io.BytesIO(data)).extract("databricks", CLI_DIR)
        os.chmod(CLI, 0o755)
    return CLI


def run_cli(args, timeout=120, cwd=None):
    """Run the CLI as the app SP (DATABRICKS_HOST/CLIENT_ID/SECRET are injected by Apps)."""
    try:
        p = subprocess.run([ensure_cli(), *args], capture_output=True, text=True, timeout=timeout, cwd=cwd)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        out = e.stdout.decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or "")
        return -1, out, "timeout"


def run_cli_json(args, timeout=120):
    rc, out, err = run_cli([*args, "-o", "json"], timeout=timeout)
    if rc != 0:
        raise HTTPException(502, detail=(err or out).strip()[-2000:])
    m = re.search(r"\{.*\}\s*$", out, re.S)
    return json.loads(m.group(0) if m else out)


def tail_logs(run_id: str, n: int, seconds: int = 8) -> str:
    """`air logs --tail` follows an active run, so read for a few seconds then stop it.
    (`--download-to` can't be used from an App: the MLflow storage endpoint is not
    reachable from serverless compute.)"""
    p = subprocess.Popen([ensure_cli(), "air", "logs", run_id, "--tail", str(n)],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        out, _ = p.communicate(timeout=seconds)
    except subprocess.TimeoutExpired:
        p.kill()
        out, _ = p.communicate()
    return out


def user_email(request: Request) -> str:
    return request.headers.get("x-forwarded-email") or request.headers.get("x-forwarded-preferred-username") or ""


# ---- template rendering (same substitutions as quickstart.sh) -----------------
def render(text: str) -> str:
    text = text.replace("/Workspace/Users/__WS_EMAIL__", EXPERIMENT_DIR)
    return (text.replace("__IMAGE__", UC_IMAGE)
                .replace("__VOL__", VOL)
                .replace("__WS_EMAIL__", "unused"))


def build_workdir(kind: str, email: str) -> str:
    """Render the job YAML (+ the launch scripts it uploads) into a fresh directory."""
    job = JOBS[kind]
    wd = tempfile.mkdtemp(prefix=f"{kind}-")
    for name in SCRIPTS:
        src = os.path.join(TEMPLATES, name)
        if os.path.exists(src):
            open(os.path.join(wd, name), "w").write(render(open(src).read()))
            os.chmod(os.path.join(wd, name), 0o755)
    yaml_text = render(open(os.path.join(TEMPLATES, job["file"])).read())
    if email:
        yaml_text += f"\npermissions:\n  - user_name: {email}\n    level: CAN_MANAGE\n"
    open(os.path.join(wd, job["file"]), "w").write(yaml_text)
    return wd


# ---- serverless CPU jobs (data prep) ----------------------------------------
def submit_prep(kind: str, email: str):
    spec = json.loads(render(open(os.path.join(TEMPLATES, JOBS[kind]["file"])).read()))
    for t in spec["tasks"]:
        t["spark_python_task"]["python_file"] = PREP_SCRIPT
    if email:
        spec["access_control_list"] = [{"user_name": email, "permission_level": "CAN_MANAGE"}]
    wd = tempfile.mkdtemp(prefix=f"{kind}-")
    path = os.path.join(wd, "job.json")
    json.dump(spec, open(path, "w"))
    try:
        rc, out, err = run_cli(["jobs", "submit", "--json", f"@{path}", "--no-wait", "-o", "json"], timeout=180)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    m = re.search(r'"run_id":\s*(\d+)', out)
    if rc != 0 or not m:
        raise HTTPException(502, detail=(err or out).strip()[-2000:])
    run_id = m.group(1)
    SUBMITTED[run_id] = {"kind": kind, "submitted_by": email, "submitted_at": time.time()}
    return {"run_id": run_id}


def job_status(run: dict) -> str:
    st = run.get("state", {})
    return st.get("result_state") or st.get("life_cycle_state") or "UNKNOWN"


def iso(ms) -> str | None:
    return time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime(ms / 1000)) if ms else None


def list_prep_runs() -> list[dict]:
    rc, out, _ = run_cli(["jobs", "list-runs", "--run-type", "SUBMIT_RUN", "--limit", "20", "-o", "json"])
    if rc != 0:
        return []
    d = json.loads(out)
    runs = d if isinstance(d, list) else d.get("runs", [])
    return [{"run_id": str(r["run_id"]), "run_name": r.get("run_name"), "status": job_status(r),
             "started_at": iso(r.get("start_time")), "kind": "prep_data", "engine": "jobs"}
            for r in runs if (r.get("run_name") or "").startswith(PREP_RUN_PREFIX)]


def get_job_run(run_id: str) -> dict | None:
    """Return the Jobs-API view if this is a data-prep run, else None (= an AI Runtime run)."""
    rc, out, _ = run_cli(["jobs", "get-run", run_id, "-o", "json"])
    if rc != 0:
        return None
    r = json.loads(out)
    if not (r.get("run_name") or "").startswith(PREP_RUN_PREFIX):
        return None
    status = job_status(r)
    end = r.get("end_time") or int(time.time() * 1000)
    return {"engine": "jobs", "status": status, "terminal": status in TERMINAL, "experiment_name": r.get("run_name"),
            "dashboard_url": r.get("run_page_url"), "mlflow_url": None,
            "duration_seconds": (end - r["start_time"]) // 1000 if r.get("start_time") else None,
            "tasks": [{"key": t["task_key"], "run_id": str(t["run_id"]), "status": job_status(t)} for t in r.get("tasks", [])]}


def job_logs(run: dict) -> str:
    parts = []
    for t in run["tasks"]:
        parts.append(f"===== {t['key']}: {t['status']} =====")
        if t["status"] in TERMINAL:
            rc, out, err = run_cli(["jobs", "get-run-output", t["run_id"], "-o", "json"])
            if rc == 0:
                o = json.loads(out)
                text = (o.get("logs") or "") + ("\n" + o["error"] if o.get("error") else "") + \
                       ("\n" + o["error_trace"] if o.get("error_trace") else "")
                parts.append(re.sub(r"\x1b\[[0-9;]*m", "", text).strip()[-6000:] or "(出力なし)")
            else:
                parts.append((err or out)[-1000:])
        else:
            parts.append("実行中です（タスクが終わるとログが表示されます）")
    return "\n".join(parts)


# ---- API ---------------------------------------------------------------------
@app.get("/api/config")
def config(request: Request):
    return {
        "catalog": CATALOG, "schema": SCHEMA, "volume": VOL, "image": UC_IMAGE,
        "experiment_dir": EXPERIMENT_DIR, "user": user_email(request),
        "service_principal": os.environ.get("DATABRICKS_CLIENT_ID"),
        "host": "https://" + os.environ.get("DATABRICKS_HOST", "").removeprefix("https://"),
        "jobs": [{"kind": k, **v} for k, v in JOBS.items()],
    }


@app.get("/api/checks")
def checks():
    results = []

    def add(name, ok, detail):
        results.append({"name": name, "ok": ok, "detail": detail})

    try:
        rc, out, err = run_cli(["--version"], timeout=200)
        add("Databricks CLI", rc == 0, (out or err).strip())
    except Exception as e:  # download failure etc.
        add("Databricks CLI", False, str(e))
        return results

    rc, out, err = run_cli(["current-user", "me", "-o", "json"])
    add("アプリの実行 ID（サービスプリンシパル）", rc == 0,
        json.loads(out).get("displayName", "") if rc == 0 else (err or out)[-500:])

    rc, out, err = run_cli(["api", "get", f"/api/2.1/unity-catalog/software-artifacts/{CATALOG}.{SCHEMA}.{IMAGE_NAME}/versions"])
    if rc == 0:
        versions = json.loads(out).get("software_artifact_versions", [])
        hit = next((v for v in versions if IMAGE_TAG in v.get("tags", [])), None)
        if hit:
            accel = hit.get("image_acceleration_details", {}).get("status", "?")
            add("学習用イメージ（Artifact Registry）", True, f"{UC_IMAGE}  digest={hit['digest'][:19]}…  acceleration={accel}")
        else:
            tags = sorted({t for v in versions for t in v.get("tags", [])})
            add("学習用イメージ（Artifact Registry）", False, f"タグ {IMAGE_TAG} がありません（存在するタグ: {tags or 'なし'}）")
    else:
        add("学習用イメージ（Artifact Registry）", False, (err or out).strip()[-500:])

    rc, out, err = run_cli(["volumes", "read", f"{CATALOG}.{SCHEMA}.{VOLUME}"])
    add("データ用 Volume", rc == 0, VOL if rc == 0 else "未作成です（下の「Volume を作成」を押してください）")
    if rc == 0:
        for ds in ("gsm8k", "geo3k"):
            rc2, out2, _ = run_cli(["fs", "ls", f"dbfs:{VOL}/{ds}"])
            add(f"データ: {ds}", rc2 == 0 and "train.parquet" in out2,
                "train/test.parquet あり" if rc2 == 0 and "train.parquet" in out2 else "未準備（「データ準備」を実行してください）")
    return results


@app.post("/api/volume")
def create_volume():
    rc, out, err = run_cli(["volumes", "create", CATALOG, SCHEMA, VOLUME, "MANAGED", "-o", "json"])
    if rc != 0 and "already exists" not in (err + out):
        raise HTTPException(502, detail=(err or out)[-1000:])
    return {"volume": VOL}


@app.post("/api/runs/{kind}")
def submit(kind: str, request: Request):
    if kind not in JOBS:
        raise HTTPException(404, detail=f"unknown job {kind}")
    email = user_email(request)
    if JOBS[kind].get("engine") == "jobs":
        return submit_prep(kind, email)
    wd = build_workdir(kind, email)
    try:
        rc, out, err = run_cli(["air", "run", "-f", JOBS[kind]["file"], "-o", "json"], timeout=600, cwd=wd)
    finally:
        shutil.rmtree(wd, ignore_errors=True)
    m = re.search(r'"run_id":\s*"(\d+)"', out)
    if rc != 0 or not m:
        raise HTTPException(502, detail=(err or out).strip()[-2000:])
    run_id = m.group(1)
    SUBMITTED[run_id] = {"kind": kind, "submitted_by": email, "submitted_at": time.time()}
    return {"run_id": run_id}


@app.get("/api/runs")
def list_runs():
    data = run_cli_json(["air", "list", "--all-status", "--limit", "20"])
    runs = data.get("data", {}).get("runs", []) + list_prep_runs()
    runs.sort(key=lambda r: r.get("started_at") or "", reverse=True)
    for r in runs:
        meta = SUBMITTED.get(r["run_id"], {})
        r["kind"] = meta.get("kind") or r.get("kind")
        r["label"] = JOBS.get(meta.get("kind"), {}).get("label") or r.get("run_name")
        r["submitted_by"] = meta.get("submitted_by")
        r["terminal"] = r.get("status") in TERMINAL
    return runs


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    jr = get_job_run(run_id)
    if jr:
        return jr
    d = run_cli_json(["air", "get", run_id]).get("data", {})
    d["terminal"] = d.get("status") in TERMINAL
    return d


@app.get("/api/runs/{run_id}/logs")
def get_logs(run_id: str, n: int = 200):
    jr = get_job_run(run_id)
    if jr:
        return {"text": job_logs(jr)}
    return {"text": tail_logs(run_id, max(10, min(n, 2000)))}


@app.post("/api/runs/{run_id}/cancel")
def cancel(run_id: str):
    cmd = ["jobs", "cancel-run", run_id, "--no-wait"] if get_job_run(run_id) else ["air", "cancel", run_id]
    rc, out, err = run_cli(cmd)
    if rc != 0:
        raise HTTPException(502, detail=(err or out)[-1000:])
    return {"ok": True}


app.mount("/static", StaticFiles(directory=os.path.join(HERE, "static")), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(HERE, "static", "index.html"))
