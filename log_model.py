"""Log the trained model to the training run in MLflow (and optionally register it in Unity Catalog).

  python log_model.py --src /tmp/model_bf16 [--register <catalog>.<schema>.<name>]

--src is a Hugging Face model folder (export_model.py's output). The weights are not
loaded: MLflow (>= 2.16.1) logs the folder as is. task="llm/v1/chat" gives the model an
OpenAI-style chat signature, which Unity Catalog requires. The run is the one AI Runtime
created for this job (MLFLOW_RUN_ID, the same run verl logs its metrics to); without it a
new run is started in the current experiment.

Registering needs mlflow's "databricks" extras (to upload the files to Unity Catalog);
images built before they were added to the Dockerfile lack them, so they are installed
here on demand.

Exit code 0 on success, 1 on failure (run_grpo.sh then falls back to copying the model to a Volume).
"""
import argparse
import os
import shutil
import subprocess
import sys
import time


def ensure_uc_extras():
    import mlflow
    spec = f"mlflow[databricks]=={mlflow.__version__}"
    cmd = ([shutil.which("uv"), "pip", "install", "-q", "--python", sys.executable, spec] if shutil.which("uv")
           else [sys.executable, "-m", "pip", "install", "-q", spec])
    print(f"[log_model] installing {spec}", flush=True)
    subprocess.run(cmd, check=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True)
    p.add_argument("--register", default="", help="Unity Catalog model name <catalog>.<schema>.<name>")
    a = p.parse_args()
    t0 = time.time()
    try:
        import mlflow

        if a.register:
            ensure_uc_extras()
            mlflow.set_registry_uri("databricks-uc")
        run_id = os.environ.get("MLFLOW_RUN_ID")
        with (mlflow.start_run(run_id=run_id) if run_id else mlflow.start_run()) as run:
            info = mlflow.transformers.log_model(
                transformers_model=a.src, task="llm/v1/chat", name="model",
                registered_model_name=a.register or None)
        print(f"[log_model] logged to MLflow run {run.info.run_id}: {info.model_uri} ({time.time() - t0:.0f}s)")
        if a.register:
            print(f"[log_model] registered in Unity Catalog: {a.register} version {info.registered_model_version}")
        return 0
    except Exception as e:  # reported; the caller falls back to a Volume copy
        print(f"[log_model] FAILED after {time.time() - t0:.0f}s: {type(e).__name__}: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
