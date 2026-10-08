# verl + FSDP2 GRPO for Gemma4-26B-A4B (MoE) on Databricks AI Runtime (`databricks air` CLI)

A working sample that submits a verl **GRPO (RL)** + **FSDP2** training job for
**google/gemma-4-26B-A4B-it** (a 25.8B Mixture-of-Experts model: 128 experts,
8 active, hybrid sliding+full attention, multimodal) to Databricks AI Runtime
serverless GPU, from your laptop, via the `databricks air` CLI (Databricks CLI ≥ v1.19.0).

> **Oct 2026 AI Runtime updates applied**: the AI Runtime CLI is now part of the Databricks
> CLI (`databricks air ...`; the standalone Python `air` / `databricks-air` is not needed), and
> custom images live in **Databricks Artifact Registry (Unity Catalog)** instead of Docker Hub —
> push with `databricks air images push`, reference with
> `environment.unity_catalog_image: <catalog>.<schema>.<image>:<tag>`. The old
> `environment.docker_image.url` field and `air register image` are not available in the new CLI.
> Requires the **AI Runtime Beta Features** and **Databricks Artifact Registry** previews.

**Verified working** on real hardware (all four `Training Progress 100% (3/3)` → `SUCCESS`;
re-verified 2026-10-06 with Databricks CLI v1.19.0 `databricks air` + Artifact Registry image):
- **Text, single node** (`GPU_8xH100`): real reward/grad metrics (rewards 0.7–0.97 on gsm8k).
- **Text, multi-node** (2 nodes = 16×H100): Ray cluster across nodes, clean worker exit.
- **Multimodal (image), single node**: geo3k image data, verl `main` (Gemma4 vision processor). See §6 of setup.md.
- **Multimodal (image), multi-node** (2 nodes = 16×H100): image data + Ray cluster.

> Multimodal note: image runs need (1) verl `main` (no release — incl. 0.8.0 — has
> the Gemma4 image processor yet), (2) a vLLM Gemma4 vision-bug workaround
> (`hf_overrides` nulls `use_bidirectional_attention`; vLLM 0.24 otherwise emits
> garbage for images — issue #41403, fixed in vLLM ≥0.25), and (3) memory tuning.
> All three are auto-applied via `MULTIMODAL=1`. With the workaround, image GRPO
> gets real rewards (single-node mean≈0.24/max 1.0, multi-node mean 0.06–0.39/max 1.0;
> nonzero grad_norm ⇒ the policy actually updates). Details in setup.md §6.

> **Full step-by-step build/run instructions are in [setup.md](setup.md)**
> (Japanese). It builds everything from scratch and is the primary document. This
> README is a short English overview.

---

## The six Gemma4-specific gotchas (all handled in this repo)

Gemma4 differs from earlier models (e.g. Qwen3.5) in ways that break a naive verl
GRPO setup. Each is handled in the shipped scripts; details in setup.md 付録B:

1. **Use the `-it` model.** `google/gemma-4-26B-A4B` (base) has no chat_template →
   `apply_chat_template` fails. Use `google/gemma-4-26B-A4B-it` (Apache-2.0, not gated).
2. **Patch verl's FSDP2 buffer broadcast.** verl 0.7.1's `fsdp2_load_full_state_dict`
   broadcasts `model.named_buffers()` unsorted; Gemma4's heterogeneous rotary buffers
   (256 vs 128 elements) then mismatch across ranks → NCCL deadlock. `run_grpo.sh`
   patches `fsdp_utils.py` at launch to sort buffers (idempotent).
3. **Raise the NCCL timeout.** The 25.8B MoE state-dict load exceeds the default 600s;
   `actor_rollout_ref.nccl_timeout=3600`.
4. **Use SDPA, not FlashAttention2.** Gemma4 has `global_head_dim=512`, above FA2's
   "head dimension at most 256" limit → SDPA (`use_remove_padding=False`).
5. **Text-only data (gsm8k) by default.** verl 0.7.1 rejects `Gemma4Processor`
   ("Unsupported processor type"), so image datasets (geo3k) fail. gsm8k is text-only.
6. **No FlashAttention wheel build needed** (SDPA) → the image is a single build.

## The version set

| package | version | why |
|---|---|---|
| base image | `databricksruntime/air:dcs-base-aws-devel-cu13` | cu13 nvcc to match torch 2.11 |
| torch | 2.11.0 (cu13) | pulled by vllm 0.24 |
| vllm | 0.24.0 | supports gemma4 (`Gemma4ForConditionalGeneration`) |
| transformers | ≥5.5.3 (5.14 used) | `gemma4` arch is in 5.5.3+ |
| verl | 0.7.1 (`--no-deps`, + runtime fsdp patch) | its wheel pins numpy<2 & vllm≤0.12 — install without deps |
| opencv-python-headless | **4.12.0.88** (pinned) | 5.0.0.93 bundles a FIPS libcrypto that ABORTS on `import cv2` |

> flash-attn is NOT used (Gemma4 → SDPA). The `Dockerfile` is shared with the
> Qwen build, so causal_conv1d / flash-linear-attention are present but unused by
> Gemma4; they can be dropped for a slimmer image.

---

## Quick start

Prerequisites: `git` and `databricks` (Databricks CLI ≥ v1.19.0) installed, and
`databricks auth login --host <workspace-url> --profile <PROFILE>` done. Whoever imports the
image (once per workspace) also needs `docker` (no `docker login` needed). The workspace admin
must enable the **AI Runtime Beta Features** and **Databricks Artifact Registry** previews.
Step-by-step flows per OS (macOS/Linux, Windows) are in setup.md §7.

```bash
git clone https://github.com/hiouchiy/databricks-air-verl-gemma4.git verl-gemma4
cd verl-gemma4
```

### Option A — `quickstart.sh` (interactive)
```bash
bash quickstart.sh          # Windows PowerShell: powershell -ExecutionPolicy Bypass -File .\quickstart.ps1
```
It asks for profile / catalog / schema / image tag / image source / volume / email, then:
creates the UC Volume → generates `gen/*.yaml` and `gen/*.sh` → prepares the image in
Artifact Registry as `<catalog>.<schema>.verl-gemma4:<tag>`. Image source:
`hub` = import the verified public image `docker.io/hiouchiy/verl-gemma4:v4-verify` (default; needs Docker),
`skip` = the image is already in UC (e.g. imported once for a workshop), `build` = build from the Dockerfile.

### Option B — the same steps by hand (if `quickstart.sh` fails)
`quickstart.sh` only does the three steps below; you can run them one by one (or resume from
the step that failed). Set your values first:
```bash
PROFILE=<PROFILE>; CATALOG=<catalog>; SCHEMA=<schema>; VOLUME=verl_workspace
EMAIL=<your-workspace-email>; TAG=v1
IMAGE="${CATALOG}.${SCHEMA}.verl-gemma4:${TAG}"      # UC image name (no registry host)
VOL="/Volumes/${CATALOG}/${SCHEMA}/${VOLUME}"
```
```bash
# 1) UC Volume (an "already exists" error is fine)
databricks volumes create "$CATALOG" "$SCHEMA" "$VOLUME" MANAGED -p "$PROFILE"

# 2) Image — once per workspace; skip if it is already in UC.
#    Pull happens on THIS machine (needs Docker + ~20GB disk). Without Docker, use crane (setup.md §3-3 / §7).
databricks air images push -p "$PROFILE" \
  --source docker.io/hiouchiy/verl-gemma4:v4-verify \
  --catalog "$CATALOG" --schema "$SCHEMA" --artifact "verl-gemma4:${TAG}"

# 3) Generate gen/ (replace the 3 placeholders __IMAGE__ / __VOL__ / __WS_EMAIL__)
mkdir -p gen
for t in grpo_gemma4.yaml grpo_gemma4_multinode.yaml grpo_gemma4_mm.yaml grpo_gemma4_mm_multinode.yaml \
         smoke_test.yaml prep_data_job.json prep_gsm8k_deps.yaml prep_geo3k_deps.yaml run_grpo.sh run_grpo_multinode.sh; do
  sed -e "s#__IMAGE__#${IMAGE}#g" -e "s#__VOL__#${VOL}#g" -e "s#__WS_EMAIL__#${EMAIL}#g" "$t" > "gen/$t"
done
grep -l -E "__IMAGE__|__VOL__|__WS_EMAIL__" gen/* || echo "OK: no placeholders left"
# data-prep script for the CPU serverless job (quickstart.sh does this for you)
databricks workspace mkdirs "/Workspace/Users/${EMAIL}/air-handson" -p "$PROFILE"
databricks workspace import "/Workspace/Users/${EMAIL}/air-handson/prep_data.py" --file prep_data.py --format AUTO --overwrite -p "$PROFILE"
for y in gen/*.yaml; do databricks air run -f "$y" --dry-run -p "$PROFILE"; done   # all should be "valid"
```
(You can also edit the placeholders in each YAML by hand — keep the generated files in `gen/`
together with `gen/run_grpo*.sh`, since the training YAMLs upload the script next to them.)

### Then run
> **Always run the files under `gen/`.** The `*.yaml` files at the repo root are templates with
> placeholders; running them directly fails (e.g. `Error: Folder Users is protected`).
```bash
databricks air run --file gen/smoke_test.yaml -p "$PROFILE" --watch              # optional, ~5 min (1xA10)
databricks jobs submit --json @gen/prep_data_job.json -p "$PROFILE"               # data (gsm8k + geo3k), CPU serverless, no GPU, ~1 min
databricks air run --file gen/grpo_gemma4.yaml -p "$PROFILE" --watch             # 1 node 8xH100, ~27 min
databricks air run --file gen/grpo_gemma4_multinode.yaml -p "$PROFILE" --watch   # 2 nodes 16xH100, ~28 min
databricks air list -p "$PROFILE"            # Ctrl-C on --watch does NOT stop the job
databricks air cancel <RUN_ID> -p "$PROFILE"
```

## Hands-on UI (Databricks App)
`handson-app/` is a Databricks App that runs the same steps from a browser (checks → data prep →
GRPO runs → status / logs / MLflow links → cancel). Deploy with
`PROFILE=<PROFILE> CATALOG=<catalog> SCHEMA=<schema> bash handson-app/deploy.sh` — see
[handson-app/README.md](handson-app/README.md) (JA). Jobs run as the app's service principal.

## Files
| file | role |
|---|---|
| `setup.md` | **Primary guide** — build everything from scratch, step by step (JA) |
| `quickstart.sh` | Automates Volume + YAML generation + image import/build to Artifact Registry (interactive; macOS/Linux/WSL) |
| `quickstart.ps1` | Windows PowerShell version of `quickstart.sh` (import or skip; no build) |
| `.gitattributes` | Forces LF for `.sh`/`.yaml` so Windows clones still run on Linux nodes |
| `Dockerfile` | Custom cu13 AI Runtime image (the version set above) |
| `grpo_gemma4.yaml` | `databricks air` workload: 8×H100 GRPO (text) via the custom image |
| `grpo_gemma4_multinode.yaml` | `databricks air` workload: 2-node (16×H100) GRPO (text) |
| `grpo_gemma4_mm.yaml` | `databricks air` workload: 8×H100 **multimodal (image)** GRPO (setup.md §6) |
| `grpo_gemma4_mm_multinode.yaml` | `databricks air` workload: 2-node **multimodal (image)** GRPO (setup.md §6) |
| `run_grpo.sh` | verl GRPO launcher, single node (Gemma4 fixes: fsdp patch, SDPA, nccl_timeout) |
| `run_grpo_multinode.sh` | verl GRPO launcher, multi-node (Ray cluster orchestration) |
| `prep_data.py` / `prep_data_job.json` | tiny gsm8k (text) + geo3k (image) data prep as a **CPU serverless job** (no GPU) |
| `prep_gsm8k_deps.yaml` / `prep_geo3k_deps.yaml` | the same data prep on AI Runtime 1xA10 (alternative; not needed normally) |
| `smoke_test.yaml` / `smoke_test.py` | 1×A10 import/arch check via the image |
