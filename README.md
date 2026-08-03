# verl + FSDP2 GRPO for Gemma4-26B-A4B (MoE) on Databricks AI Runtime (`air` CLI)

A working sample that submits a verl **GRPO (RL)** + **FSDP2** training job for
**google/gemma-4-26B-A4B-it** (a 25.8B Mixture-of-Experts model: 128 experts,
8 active, hybrid sliding+full attention, multimodal) to Databricks AI Runtime
serverless GPU, from your laptop, via the `air` CLI.

**Verified working** on real hardware (all four `Training Progress 100% (3/3)` → `SUCCESS`):
- **Text, single node** (`GPU_8xH100`): real reward/grad metrics (rewards 0.7–0.97 on gsm8k).
- **Text, multi-node** (2 nodes = 16×H100): Ray cluster across nodes, clean worker exit.
- **Multimodal (image), single node**: geo3k image data, verl `main` (Gemma4 vision processor). See §6 of setup.md.
- **Multimodal (image), multi-node** (2 nodes = 16×H100): image data + Ray cluster.

> Multimodal note: image runs need verl `main` (not any release — 0.8.0 lacks the
> Gemma4 image processor), enabled at launch via `MULTIMODAL=1`, plus memory tuning.
> The image pipeline runs end-to-end, but the geo3k reward returned 0 (the reward fn
> doesn't parse Gemma4's answer format) — pipeline works; real training needs reward
> tuning. Details in setup.md §6.

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

Prerequisites: `git`, `databricks`, `air`, and `docker` CLIs installed; `databricks
auth login` and `docker login` done. Then clone this repo and run the script:

```bash
git clone <REPO_URL> verl-gemma4
cd verl-gemma4
bash quickstart.sh
```

It interactively collects your profile / Docker Hub user / catalog / schema /
volume / email, then automates: create UC Volume → build the image (single build)
→ push + register. Data prep and training are run manually afterward (see setup.md).

## Files
| file | role |
|---|---|
| `setup.md` | **Primary guide** — build everything from scratch, step by step (JA) |
| `quickstart.sh` | Automates the image build + register (interactive) |
| `Dockerfile` | Custom cu13 AI Runtime image (the version set above) |
| `grpo_gemma4.yaml` | air workload: 8×H100 GRPO (text) via the custom image |
| `grpo_gemma4_multinode.yaml` | air workload: 2-node (16×H100) GRPO (text) |
| `grpo_gemma4_mm.yaml` | air workload: 8×H100 **multimodal (image)** GRPO (setup.md §6) |
| `grpo_gemma4_mm_multinode.yaml` | air workload: 2-node **multimodal (image)** GRPO (setup.md §6) |
| `run_grpo.sh` | verl GRPO launcher, single node (Gemma4 fixes: fsdp patch, SDPA, nccl_timeout) |
| `run_grpo_multinode.sh` | verl GRPO launcher, multi-node (Ray cluster orchestration) |
| `prep_gsm8k_deps.yaml` | tiny text-only gsm8k data prep (base env, no image) |
| `smoke_test.yaml` / `smoke_test.py` | 1×A10 import/arch check via the image |
