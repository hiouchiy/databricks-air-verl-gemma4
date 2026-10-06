#!/usr/bin/env bash
# =============================================================================
# verl GRPO + FSDP2 launcher for Gemma4-26B-A4B (MoE) on a single 8x H100 node.
#
# Adapted from the OFFICIAL verl MoE+FSDP2 example (Qwen3.5-27B MoE):
#   https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh
# verl has NO Gemma4-specific GRPO script (v0.8.0 ships only a Gemma4 tool
# parser), so this adapts the proven Qwen3.5-27B MoE+FSDP2 recipe to Gemma4:
#   - MODEL_PATH -> google/gemma-4-26B-A4B-it (MoE: 128 experts, 8 active, no shared;
#     hybrid sliding+full attention; multimodal — arch Gemma4ForConditionalGeneration)
#   - FSDP2 wrap target -> Gemma4TextDecoderLayer (from transformers modeling_gemma4.py)
#   - paths point at UC Volumes / HF cache instead of ${HOME}/verl
#   - logger uses mlflow (AI Runtime injects the MLflow context) instead of wandb
#
# Launched by train_grpo.yaml under the air CLI. Hyperparameter values are read
# from the file at $HYPERPARAMETERS_PATH (air `parameters:` block).
#
# REQUIRED DEPENDENCY (from the official script header):
#   GPU: vllm==0.18.0, transformers@<cc7ab9be>
# =============================================================================
set -xeuo pipefail

# --- Disable OpenSSL FIPS enforcement -----------------------------------------
# The node's OpenSSL aborts a Ray worker with "FATAL FIPS SELFTEST FAILURE"
# (Fatal Python error: Aborted) partway into verl init. Force FIPS off for the
# driver; Ray workers inherit the process environment, so exporting here covers
# them too. (Do NOT touch RAY_RUNTIME_ENV_HOOK — setting it to "" makes Ray try
# to load an empty class path and crash with "valid path like mymodule.provider_class".)
export OPENSSL_FORCE_FIPS_MODE=0
export OPENSSL_FIPS=0

# Reduce CUDA memory fragmentation (helps avoid OOM when rollout + training share
# the GPU, especially for memory-heavy multimodal/image batches).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# --- MULTIMODAL mode: upgrade verl to main (Gemma4 vision processor) ----------
# The v3 image ships verl 0.7.1, whose processor factory rejects Gemma4Processor
# ("Unsupported processor type") — so IMAGE data (geo3k) can't build messages.
# verl's Gemma4Processor support is only on the MAIN branch (NOT in any release
# incl. 0.8.0; verified). For image training, set MULTIMODAL=1 to install verl
# main (pinned commit 8718ca30, 0.10.0.dev0) with --no-deps at launch (keeps vllm0.24/torch2.11/transformers5.14).
# main also already sorts named_buffers, so the fsdp patch below becomes a no-op.
if [ "${MULTIMODAL:-0}" = "1" ]; then
    # Pinned to the verl main commit verified on 2026-10-06 (verl 0.10.0.dev0); main
    # moves daily, so an unpinned @main can break without notice. Override with VERL_REF.
    VERL_REF="${VERL_REF:-8718ca30a3f002f93b7c4fd99b9b2506718681bc}"
    echo "[multimodal] installing verl main@${VERL_REF} (--no-deps) for Gemma4 vision processor"
    uv pip install --python /opt/venv/bin/python3 --no-deps --force-reinstall \
        "git+https://github.com/verl-project/verl.git@${VERL_REF}" 2>&1 | tail -3 || \
      uv pip install --no-deps --force-reinstall "git+https://github.com/verl-project/verl.git@${VERL_REF}" 2>&1 | tail -3
    # verl main's TaskRunnerV1 imports TransferQueue (a new dep, PyPI: TransferQueue).
    # Install it --no-deps so it does NOT drag transformers down to verl main's
    # declared pin (<5.11) — we must keep transformers 5.14.1 for gemma4 + vllm 0.24.
    uv pip install --python /opt/venv/bin/python3 --no-deps "TransferQueue==0.1.8" 2>&1 | tail -2 || \
      uv pip install --no-deps "TransferQueue==0.1.8" 2>&1 | tail -2 || true
    python3 -c "import verl, transfer_queue; import transformers; print('[multimodal] verl', verl.__version__, '| transformers', transformers.__version__, '| transfer_queue OK')" 2>&1 | tail -1 || true
fi

# --- Runtime deps: mathruler (+ qwen_vl_utils) --------------------------------
# verl's geo3k reward fn imports mathruler (extract_boxed_content/grade_answer).
# verl's rollout process_vision_info for the image data path (geo3k uses
# data.image_key=images) imports qwen_vl_utils regardless of the model family, so
# we install it too. Both are tiny pure-python pkgs; install at launch from public
# PyPI (the node has egress — it pulls model weights from HF). For a self-contained
# image, add them to the Dockerfile deps instead.
# The venv has uv, NOT pip (`python3 -m pip` → "No module named pip"). Use uv,
# targeting the system venv so the Ray workers (which share /opt/venv) see it.
uv pip install --python /opt/venv/bin/python3 qwen-vl-utils mathruler 2>&1 | tail -3 || \
  uv pip install qwen-vl-utils mathruler 2>&1 | tail -3 || true
python3 -c "import qwen_vl_utils, mathruler; print('qwen_vl_utils + mathruler OK')" 2>&1 | tail -1 || true

# --- PATCH: verl 0.7.1 Gemma4 FSDP2 buffer-broadcast deadlock -----------------
# verl 0.7.1's fsdp2_load_full_state_dict broadcasts model.named_buffers() WITHOUT
# sorting. FSDP2 can return buffers in a different order per rank; Gemma4 has
# heterogeneous rotary buffers (256 vs 128 elements), so a mismatched order feeds
# different-sized tensors into the same NCCL broadcast → deterministic deadlock
# (observed: SeqNum=1050 BROADCAST NumelIn=256 hangs in ref_policy init_model).
# verl main fixes this by sorting: sorted(model.named_buffers(), key=lambda x: x[0]).
# Apply that fix in-place to the installed verl (idempotent).
python3 - <<'PYPATCH'
import re, pathlib
try:
    import verl.utils.fsdp_utils as F
    p = pathlib.Path(F.__file__)
    src = p.read_text()
    old = "for name, buf in model.named_buffers():"
    new = "for name, buf in sorted(model.named_buffers(), key=lambda x: x[0]):"
    if new in src:
        print("[fsdp patch] already sorted — no change")
    elif old in src:
        p.write_text(src.replace(old, new))
        print("[fsdp patch] applied: sorted(model.named_buffers()) in fsdp_utils.py")
    else:
        print("[fsdp patch] WARNING: target line not found; verl may have changed. Skipping.")
except Exception as e:
    print(f"[fsdp patch] WARNING: could not patch fsdp_utils: {e}")
PYPATCH

# ---- read air parameters -> shell vars ---------------------------------------
# air writes the `parameters:` block as a YAML file at $HYPERPARAMETERS_PATH
# (NOT JSON). Parse YAML (yaml if available, else a tiny "key: value" fallback);
# print nothing on a missing key so the shell `|| echo <default>` kicks in.
HP="${HYPERPARAMETERS_PATH:-}"
getp() {
  python3 - "$1" <<'PY'
import os, sys
key = sys.argv[1]
path = os.environ.get("HYPERPARAMETERS_PATH", "")
val = None
try:
    import yaml
    with open(path) as f:
        d = yaml.safe_load(f) or {}
    val = d.get(key)
except Exception:
    try:  # minimal "key: value" fallback if pyyaml is absent
        with open(path) as f:
            for line in f:
                if ":" in line:
                    k, _, v = line.partition(":")
                    if k.strip() == key:
                        val = v.strip().strip('"').strip("'"); break
    except Exception:
        val = None
if val is not None:
    print(val)
PY
}

MODEL_PATH=${MODEL_PATH:-$( [ -n "$HP" ] && getp model_name || echo "google/gemma-4-26B-A4B-it" )}
MODEL_PATH=${MODEL_PATH:-google/gemma-4-26B-A4B-it}
# Default dataset is TEXT-ONLY gsm8k (see prep_gsm8k_deps.yaml). geo3k (images)
# needs a multimodal processor that verl 0.7.1 does not support for Gemma4
# (raises "Unsupported processor type: Gemma4Processor"). To use images once verl
# gains Gemma4 vision support, point these at .../geo3k/*.parquet and set IMAGE_KEY=images.
TRAIN_FILE=${TRAIN_FILE:-$( [ -n "$HP" ] && getp train_files || echo "__VOL__/gsm8k/train.parquet" )}
TRAIN_FILE=${TRAIN_FILE:-__VOL__/gsm8k/train.parquet}
TEST_FILE=${TEST_FILE:-$( [ -n "$HP" ] && getp val_files || echo "__VOL__/gsm8k/test.parquet" )}
TEST_FILE=${TEST_FILE:-__VOL__/gsm8k/test.parquet}
CKPTS_DIR=${CKPTS_DIR:-$( [ -n "$HP" ] && getp output_dir || echo "__VOL__/ckpt/grpo-gemma4-26b-a4b" )}
CKPTS_DIR=${CKPTS_DIR:-__VOL__/ckpt/grpo-gemma4-26b-a4b}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-$( [ -n "$HP" ] && getp total_epochs || echo 15 )}
# SMOKE: cap total optimizer steps so we validate the pipeline in minutes, not
# hours. Full training is left to the customer. Set to 0 to disable the cap.
TOTAL_TRAIN_STEPS=${TOTAL_TRAIN_STEPS:-$( [ -n "$HP" ] && getp total_training_steps 2>/dev/null || echo 3 )}

# ---- single-node 8x H100 topology (verl expects these) -----------------------
DEVICE=gpu
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-Gemma4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Gemma4-26B-A4B}
NNODES=${NNODES:-1}
GEN_TP=${GEN_TP:-4}                       # vLLM tensor parallel for rollout
SP_SIZE=${SP_SIZE:-1}                     # Ulysses sequence parallel
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
n_devices_per_node=${NDEVICES_PER_NODE:-8}
fsdp_size=${FSDP_SIZE:-8}

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE:-8}
    # Prompt/response lengths are memory-sensitive. Image inputs (geo3k) expand the
    # prompt with many vision tokens + a vision tower, so multimodal runs OOM at the
    # text defaults — the multimodal YAML lowers these via env.
    data.max_prompt_length=${MAX_PROMPT_LEN:-1024}
    data.max_response_length=${MAX_RESPONSE_LEN:-1024}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)
# Multimodal image column is OPT-IN. Default (gsm8k) is text-only, so image_key is
# NOT set. Set IMAGE_KEY=images ONLY with an image dataset (geo3k) AND a verl build
# that supports the Gemma4 vision processor — otherwise verl asserts
# "processor is needed to process image and video".
if [ -n "${IMAGE_KEY:-}" ]; then
    DATA+=( data.image_key=${IMAGE_KEY} )
fi

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    # NCCL collective timeout (seconds). Default is 600s, but Gemma4-26B-A4B is a
    # 25.8B MoE (128 experts → very many tensors); fsdp2_load_full_state_dict takes
    # >600s to load+shard, so a tiny init BROADCAST times out and the NCCL watchdog
    # aborts the whole job (verified: SeqNum=1050 BROADCAST ran 600025ms in init_model).
    # Raise to 3600s to give the MoE state-dict load headroom.
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-3600}
    # ATTENTION: Gemma4 requires SDPA, NOT FlashAttention2. Gemma4's text config has
    # global_head_dim=512 (head_dim=256), but FA2's forward kernel supports head
    # dimension AT MOST 256 → "RuntimeError: FlashAttention forward only supports head
    # dimension at most 256" during the actor/ref forward (verified on run 5). So we
    # default ATTN=sdpa here. use_remove_padding relies on the flash-attn varlen path,
    # so it must be False under SDPA. (Set ATTN=flash_attention_2 REMOVE_PADDING=True
    # only for models whose head_dim<=256.)
    actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING:-False}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # Activation offload moves activations to CPU to cut peak actor-update memory.
    # OFF by default (text runs fit); the multimodal YAML turns it ON to avoid the
    # actor-update OOM seen with image inputs (vision tower + image tokens).
    actor_rollout_ref.model.enable_activation_offload=${ACT_OFFLOAD:-False}
)
# Force SDPA by default (Gemma4 head_dim exceeds FA2's limit). Override with
# ATTN=flash_attention_2 only if you know the model's head_dim<=256.
if [ "${ATTN:-sdpa}" = "sdpa" ]; then
    MODEL+=( +actor_rollout_ref.model.override_config.attn_implementation=sdpa )
fi

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH:-8}
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.kl_loss_coef=0.01
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    # Pass an explicit LIST of the decoder layer class for FSDP2 auto-wrap.
    # Gemma4's text decoder layer class is Gemma4TextDecoderLayer (verified in
    # transformers modeling_gemma4.py). Relying on model._no_split_modules can hit
    # verl's apply_fsdp2 bug ("'set' object is not subscriptable"), so set it here.
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer]
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.entropy_from_logits_with_chunking=True
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=${SP_SIZE}
    actor_rollout_ref.ref.use_torch_compile=False
    actor_rollout_ref.ref.fsdp_config.offload_policy=True
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[Gemma4TextDecoderLayer]
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=${INFER_BACKEND}
    actor_rollout_ref.rollout.ignore_eos=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.rollout.tensor_model_parallel_size=${GEN_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.n=${ROLLOUT_N:-5}
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
    actor_rollout_ref.rollout.free_cache_engine=True
    # enforce_eager: False uses CUDA graphs (faster) but graph capture consumes
    # several GiB at vLLM init. Multimodal init OOM'd on that → the multimodal YAML
    # sets ENFORCE_EAGER=True (disables graph capture, lower startup memory).
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    # Actor->vLLM weight sync uses fixed-size buckets. Gemma4's embedding
    # (262144 vocab x 2816 hidden, fp32 ≈ 2.95GB) exceeds the 2GB default and verl
    # aborts with "too large to fit in the bucket". This field lives on
    # CheckpointEngineConfig, i.e. under rollout.checkpoint_engine. Raise to 6144 MB
    # (same safe value proven on the Qwen3.5 stack; adjust if the bucket error persists).
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
    # Disable vLLM's custom (peer-to-peer) all-reduce kernel. On Gemma4 MoE + H100
    # with CUDA-graph capture (enforce_eager=False), the custom all-reduce crashes
    # during graph capture: "Cuda error custom_all_reduce.cuh:455 'invalid argument'"
    # → EngineCore init fails and the run dies. Falling back to NCCL all-reduce is
    # stable (Qwen3.5 on the same image never hits this). This is a vLLM ENGINE ARG
    # (EngineArgs.disable_custom_all_reduce), NOT an env var — the earlier
    # VLLM_DISABLE_CUSTOM_ALL_REDUCE env does not exist in vLLM 0.24 and was ignored.
    # verl forwards rollout.engine_kwargs.vllm.* into vLLM (**engine_kwargs), the same
    # path used by the multimodal hf_overrides below.
    '+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_custom_all_reduce=True'
)
# MULTIMODAL vLLM fix: vLLM 0.24's Gemma4 vision path emits garbage (word salad) for
# image inputs due to a use_bidirectional_attention / use_mm_prefix regression
# (vLLM issue #41403; fixed in vLLM >=0.25). Workaround WITHOUT upgrading vLLM: pass
# hf_overrides to the rollout engine to null out use_bidirectional_attention. Verified
# on a node: with this override the image response is coherent + contains \boxed{}.
# verl forwards rollout.engine_kwargs.vllm.* into vLLM's AsyncLLM (**engine_kwargs);
# hf_overrides is a dict (non-None) so verl's None-filter keeps it. Only set it in
# multimodal mode (text runs don't need it).
if [ "${MULTIMODAL:-0}" = "1" ]; then
    ROLLOUT+=( '+actor_rollout_ref.rollout.engine_kwargs.vllm.hf_overrides={text_config:{use_bidirectional_attention:null}}' )
fi

TRAINER=(
    trainer.critic_warmup=0
    trainer.logger=['console','mlflow']    # AI Runtime injects MLflow context
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${n_devices_per_node}
    trainer.nnodes=${NNODES}
    trainer.balance_batch=False
    trainer.default_local_dir="${CKPTS_DIR}"
    trainer.val_before_train=False
    trainer.save_freq=-1
    trainer.test_freq=-1
    trainer.total_epochs=${TOTAL_EPOCHS}
)

# SMOKE: cap optimizer steps (a few steps is enough to prove the stack runs).
if [ "${TOTAL_TRAIN_STEPS}" != "0" ]; then
    TRAINER+=( trainer.total_training_steps=${TOTAL_TRAIN_STEPS} )
fi

# --- Clean shutdown so the AI Runtime job actually terminates (avoids billing) --
# Even after training finishes, the job can stay RUNNING because a Ray/DataLoader
# worker gets SIGKILLed at teardown and the Ray runtime never fully exits (observed
# on both success and failure → up to timeout_minutes of 8xH100 billed). So we stop
# Ray explicitly on exit.
#
# This MUST be a `trap ... EXIT`, not inline after the launch: `set -e` + pipefail
# means a verl FAILURE in the `... | tee` pipeline exits the script IMMEDIATELY,
# before any post-launch line. Gemma4 GRPO failing is exactly when cleanup matters
# most (a failed job otherwise lingers RUNNING and keeps billing 8xH100), so cleanup
# has to run on the failure path too. The trap fires on BOTH paths; `$?` at trap
# entry is verl's status (via pipefail) and re-exiting with it marks the run terminal.
cleanup() {
  rc=$?
  ray stop --force 2>/dev/null || true
  echo "[run_grpo] verl exit code=${rc}; ray stopped; exiting."
  exit ${rc}
}
trap cleanup EXIT

python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee logs/grpo-gemma4-26b-a4b-${start_time}.log
# On success, fall through to the EXIT trap (rc=0). On failure, `set -e` jumps
# straight to the EXIT trap with verl's non-zero code.
