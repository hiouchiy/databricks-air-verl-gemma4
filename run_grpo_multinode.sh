#!/usr/bin/env bash
# =============================================================================
# verl GRPO + FSDP2 launcher for Gemma4-26B-A4B (MoE) on 2 x 8xH100 (multi-node).
#
# Adapted from the OFFICIAL verl MoE+FSDP2 example (Qwen3.5-27B MoE):
#   https://github.com/verl-project/verl/blob/main/examples/grpo_trainer/run_qwen3_5_27b_fsdp.sh
# verl has NO Gemma4-specific GRPO script (v0.8.0 ships only a Gemma4 tool parser),
# so this adapts the proven MoE+FSDP2 recipe to Gemma4:
#   - MODEL_PATH -> google/gemma-4-26B-A4B-it (MoE: 128 experts, 8 active, no shared)
#   - FSDP2 wrap target -> Gemma4TextDecoderLayer (transformers modeling_gemma4.py)
#   - paths point at UC Volumes / HF cache instead of ${HOME}/verl
#   - logger uses mlflow (AI Runtime injects the MLflow context) instead of wandb
#   - forms a Ray cluster across the AI Runtime nodes (see the ray head/worker block)
#
# Launched by grpo_gemma4_multinode.yaml under the air CLI. Hyperparameter values
# are read from the file at $HYPERPARAMETERS_PATH (air `parameters:` block).
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

# --- MULTIMODAL mode: upgrade verl to main (Gemma4 vision processor) ----------
# Same as run_grpo.sh: MULTIMODAL=1 installs verl main (0.9.0.dev0) + TransferQueue
# (both --no-deps to keep transformers 5.14.1) so verl accepts Gemma4Processor for
# image data. This must run on EVERY node before verl launches.
if [ "${MULTIMODAL:-0}" = "1" ]; then
    echo "[multimodal] installing verl main (--no-deps) for Gemma4 vision processor"
    uv pip install --python /opt/venv/bin/python3 --no-deps --force-reinstall \
        "git+https://github.com/verl-project/verl.git@main" 2>&1 | tail -3 || \
      uv pip install --no-deps --force-reinstall "git+https://github.com/verl-project/verl.git@main" 2>&1 | tail -3
    uv pip install --python /opt/venv/bin/python3 --no-deps "TransferQueue==0.1.8" 2>&1 | tail -2 || \
      uv pip install --no-deps "TransferQueue==0.1.8" 2>&1 | tail -2 || true
    python3 -c "import verl, transfer_queue, transformers; print('[multimodal] verl', verl.__version__, '| transformers', transformers.__version__, '| transfer_queue OK')" 2>&1 | tail -1 || true
fi

# --- Runtime deps: mathruler (+ qwen_vl_utils) --------------------------------
# qwen_vl_utils (verl rollout process_vision_info for images) + mathruler (geo3k/
# gsm8k reward fn). Tiny pure-python pkgs; install at launch (node has egress).
# The venv has uv, NOT pip. Target the system venv so Ray workers see it.
uv pip install --python /opt/venv/bin/python3 qwen-vl-utils mathruler 2>&1 | tail -3 || \
  uv pip install qwen-vl-utils mathruler 2>&1 | tail -3 || true
python3 -c "import qwen_vl_utils, mathruler; print('qwen_vl_utils + mathruler OK')" 2>&1 | tail -1 || true

# --- PATCH: verl 0.7.1 Gemma4 FSDP2 buffer-broadcast deadlock -----------------
# verl 0.7.1's fsdp2_load_full_state_dict broadcasts model.named_buffers() WITHOUT
# sorting. FSDP2 can return buffers in a different order per rank; Gemma4 has
# heterogeneous rotary buffers (256 vs 128 elements), so a mismatched order feeds
# different-sized tensors into the same NCCL broadcast → deterministic deadlock.
# verl main fixes this by sorting. Apply that fix in-place (idempotent, per node).
# This must run on EVERY node (the ref/actor init_model broadcasts on all ranks).
python3 - <<'PYPATCH'
import pathlib
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
# Default dataset is TEXT-ONLY gsm8k (verl 0.7.1 can't build Gemma4 image messages;
# see run_grpo.sh). Point at geo3k + set IMAGE_KEY=images for the image
# path once verl supports the Gemma4 vision processor.
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

# ---- MULTI-NODE 2x 8xH100 topology -------------------------------------------
# AI Runtime injects NUM_NODES / NODE_RANK / MASTER_ADDR / MASTER_PORT /
# LOCAL_WORLD_SIZE and runs this command ONCE PER NODE. verl uses Ray (not
# torchrun), so we must form a Ray cluster across nodes ourselves (see the
# ray head/worker block just before the launch below).
DEVICE=gpu
INFER_BACKEND=${INFER_BACKEND:-vllm}
PROJECT_NAME=${PROJECT_NAME:-GRPO-Gemma4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-GRPO-Gemma4-26B-A4B-2node}
NNODES=${NNODES:-${NUM_NODES:-2}}          # 2 nodes (or inherit AI Runtime NUM_NODES)
GEN_TP=${GEN_TP:-4}                        # vLLM TP — keep <=8 (intra-node)
SP_SIZE=${SP_SIZE:-1}                      # Ulysses sequence parallel
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.6}
n_devices_per_node=${NDEVICES_PER_NODE:-${LOCAL_WORLD_SIZE:-8}}
# fsdp_size=8 shards within each node (HSDP-style replicate across nodes) — keeps
# the heavy FSDP collectives node-local. Set FSDP_SIZE=16 to fully shard across
# both nodes (less memory/GPU, more cross-node traffic).
fsdp_size=${FSDP_SIZE:-8}

start_time=$(date +%Y%m%d)_$(date +%H%M%S)
mkdir -p logs

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    # 16 GPUs → verl's minimal batch unit is 16. train_batch_size must satisfy
    # (train_batch_size * rollout.n) % 16 == 0. With n=5, 16*5=80 (OK). 8 gave 40
    # → "real_train_batch_size (40) must be divisible by minimal possible batch (16)".
    # Env-configurable so the multimodal YAML can trim memory (image path is heavy).
    data.train_batch_size=${TRAIN_BATCH_SIZE:-16}
    data.max_prompt_length=${MAX_PROMPT_LEN:-1024}
    data.max_response_length=${MAX_RESPONSE_LEN:-1024}
    data.filter_overlong_prompts=True
    data.truncation='error'
    data.shuffle=False
)
# Multimodal image column is OPT-IN (text-only gsm8k default). Set IMAGE_KEY=images
# only with an image dataset AND a verl build supporting the Gemma4 vision processor.
if [ -n "${IMAGE_KEY:-}" ]; then
    DATA+=( data.image_key=${IMAGE_KEY} )
fi

MODEL=(
    actor_rollout_ref.model.path=${MODEL_PATH}
    # NCCL collective timeout (seconds). Default 600s is too short for Gemma4's
    # 25.8B MoE (128 experts) state-dict load+shard — a tiny init BROADCAST times
    # out and the NCCL watchdog aborts the job. Across 2 nodes the load is even
    # slower, so raise to 3600s. (verified on single-node: init BROADCAST hit 600025ms.)
    actor_rollout_ref.nccl_timeout=${NCCL_TIMEOUT:-3600}
    # Gemma4 requires SDPA, NOT FA2: global_head_dim=512 (head_dim=256) exceeds FA2's
    # "head dimension at most 256" kernel limit. use_remove_padding needs the flash-attn
    # varlen path, so it must be False under SDPA. (See run_grpo.sh.)
    actor_rollout_ref.model.use_remove_padding=${REMOVE_PADDING:-False}
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    # Activation offload (CPU) to cut actor-update peak memory — the multimodal YAML
    # turns this ON (image path OOM'd at actor update on single node).
    actor_rollout_ref.model.enable_activation_offload=${ACT_OFFLOAD:-False}
)
if [ "${ATTN:-sdpa}" = "sdpa" ]; then
    MODEL+=( +actor_rollout_ref.model.override_config.attn_implementation=sdpa )
fi

ACTOR=(
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH:-16}
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
    # Must exceed a single image's token count (geo3k = 2496) for multimodal.
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-8192}
    actor_rollout_ref.rollout.free_cache_engine=True
    # enforce_eager=True disables CUDA-graph capture → lower vLLM startup memory
    # (multimodal init needs the headroom). Text runs keep False (faster).
    actor_rollout_ref.rollout.enforce_eager=${ENFORCE_EAGER:-False}
    actor_rollout_ref.rollout.enable_prefix_caching=False
    # Actor->vLLM weight sync uses fixed-size buckets. Gemma4's embedding
    # (248320 vocab x 4096, fp32 ≈ 3.8GB) exceeds the 2GB default and verl aborts
    # with "too large to fit in the bucket". In verl 0.7.1 this field lives on
    # CheckpointEngineConfig, i.e. under rollout.checkpoint_engine (the runtime
    # error message's "rollout.update_weights_bucket_megabytes" hint is wrong for
    # 0.7.1 — that path raises "unexpected keyword argument"). Raise to 6144 MB.
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=6144
)
# MULTIMODAL vLLM fix (see run_grpo.sh): null out use_bidirectional_attention via
# hf_overrides to avoid vLLM 0.24's Gemma4 vision garbage-output bug (issue #41403).
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

# ---- Ray cluster orchestration across the AI Runtime nodes -------------------
# verl needs a Ray cluster spanning all nodes. Ray does NOT auto-discover peers
# on AI Runtime, so: NODE_RANK 0 starts the head and launches verl (which
# connects to the local head); every other node joins with `ray start --address`.
# The worker then POLLS the head and exits cleanly once training finishes (the
# head's GCS port stops responding) — do NOT use `ray start --block`, which hangs
# the worker forever after the head is done and leaves the whole job RUNNING.
# (Adapted from verl's SkyPilot/Slurm pattern onto AI Runtime's NODE_RANK/MASTER_ADDR.)
RAY_PORT=${RAY_PORT:-6379}
HEAD_ADDR="${MASTER_ADDR:-127.0.0.1}"
NODE_RANK="${NODE_RANK:-0}"

if [ "${NNODES}" -gt 1 ] && [ "${NODE_RANK}" != "0" ]; then
    # ---- WORKER node: join the head, then wait until the head goes away ----
    echo "[node ${NODE_RANK}] joining Ray head ${HEAD_ADDR}:${RAY_PORT}"
    sleep 15   # give the head a head-start
    JOINED=0
    for i in $(seq 1 30); do
        if ray start --address="${HEAD_ADDR}:${RAY_PORT}" --num-gpus="${n_devices_per_node}"; then
            JOINED=1; break
        fi
        echo "[node ${NODE_RANK}] head not up yet; retry ${i}"; sleep 10
    done
    [ "${JOINED}" = "1" ] || { echo "[node ${NODE_RANK}] failed to join Ray head"; exit 1; }
    echo "[node ${NODE_RANK}] joined. Waiting until the head/training finishes..."
    # Poll the head GCS port; when it stops accepting connections, training is done.
    MISS=0
    while true; do
        if python3 -c "import socket,sys; s=socket.socket(); s.settimeout(5); sys.exit(0 if s.connect_ex(('${HEAD_ADDR}',${RAY_PORT}))==0 else 1)" 2>/dev/null; then
            MISS=0
        else
            MISS=$((MISS+1))
            echo "[node ${NODE_RANK}] head unreachable (${MISS}/3)"
            [ "${MISS}" -ge 3 ] && { echo "[node ${NODE_RANK}] head gone; exiting worker cleanly"; ray stop 2>/dev/null || true; exit 0; }
        fi
        sleep 15
    done
fi

if [ "${NNODES}" -gt 1 ]; then
    # ---- HEAD node (rank 0): start head, wait for all nodes, then launch verl ----
    echo "[head] starting Ray head on ${HEAD_ADDR}:${RAY_PORT}"
    ray start --head --node-ip-address="${HEAD_ADDR}" --port="${RAY_PORT}" \
        --dashboard-host=0.0.0.0 --dashboard-port=8265 \
        --num-gpus="${n_devices_per_node}"
    echo "[head] waiting for all ${NNODES} nodes to join..."
    for i in $(seq 1 60); do
        JOINED=$(ray status 2>/dev/null | grep -cE "GPU|node_" || echo 0)
        NGPU=$(python3 -c "import ray; ray.init(address='auto'); print(int(ray.cluster_resources().get('GPU',0)))" 2>/dev/null || echo 0)
        echo "[head] cluster GPUs: ${NGPU} / $((NNODES*n_devices_per_node))"
        [ "${NGPU}" -ge "$((NNODES*n_devices_per_node))" ] && break
        sleep 10
    done
fi

# Launch verl on the head (single-node: runs directly; multi-node: on rank 0).
python3 -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${REF[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "$@" 2>&1 | tee logs/grpo-gemma4-26b-a4b-${start_time}.log
RC=${PIPESTATUS[0]}

# Multi-node: after training, stop the head's Ray so the worker's head-poll sees
# it disappear and exits cleanly (otherwise the worker lingers and the whole job
# stays RUNNING). --force also reaps SIGKILLed DataLoader/Ray workers that would
# otherwise keep the job RUNNING (see setup.md §P0-2). Then exit with rc.
echo "[head] training finished (rc=${RC}); stopping Ray"
ray stop --force 2>/dev/null || true
exit ${RC}
