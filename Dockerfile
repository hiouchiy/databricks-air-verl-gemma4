# =============================================================================
# Custom AI Runtime image: verl + vLLM 0.24 + Gemma4 (MoE) stack.
#
# MUST build FROM the Databricks AI Runtime base image (CUDA/NCCL/cloud-NW /
# entrypoint pre-configured). You cannot use an arbitrary Docker image directly.
#
# Base facts (verified by running the image):
#   databricksruntime/air:dcs-base-aws-devel-cu13
#     = Ubuntu + CUDA 13.0 (nvcc 13.0) + Python 3.12.3 + uv, venv at /opt/venv,
#       NO torch/transformers/vllm preinstalled (bare CUDA+venv).
#
# Version rationale (Gemma4 = gemma4 arch, Gemma4ForConditionalGeneration) — the
# vLLM 0.24 stack is shared with the Qwen3.5 build and verified on a real node:
#   - gemma4 requires transformers>=5.5.3 (model_type "gemma4"; config says 5.5.0.dev0).
#   - transformers 5.5.3+ pairs with vllm>=0.24, which pulls torch==2.11.0 (cu13).
#     vLLM 0.24.0 requirements/common.txt pins `transformers >= 5.5.3`; the 0.24
#     model registry includes gemma4 (Gemma4ForConditionalGeneration / gemma4_mm).
#   - torch 2.11 is a cu13 build → flash-attn must be compiled against cu13, so the
#     base MUST be the -cu13 devel image (nvcc 13.0). => dcs-base-aws-devel-cu13.
#   - the released `verl` wheel (through 0.8.0) hard-pins numpy<2.0.0 AND vllm<=0.12.0
#     — incompatible with vllm 0.24 (needs numpy>=2). So verl is installed with
#     --no-deps; we provide its real runtime deps ourselves.
#   - NOTE vs the Qwen3.5 image: Gemma4 does NOT use Gated-DeltaNet, so causal_conv1d
#     / flash-linear-attention are NOT required by Gemma4. They are kept below only
#     because they are cheap and harmless; they can be dropped for a slimmer image
#     once a Gemma4-only build is confirmed (left in for now to reuse the proven layer set).
#
# Build (amd64) + push to Docker Hub (logged in via `docker login`):
#   docker build --platform linux/amd64 \
#     --build-arg PIP_INDEX_URL=https://pypi.org/simple \
#     --build-arg BUILD_FLASH_ATTN=1 \
#     -v "$PWD/wheelhouse:/wheelhouse:ro" \
#     -t docker.io/<DOCKERHUB_USER>/verl-gemma4:v1 -f Dockerfile .
#   docker push docker.io/<DOCKERHUB_USER>/verl-gemma4:v1
#   air register image <DOCKERHUB_USER>/verl-gemma4:v1 -p <PROFILE>
# =============================================================================
FROM databricksruntime/air:dcs-base-aws-devel-cu13

# Databricks base manages Python in /opt/venv via uv. Install on top with uv pip.
ARG PIP_INDEX_URL=https://pypi.org/simple
ENV CUDA_HOME=/usr/local/cuda
# QEMU cross-arch builds get flaky mid-stream on huge wheels (torch/cudnn ~0.5-1GB).
# Lower concurrency + long timeout + retries. CRITICAL: do NOT use --no-cache, so
# uv's download cache persists across retry iterations — a flaky wheel resumes
# from cache instead of re-downloading the entire dependency set every attempt.
ENV UV_HTTP_TIMEOUT=1200
ENV UV_CONCURRENT_DOWNLOADS=1
# UV_NO_CACHE: do NOT persist uv's download cache into image layers. The cache
# was ~11GB (redundant with the installed venv) and bloated the image to 31GB,
# which timed out workspace-registry replication. Installing from the mounted
# /wheelhouse means no re-download cost anyway. This keeps the image ~20GB.
ENV UV_NO_CACHE=1

# Offline wheelhouse (full binary closure, ~4.3GB) fetched on a Databricks node
# into a UC Volume and copied to ./wheelhouse (see STATUS.md / build_wheelhouse.yaml).
# We BIND-MOUNT it per-RUN (not COPY) so the 4.3GB never lands in an image layer
# — COPYing it caused "no space left on device" at layer-commit. Each install RUN
# gets it at /wheelhouse via --mount and uses --find-links (index still used as a
# fallback for the causal_conv1d/flash-attn sdists that compile at build time).
ENV FIND_LINKS="--find-links /wheelhouse"

# The cu13 base's venv Python lacks dev headers; causal_conv1d/flash-attn C++
# extensions need Python.h at /usr/include/python3.12/. Install the dev headers.
RUN apt-get update && apt-get install -y --no-install-recommends python3.12-dev \
    && rm -rf /var/lib/apt/lists/*

# --- vLLM 0.24.0: pulls the Qwen3.5-capable set (torch==2.11.0/cu13,
#     transformers>=5.5.3, opencv, numpy>=2). Big wheels from /wheelhouse.
RUN for i in $(seq 1 8); do \
      uv pip install ${FIND_LINKS} --index-url ${PIP_INDEX_URL} "vllm==0.24.0" && break || \
      { echo "vllm attempt $i disconnected; resuming from cache"; sleep 10; }; \
    done

# --- transformers 5.5.3+ (qwen3_5 arch) + verl runtime deps -------------------
# Re-pin torch==2.11.0 so the resolver does not drift off the vllm 0.24 ABI.
RUN for i in $(seq 1 8); do \
      uv pip install ${FIND_LINKS} --index-url ${PIP_INDEX_URL} \
        "torch==2.11.0" \
        "transformers>=5.5.3" "accelerate>=0.34" "datasets>=3.0" \
        "hydra-core" "omegaconf" "einops" "ninja" "codetiming" "dill" "peft" \
        "pylatexenc" "torchdata" "ray[default]>=2.41.0" "wandb" "tensorboard" \
        "tensordict>=0.8.0,<=0.10.0,!=0.9.0" "pyarrow>=19.0.0" "mlflow>=3.6" \
        "qwen-vl-utils" "mathruler" && break || \
      { echo "deps attempt $i disconnected; resuming from cache"; sleep 10; }; \
    done

# --- verl with --no-deps (its wheel pins numpy<2 & vllm<=0.12, both incompatible
#     with vllm 0.24). Real runtime deps supplied above; --no-deps lets verl
#     coexist with the Qwen3.5-era stack.
RUN for i in $(seq 1 8); do \
      uv pip install ${FIND_LINKS} --no-deps "verl==0.7.1" --index-url ${PIP_INDEX_URL} && break || \
      { echo "verl attempt $i disconnected; resuming from cache"; sleep 10; }; \
    done

# --- Qwen3.5 Gated-DeltaNet kernels -------------------------------------------
# fla ships a pure-python wheel; causal_conv1d compiles against nvcc (devel base).
RUN for i in $(seq 1 8); do \
      uv pip install ${FIND_LINKS} --index-url ${PIP_INDEX_URL} "flash-linear-attention>=0.5.1" && break || \
      { echo "fla attempt $i disconnected; resuming from cache"; sleep 10; }; \
    done

# causal_conv1d builds from sdist (no prebuilt wheel). Needs nvcc (devel base).
RUN uv pip install ${FIND_LINKS} --index-url ${PIP_INDEX_URL} --no-build-isolation "causal-conv1d"

# --- Downgrade opencv-python-headless (CRITICAL) — must be the LAST pip op ------
# vllm 0.24 pulls opencv-python-headless==5.0.0.93, whose BUNDLED libcrypto
# (libcrypto-*.so.1.1.1k, OpenSSL 1.1.1k) is FIPS-enforcing and ABORTS on `import
# cv2` ("crypto/fips/fips.c: FATAL FIPS SELFTEST FAILURE"), killing any verl
# worker that imports cv2 via mistral_common. It's a bundled canister, so OPENSSL_*
# env vars do NOT help. opencv 4.12.0.88 ships FIPS-free bundled libs. This runs
# LAST (after verl/fla/causal_conv1d) so nothing reinstalls 5.0 afterward; we
# uninstall 5.0 first to force removal of its .libs dir. Verified in-image:
# cv2 4.12.0 OK + transformers.tokenization_mistral_common OK.
RUN uv pip uninstall opencv-python-headless opencv-python 2>/dev/null || true; \
    uv pip install ${FIND_LINKS} --index-url ${PIP_INDEX_URL} "opencv-python-headless==4.12.0.88" && \
    python3 -c "import cv2, transformers.tokenization_mistral_common; print('cv2', cv2.__version__, 'FIPS-clean')"

# flash-attn: enables FlashAttention2 (higher throughput than the SDPA fallback).
# We install a PREBUILT wheel from /wheelhouse — compiled once on an AI Runtime
# H100 node against this exact torch 2.11/cu13 ABI (see build_flash_attn.yaml),
# NOT compiled here (a from-source build peaks >10GB/job and takes ~76 min).
# Set BUILD_FLASH_ATTN=1 (default) to include it; the prebuilt wheel install is
# instant. If the wheel is absent, the image still works via SDPA — set
# run_grpo.sh to attn_implementation=sdpa + use_remove_padding=False in that case.
ARG BUILD_FLASH_ATTN=1
RUN if [ "${BUILD_FLASH_ATTN}" = "1" ] && ls /wheelhouse/flash_attn-*.whl >/dev/null 2>&1; then \
      uv pip install --no-deps /wheelhouse/flash_attn-*.whl && \
      python3 -c "import flash_attn; print('flash_attn', flash_attn.__version__, 'installed (prebuilt wheel)')"; \
    else \
      echo "flash-attn NOT installed (no prebuilt wheel in /wheelhouse). Qwen3.5 uses SDPA fallback; build the wheel via build_flash_attn.yaml and place it in ./wheelhouse."; \
    fi

# --- Training code baked in (WORKDIR is ignored; use absolute paths) ----------
COPY ./run_grpo.sh /app/run_grpo.sh
COPY ./smoke_test.py /app/smoke_test.py
