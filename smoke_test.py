"""
Cheap smoke test (1x A10) — validate the environment BEFORE burning 8x H100.

Checks, in order:
  1. verl / vllm / transformers / torch import and versions
  2. transformers recognizes the `gemma4` architecture (config load, NO weights)
  3. AutoConfig for google/gemma-4-26B-A4B loads (downloads only config, no token)
  4. the Gemma4 text decoder layer class is importable under the expected name
     (Gemma4TextDecoderLayer — the FSDP2 wrap target)

Exits non-zero on the first failure so `air` marks the run failed and the log
shows exactly which layer of the stack is the problem.
"""

import sys


def check(name, fn):
    try:
        val = fn()
        print(f"[OK]   {name}: {val}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


def main() -> None:
    ok = True

    def _torch():
        import torch
        return f"torch {torch.__version__}, cuda={torch.cuda.is_available()}, ndev={torch.cuda.device_count()}"

    def _tf():
        import transformers
        return transformers.__version__

    def _vllm():
        import vllm
        return vllm.__version__

    def _verl():
        import verl
        return getattr(verl, "__version__", "unknown")

    def _arch():
        # gemma4 must be a registered model type in this transformers build.
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
        assert "gemma4" in CONFIG_MAPPING_NAMES, "gemma4 not in CONFIG_MAPPING_NAMES"
        return f"gemma4 -> {CONFIG_MAPPING_NAMES['gemma4']}"

    def _cfg():
        # Downloads only config.json (tiny), not weights. Gemma4 is Apache-2.0 and
        # NOT gated, so no HF token is required.
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained("google/gemma-4-26B-A4B")
        # report MoE + text decoder facts so the log confirms what we expect
        tcfg = getattr(cfg, "text_config", cfg)
        experts = getattr(tcfg, "num_experts", "?")
        topk = getattr(tcfg, "top_k_experts", "?")
        return f"{cfg.model_type} / {type(cfg).__name__} (experts={experts}, top_k={topk})"

    def _decoder_cls():
        # The FSDP2 wrap target used by run_grpo.sh must exist under this name.
        from transformers.models.gemma4 import modeling_gemma4
        assert hasattr(modeling_gemma4, "Gemma4TextDecoderLayer"), \
            "Gemma4TextDecoderLayer not found in modeling_gemma4"
        return "Gemma4TextDecoderLayer present"

    def _flash():
        import flash_attn
        return flash_attn.__version__

    ok &= check("torch", _torch)
    ok &= check("transformers", _tf)
    ok &= check("vllm", _vllm)
    ok &= check("verl", _verl)
    ok &= check("gemma4 arch registered", _arch)
    ok &= check("gemma-4-26B-A4B AutoConfig", _cfg)
    ok &= check("Gemma4TextDecoderLayer import", _decoder_cls)

    # flash-attn is OPTIONAL for the smoke: Gemma4 falls back to SDPA without it.
    # The custom Docker image bakes flash-attn in (prebuilt wheel) for throughput.
    if not check("flash-attn (optional)", _flash):
        print("[INFO] flash-attn absent — OK for smoke; bake it into the image for training.")

    print("\nSMOKE TEST:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
