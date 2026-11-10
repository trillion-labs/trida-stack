"""vLLM general-plugin entrypoint for the Trida block-diffusion port.

This module exposes a single callable, ``register_trida``, wired to the
``vllm.general_plugins`` entry-point group (see pyproject.toml). vLLM's
``load_general_plugins`` runs every general plugin in EVERY process
(process0, the EngineCore subprocess, and each worker), which is exactly
what we need: ``ModelRegistry.register_model`` must run inside the
EngineCore subprocess, not just the launcher.

It does two things:
  1. registers ``Qwen3_5ForBlockDiffusion`` in the ModelRegistry, and
  2. injects a ``MODELS_CONFIG_MAP`` entry for that architecture so vLLM's
     per-arch ``verify_and_update_config`` runs — delegating to the base
     ``Qwen3_5ForCausalLM`` config hook (GDN/hybrid backbone setup) and
     then applying diffusion-specific defaults (mixed causal/bidirectional
     attention backend, canvas_length -> DiffusionConfig).

Designed to be idempotent (safe to call multiple times / in every process).
"""
from __future__ import annotations

import os


def _register_model() -> None:
    from vllm import ModelRegistry

    # register_model is idempotent for the same target; re-registering the
    # same (name, path) is a no-op-ish overwrite.
    ModelRegistry.register_model(
        "Qwen3_5ForBlockDiffusion",
        "vllm_native_diffusion.qwen3_5_diffusion:Qwen3_5ForBlockDiffusion",
    )


def _register_config_hook() -> None:
    """Add a MODELS_CONFIG_MAP entry for our architecture.

    Without this, vLLM skips the per-arch config hook for an unknown arch
    name. is_hybrid still triggers the GDN/mamba hybrid setup automatically,
    but we also want the same attention-backend / diffusion defaulting that
    the in-tree DiffusionGemma arch gets. We delegate to the base Qwen3.5
    ForCausalLM config for the backbone, then apply diffusion defaults.
    """
    from vllm.model_executor.models import config as _cfg
    from vllm.model_executor.models.config import (
        MODELS_CONFIG_MAP,
        Qwen3_5ForCausalLMConfig,
        VerifyAndUpdateConfig,
    )

    class Qwen3_5ForBlockDiffusionConfig(VerifyAndUpdateConfig):
        @staticmethod
        def verify_and_update_config(vllm_config) -> None:
            # 1) Backbone setup identical to the AR path (GDN, hybrid KV, etc.).
            Qwen3_5ForCausalLMConfig.verify_and_update_config(vllm_config)

            # 2) Diffusion sampler materializes [num_seqs, canvas, vocab] fp32
            #    transients; cap concurrency the way DiffusionGemma does.
            from vllm.config.scheduler import SchedulerConfig

            sc = vllm_config.scheduler_config
            if (
                sc is not None
                and sc.max_num_seqs >= SchedulerConfig.DEFAULT_MAX_NUM_SEQS
            ):
                sc.max_num_seqs = 8

            # 3) Mixed causal/bidirectional within a batch: keep FlashInfer out
            #    of auto-selection (mirrors DiffusionGemma's handling).
            ac = getattr(vllm_config, "attention_config", None)
            if ac is not None and getattr(ac, "backend", None) is None:
                if not getattr(ac, "use_non_causal", False):
                    ac.use_non_causal = True

            # 4) Auto-create DiffusionConfig from hf_config.canvas_length if the
            #    user didn't pass --diffusion-config.
            if vllm_config.diffusion_config is None:
                from vllm.config.diffusion import DiffusionConfig

                hf_config = vllm_config.model_config.hf_config
                canvas_length = getattr(hf_config, "canvas_length", None)
                if canvas_length is not None:
                    vllm_config.diffusion_config = DiffusionConfig(
                        canvas_length=canvas_length,
                    )

        @staticmethod
        def verify_and_update_model_config(model_config) -> None:
            Qwen3_5ForCausalLMConfig.verify_and_update_model_config(model_config)

    MODELS_CONFIG_MAP["Qwen3_5ForBlockDiffusion"] = Qwen3_5ForBlockDiffusionConfig
    # keep a module handle for debugging
    _cfg.Qwen3_5ForBlockDiffusionConfig = Qwen3_5ForBlockDiffusionConfig


def register_trida() -> None:
    tag = f"[trida-plugin pid={os.getpid()}]"
    try:
        _register_model()
        _register_config_hook()
        # Install the GDN block-END readout monkeypatch (FLARE two-stream) onto
        # the vLLM GDN layer. Must run in EVERY process (esp. the EngineCore /
        # worker) so the patched `_forward_core` is used at inference time.
        from vllm_native_diffusion.qwen3_5_diffusion import (
            _install_gdn_block_readout, _install_spec_builder_shim, _install_bidir_canvas,
        )
        _install_gdn_block_readout()
        _install_spec_builder_shim()   # no-op unless TRIDA_SELFSPEC_N is set (spec-shape self-spec)
        _install_bidir_canvas()        # E3: no-op unless TRIDA_SS_BIDIR=1
        if os.environ.get("TRIDA_SELFSPEC_N"):
            # Self-spec verifies greedily against the AR logits; per-request sampling params are
            # ignored, so accept the temperature / seed fields that OpenAI-style agentic harnesses
            # (FunctionChat 0.1, Ko-AgentBench 0.7) always send instead of rejecting the request.
            from vllm.sampling_params import SamplingParams
            SamplingParams._validate_diffusion = lambda self, model_config: None
        print(f"{tag} registered Qwen3_5ForBlockDiffusion + config hook + GDN readout patch", flush=True)
    except Exception as e:  # pragma: no cover - visibility in subprocess logs
        print(f"{tag} FAILED to register: {e!r}", flush=True)
        raise
