# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- `inference/mlx/`: on-device (Apple Silicon, MLX) serving of Trida2.0-4B with lossless
  self-speculative decoding (`bd_bidir_shift` b7/g4 semantics), an OpenAI-compatible server with
  tool calls and a prompt cache, a minimal agent loop, and MLX quantization.

## [0.1.0] — 2026-09-30

Initial public release of `trida-stack` — the training and serving stack for
Trillion Labs' **Trida** two-stream block-diffusion language models
(Qwen3 / Qwen3.5-based), under the Apache License 2.0.

### Added

- **Two-stream block-diffusion training stack** (`train/`): AR → diffusion SFT
  (Fast-dLLM v2 style) that converts a pretrained autoregressive checkpoint into
  a block-diffusion model, with an AR auxiliary loss that keeps a usable causal
  head. Native-torch FSDP2 (no DeepSpeed); packing, length/response bucketing,
  multi-turn supervision, activation offload, and fused cross-entropy. Supports
  dense Qwen3 (full attention) and hybrid Qwen3.5 (gated-delta linear + periodic
  full attention) paths. Saved checkpoints are stock HF causal LMs plus a
  `<|mask|>` token and a `block_diffusion.json` sidecar.
- **nano-inference serving stack** (`inference/`): a thin, nanoGPT-style wrapper
  (`serve.py` / `chat.py` / `eval.py`) that serves one checkpoint in several
  decode modes — causal, block-diffusion, and lossless self-speculative — behind
  an OpenAI-compatible API, plus per-family decode configs and quick quality /
  throughput benchmarks (gsm8k / mmlu_pro / ifeval).
- **SGLang serving recipe**: true block-diffusion (bidirectional-within-block
  attention, variable-length commits) and the self-speculative path.
- **vLLM serving backend** (`inference/vllm/`): out-of-tree plugin
  (`VLLM_PLUGINS=trida_diffusion`) for block-diffusion and self-spec / AR-Trust
  decoding, with a CPU correctness test for the two-stream GDN decode crux.
- **Benchmark harness wrappers** (`benchmark/`): first-party register/run drivers
  and serving glue for BFCL, tau2-bench, FunctionChat, KoAgentBench, SWE-bench,
  Terminal-Bench, and lm-evaluation-harness (the harnesses themselves install
  separately).
- Community and compliance docs: `README.md`, `COMPLIANCE.md`, `NOTICE`,
  `LICENSE`, `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, issue / PR
  templates, `CITATION.cff`, and the dataset catalog.
- **Setup docs for the one-time third-party kernel fetch** in the root, `train/`,
  `inference/` and `inference/vllm/` READMEs — the fetched kernels are
  PolyForm-Noncommercial and not vendored, so a fresh clone cannot build the
  two-stream training or diffusion-serving paths without running the recipe.

### Notes

- The two-stream Gated-DeltaNet + block-diffusion Triton kernels are **not
  bundled**. They are fetched/patched via recipe
  (`train/block_gated_delta_rule/fetch_kernels.sh`, `inference/sglang/`) and are
  **PolyForm Noncommercial 1.0.0** — the two-stream training and diffusion-serving
  paths are **noncommercial-only**. See [`COMPLIANCE.md`](COMPLIANCE.md) §1b.

[Unreleased]: https://github.com/trillion-labs/trida-stack/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/trillion-labs/trida-stack/releases/tag/v0.1.0
