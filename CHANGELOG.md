# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.1] — 2026-10-13

Supply-chain and community operations. No change to the training or serving code --
`train/` and `inference/` are byte-identical to 0.1.0.

### Added

- **Supply-chain CI** (`.github/workflows/supply-chain.yml`): pinned-commit drift check
  across the files that quote the third-party kernel commit, dependency advisories via
  `pip-audit` gated by an approved-exception policy, and a weekly run of the kernel
  fetch recipe against real upstream (resolve, checksums, patch applies, output parses).
- **Release SBOM** (`.github/workflows/release-sbom.yml`): source tarball, SHA-256
  checksums, SPDX and CycloneDX from one scan boundary, metadata validation, a scan
  gated by the same exception policy, and build/SBOM attestation.
- **Resolved dependency sets** — `requirements-lock.txt` (85 packages) and
  `inference/requirements-lock.txt` (49), so the SBOM describes the real surface rather
  than the single pinned declaration.
- **Community operations** (`docs/community/`): channel matrix, chat policy, moderator
  runbook, label definitions and a launch plan; `COMMUNITY.md`; a help/contribute/report
  section in the README.
- **Community automation**: a weekday alert for items with no human response for 48h,
  and an alert when `help wanted` or `good first issue` is applied. Both post to
  whichever chat webhook secret is configured and print without failing when none is.
- `.github/CODEOWNERS` over licensing, security and CI paths, with a CI check that every
  handle still resolves.
- `docs/metrics/README.md` — what response-time measurement covers, and the explicit
  commitment not to measure individuals.
- `COMPLIANCE.md` §4b — base-model and derived-checkpoint licensing.

### Changed

- Build backend requires `setuptools>=83` (PYSEC-2026-3447).
- Dependabot update PRs paused as `.github/dependabot.yml.disabled`.

### Fixed

- `release-sbom.yml` ran its tooling from the released tag, which fails for any tag
  older than the tooling. It now builds the artifact with `git archive <tag>` while the
  tooling stays on the default branch.
- CODEOWNERS handles after two owner renames; GitHub does not redirect a renamed handle
  and CODEOWNERS ignores unknown ones silently.

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

[Unreleased]: https://github.com/trillion-labs/trida-stack/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/trillion-labs/trida-stack/releases/tag/v0.1.1
[0.1.0]: https://github.com/trillion-labs/trida-stack/releases/tag/v0.1.0
