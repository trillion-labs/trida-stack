# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Typed decisions in the MLX server (`POST /v1/decide`): Zoro-1's pointer head answers choice / yes-no /
  score questions over a state (one shared state prefill, a branch per question), and below a confidence
  threshold the model reasons once (self-spec) and decides again. The server also gained a `<tool_call>`
  ban for `tool_choice: "none"` (tools stay in the prompt, so the prompt cache still matches), and two
  opt-in generation guards, off by default so evals and speedrun runs decode unguarded: a tool-call gate
  (`--tool-call-gate P`, a call only where its raw probability is at least P) and a loop guard
  (`--loop-guard`: exact and n-gram repeats; a loop while thinking closes the thinking and answers).
  `/v1/models` reports both. Also `--log-requests`.
- Zoro-1 Chat, a native macOS app for the MLX server (`inference/mlx/app`): streaming chat with decode
  stats, images, agent tools (files, a per-chat terminal, screenshots, UI control), decision-model tool
  routing, reply checks with one rewrite, `/decide`, compaction near the context limit, saved chats, and a
  self-contained bundle / DMG build.
- `trida-mlx-datamine` (the app's chats and request logs → scrubbed candidate training data: preference
  pairs, feedback, tool use, decisions, failures, trajectories) and `trida-mlx-simuser` (an LLM plays the
  user of the app to collect more trajectories, through a local, token-protected automation API).

- Image+text training for Qwen3.5 vision-language checkpoints. Conversations under
  `--pack --multiturn` may carry image parts. The vision encoder is frozen and the
  projector trains at `--projector_lr_scale` × `--lr`. Images above
  `--max_image_pixels` are downscaled, and image tokens get Qwen's 3-D RoPE
  positions. `train/tools/compose_vl_checkpoint.py` attaches the base vision tower
  to a text-only checkpoint. `shape_for_sglang.py` passes VL checkpoints through.
  The FSDP ranks stay in lockstep when a batch has no images.
- Typed-decision training in [kev](https://github.com/jaredpalmer/kev)'s scheme
  (`train/jev.py`, `--jev_data` / `--jev_only`). A pointer head on the clean causal
  stream answers typed questions (yes/no, pick one, score) over a state. kev's
  encoding, augmentation and loss are ported. It runs alone or interleaved with SFT,
  with `--jev_head_lr_scale`, `--jev_loss_weight` and `--jev_micro_per_step`.
  `train/tools/export_kev_checkpoint.py` exports a checkpoint to kev's layout for
  `kev.benchmark` / `kev.serve`.
- `--fp32_master` (fp32 weights and AdamW state over bf16 compute; needed at small
  LRs) and `--adam_beta2`.

### Changed

- Vulnerability reports have a second private channel: GitHub private vulnerability
  reporting is enabled, so the Security tab's **Report a vulnerability** button now
  works, and `SECURITY.md` names it beside the role-alias email. `community-check`
  fails if that route disappears from the document, which is what kept the address
  check honest.

- The handle checker reads every file that names people, not only `CODEOWNERS`.
  `hongjoonahn-tri` was fixed in `CODEOWNERS` two weeks ago and survived in
  `GOVERNANCE.md` -- the document that says who is responsible -- as a link that 404s,
  because the check added at the time read one file. It now covers `GOVERNANCE.md` and
  `docs/community/README.md`, and a failure names which file to edit.

- `--model_id` now defaults to `Qwen/Qwen3.5-4B` (was `Qwen/Qwen3-4B`).
- Assistant turns without reasoning now train with their empty
  `<think>\n\n</think>` block. They were previously left unsupervised.

- `--pack --multiturn` now packs conversations with buffered first-fit
  (`PackedMultiTurnRowDataset`). A conversation that doesn't fit the current row
  waits for a later one instead of being dropped; only conversations longer than
  `--max_length` are skipped. `--micro_batch_size` is again the number of rows per
  forward. **Removed:** `--pack_examples` and `--max_packed_rows`. The packer also
  enforces its two preconditions instead of failing silently: a conversation that does
  not fit a row alone, and a `buffer_size` below one, each raise instead of yielding an
  oversized row or dropping the whole stream.

- All four scheduled workflows moved off the hour and into the UTC evening, and
  their weekday fields shifted a day to compensate: 21:00 UTC is already the next
  day in Seoul, so `* * 1-5` would have run Tuesday through Saturday. Measured
  lateness was 38-40 minutes every weekday on `0 0` and 13-24 on `0 3`. The
  snapshot now runs last, so it sees that morning's results rather than last
  week's.

### Fixed

- The metrics population split is read from `docs/metrics/first-party.txt` instead
  of an API lookup. Organisation membership here is private, so neither
  `/collaborators` nor `author_association` can answer it from a workflow: both
  classified 32 of our own items as outside contributions when collected in Actions,
  while passing locally against a user token. Snapshots also record distinct author
  counts, which is what made the bug legible -- 32 external items from 1 author.
  `tools/check_first_party.py` fails CI when a listed login stops resolving.

### Security

- `filelock` floored at `>=4.2.1` and both lock files moved off 4.1.0. Six
  advisories, the worst GHSA-v4x8-3wr5-3wq8 (HIGH). It is transitive, so the floor
  goes in `requirements.txt` as well as the lock: regenerating the lock otherwise
  drops straight back to a vulnerable resolution.

- Every GitHub Action is pinned to a commit SHA instead of a moving tag, and
  `tools/check_action_pins.py` fails CI if one comes back as a tag. The repository
  pins third-party kernels to a commit and fails on drift; the workflows that
  generate the SBOM and sign its attestation were running from tags upstream can
  move. `ci.yml` also moves from checkout v4 to v6, which the rest already used.

### Changed

- `--model_id` now defaults to `Qwen/Qwen3.5-4B` (was `Qwen/Qwen3-4B`).
- Assistant turns without reasoning now train with their empty
  `<think>\n\n</think>` block. They were previously left unsupervised.
- `--pack --multiturn` now packs conversations with buffered first-fit
  (`PackedMultiTurnRowDataset`). A conversation that doesn't fit the current row
  waits for a later one instead of being dropped; only conversations longer than
  `--max_length` are skipped. `--micro_batch_size` is again the number of rows per
  forward. **Removed:** `--pack_examples` and `--max_packed_rows`.

- The metrics workflow opens a pull request instead of pushing to the default
  branch. Two reasons, and the second one is that the old version could not have
  worked: the ruleset on main requires a pull request and only an organization
  admin bypasses it, so a direct push from `github-actions[bot]` is rejected. The
  first is that a weekly bot commit does not justify standing write access to main.
  The PR body also asks for the period review to be added before merging.

- The four community duty roles are assigned across all three maintainers, each
  with a backup who is not the person holding it. Moderator and security contact
  are deliberately different people: of the four, that is the pair whose overlap
  carries the sharpest conflict, since a report about the security contact would
  otherwise be received by its subject. This closes launch-hold condition H3 and
  makes the conduct recusal in `GOVERNANCE.md` executable -- whoever a report is
  about, someone is left to handle it.

- `docs/community/README.md`: the pre-launch checklist names the three participant
  types and where each one's question is meant to end up, which was the last item on
  it that nobody could close by writing something down. Its metrics line also said
  the numbers are not kept in this repository, which stopped being true when the
  weekly snapshots landed.

- Conduct reports go to `admin@trillionlabs.co` (was `conduct@`). Both reporting
  addresses are role aliases rather than an individual, so neither depends on one
  person reading their mail. `CODE_OF_CONDUCT.md` now also states that a report about
  a maintainer should be sent anyway, because the recusal that makes that safe is
  written into `GOVERNANCE.md`.

### Added

- `inference/serve_decide.py`, `decide_engine.py`, `decide_request.py` — a decision
  server for checkpoints that carry a pointer head (`jev_head.pt`): typed decisions
  (`/v1/decide`: yes/no, choice, score, with probabilities), reason-then-decide (the
  model writes its reasoning, then decides with it in the state) and OpenAI-compatible
  chat with tool calls, all from one in-process engine and one copy of the weights. It
  runs on stock SGLang ≥ 0.5.22 (Apache-2.0) with causal decoding; the two-stream
  decode modes stay with `serve.py`. Each question becomes K+1 prefix prompts whose
  last hidden states feed the head, so the prefix cache shares the work. The request
  format follows kev's (Apache-2.0, credited in `NOTICE`). CPU tests in
  `test_decide_cpu.py`. `serve.py --mode decide` launches it (under `$DECIDE_PYTHON`).
  Chat streams (server-sent events, reasoning before content), reports `usage` and
  serves `/v1/models`, so `chat.py` and `eval.py` run against it unchanged. The HTTP
  server runs on the engine's event loop, so concurrent decisions and chat batch
  together.

- `inference/jev.py` — the Jev request encoding a Trida checkpoint's typed-decision
  head reads (`materialize`, `encode_rows`, including `image_tokens`), and
  `inference/jev_head.py`, the `PointerHead` that `jev_head.pt` loads into. The
  encoder is the training stack's, copied unchanged and kept torch-free, so a
  serving process or demo builds the same rows as the training encoder.
  `inference/test_jev_golden.py` checks the rows against golden fixtures from the
  training encoder (provenance in `inference/fixtures/jev_golden/PROVENANCE.md`).
  Derived from kev (Apache-2.0); attributed in `NOTICE` and `COMPLIANCE.md` §1.
- `scripts/export_kev_checkpoint.py` — exports a checkpoint to kev's layout. The
  Trida2.1-4B model card points here for scoring and serving with kev's tools.

- `inference/mlx/`: on-device (Apple Silicon, MLX) serving of Trida2.0-4B with lossless
  self-speculative decoding (`bd_bidir_shift` b7/g4 semantics), an OpenAI-compatible server with
  tool calls and a prompt cache, a minimal agent loop, MLX quantization, and offline tests.
- `inference/mlx/`: image input for vision-language checkpoints (Trida-2.0-4B-1006): Qwen3.5 vision
  encoder + projector, HF-identical preprocessing, multimodal RoPE, image-aware prompt cache,
  OpenAI `image_url` parts in the server.
- `metrics/` and `docs/metrics/reviews/` — the response metrics are now collected in
  the repository on a weekly snapshot, with a written review per period recording what
  was measured, what it showed and what was decided. The external series is `no_data`
  because there are no outside contributors yet; a separate internal series measures
  maintainer response time so a period with no external items still produces a number
  rather than a row of nulls. The internal series carries a count, never a rate.

### Removed

- `docs/community/LAUNCH_PLAN.md`. It was an internal execution plan — phase tables
  with owners, acceptance criteria, a rollback plan, and a readiness checklist whose
  rows still read ❌ — published where only rules and expectations belong. It had also
  started to rot: three references to a file that is not in this repository, and two
  claims that stopped being true this week. A public document that is wrong is worse
  than no public document. The rules it referenced stay in `docs/community/`.

### Added

- `docs/oss/` — project evaluation records: the licence policy (allowed /
  conditional / prohibited), a five-criteria selection scorecard, an intake register,
  and the evaluation record behind each of the eight components already adopted.
  The records postdate the decisions they describe and say so.

- `GOVERNANCE.md` — roles, how a decision is made, where decisions are recorded,
  the release procedure, and how maintainers join or step down. States plainly that
  `require_code_owner_review` is not yet enabled, so CODEOWNERS is advisory today.

### Fixed

- SGLang backend patch: self-speculative decoding corrupted requests when a new
  request was prefilled inside a running decode batch. Their outputs drifted below
  AR quality under concurrent load. The patch also stops a radix-cache lock
  underflow (`mamba_lock_ref=0`) on the same path. `inference/sglang/README.md`
  documents the remaining memory spike when a long prompt joins a sampling batch,
  sized per prompt token.

- BFCL handler (`benchmark/bfcl_v4/trida_handler.py`): every prompt silently fell
  back to the QwenFC JSON format. The handler imported `inference.bd_generate`, which
  is not in this repo, from a placeholder path. It now loads the served checkpoint's
  own chat template. It also stops adding a second `<think>` prefill and types XML
  tool-call values per test language (Python literals for Python; strings for
  Java/JavaScript, which BFCL checks as strings).
- FunctionChat runner: it resumed from a per-model-name response and grade cache, so
  new checkpoints were partly graded with earlier models' results. Each run now starts
  clean.

## [0.1.2] — 2026-10-13

A security dependency bump, and the supply-chain plumbing that v0.1.1 was cut thirty
minutes too early to include.

### Changed

- `datasets` floored at `>=5.0.1` (GHSA-379c-qx7v-6h59, path traversal in
  folder-based builders). Unpinned it resolved to 2.14.4; we do not use those
  builders, but nothing held us at an old version either.
- `SECURITY.md` states the support window: the latest release only. This is also the
  denominator of the SBOM-coverage metric, so older tags are out of scope by policy
  rather than counted as gaps.

### Fixed

- Vulnerability exceptions now carry every id a finding is known by. pip-audit leads
  with PYSEC and grype with GHSA, so an exception keyed to one silently failed to
  cover the other -- all eight approved exceptions were blocking the release.
- The release workflow no longer passes `--clobber`. Replacing an SBOM already
  published against a tag rewrites what consumers were told it contains; the upload
  now fails if the tag already has assets.

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

[Unreleased]: https://github.com/trillion-labs/trida-stack/compare/v0.1.2...HEAD
[0.1.2]: https://github.com/trillion-labs/trida-stack/releases/tag/v0.1.2
[0.1.1]: https://github.com/trillion-labs/trida-stack/releases/tag/v0.1.1
[0.1.0]: https://github.com/trillion-labs/trida-stack/releases/tag/v0.1.0
