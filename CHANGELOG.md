# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- MLX server: `tool_choice: "none"` now keeps the tool schemas in the prompt and bans `<tool_call>`
  (before, the tools were dropped from the prompt), so clients sending it (evals, Hermes) get a different
  prompt than earlier baselines.

### Added

- Zoro-1 Chat 0.6.0: **a personal knowledge graph** learned from chats (`inference/zoro-1/mlx/app/Knowledge`,
  a standalone Swift package with its own `kg-selftest`).
  - **Learning:** after each reply, Zoro-1 decides whether there is something lasting to remember and
    proposes `subject —relation→ object` facts from your messages, grounded web answers and the documents
    it read. Web and document facts need a quote that is really in what was read. Each relation has types
    for its two ends: a fact that fits only the other way round is turned around, and one that fits neither
    way is dropped. Zoro-1 also types new entries (a Role type for job titles) and links names to existing
    entries.
  - **Provenance:** each fact records when and how it was added, the chat, the model, the source page or
    file with its quote, your message, and a history of changes. Functional relations end the older fact
    instead of deleting it.
  - **`/update-kg [focus]`:** learns from the whole chat (every turn, page and document), without the gate.
  - **Using it:** the facts relevant to a message go with it, fixed when the message is sent so the prompt
    cache still matches; `recall`, `remember` and `forget` tools.
  - **Knowledge window** (⇧⌘K): Focus, All (every entry and group, isolated ones too, laid out by force;
    drag, pinch, Fit) and Relations views. Facts are arrows from subject to object, dashed once ended.
    An inspector edits entries and facts. A chip under each reply lists what it added, with Undo.
  - **Brain button** in the sidebar: a pixel brain with a graph on it that shows learning, new facts and
    relations waiting for review.
- Zoro-1 Chat:
  - **Web browsing:** `fetch_url` returns a page outline (title, sections, links, start of the text), and
    `find_in_page`, `read_section` and `open_link` navigate the page. The system prompt carries today's
    date and asks for answers quoted from the page, and a "backed by the pages" reply check runs after web
    tools.
  - **Auto and manual approvals:** auto mode runs writes and commands without asking. Risky patterns always
    ask, and Zoro-1 judges the rest (P(safe) ≥ 0.8 runs).
  - **Code viewer:** a right-hand panel with each written file's diff or current contents.
  - **Scratch folder:** each chat gets a working folder that the model writes to without approval.
    Long tool outputs are saved there with a preview and a path, `read_file` reads slices with `offset`,
    and the toolbar's *Scratch* button lists the files.
  - **Decision labels:** each reply keeps its typed decisions. You mark the right answers, and
    `trida-mlx-datamine` writes them to `decision_labels.jsonl` as kev records (`--jev_data`).
  - **`read_pdf`:** reads PDF text, with scanned pages as images.
  - **Undo for `write_file`.**
  - **Chats:** search and rename, and a context meter next to the composer.
  - **`locate_files`:** a folder step for folders over 480 files, and an on-disk index of file summaries.
  - **Agent mode is always on;** the toggle and `/agent` are gone.
  - **Mascot:** a bigger panel; tool props and speed lines; tidying during compaction; numbers under the
    caption; reactions to feedback, edits and clicks; listening and dozing; trainer levels with
    accessories and badges; time-of-day touches; the mascot as the app icon and a live Dock icon.
- Zoro-1 Chat:
  - **Look:** a forest-and-sand theme for light and dark mode, sand bubbles for the user, and serif
    Markdown headings.
  - **Reply footer:** one line under each reply, with a self-spec draft meter and a check seal.
  - **Sidebar:** chat rows with a green selection.
  - **Mascot:** an animated pixel figure in the sidebar, showing the model's state (reading, thinking,
    deciding, using a tool, answering, checking, done, error, waiting).
- Zoro-1 Chat renders replies from Markdown natively: headings, nested and task lists, quotes,
  aligned tables, rules, code blocks with a language label and a copy button, and display math shown as
  LaTeX source. The parser is `Markdown.parse` in TridaChatCore.
- `inference/zoro-1/mlx/filesearch/`: an offline eval of file search with the decision model
  (DecisionRAG-style, arXiv:2610.35868): 35 questions with known answer files over a pinned snapshot of
  this repository, comparing BM25, per-file yes/no, per-batch choice and tree navigation.
- Zoro-1 Chat: a `locate_files` decision tool that finds files by meaning in any language. It ranks
  every file's opening lines with per-batch choice questions, plus keyword search's top 10.
- Shared SGLang/MLX decision requests and answer formatting, torch-free pointer-head
  conversion to safetensors, bounded state reuse, and parity/HTTP regression tests (#61).

- Typed decisions in the MLX server (`POST /v1/decide`): Zoro-1's pointer head answers choice / yes-no /
  score questions over a state (one shared state prefill, a branch per question), and below a confidence
  threshold the model reasons once (self-spec) and decides again. The server also gained a `<tool_call>`
  ban for `tool_choice: "none"` (tools stay in the prompt, so the prompt cache still matches), and two
  opt-in generation guards, off by default so evals and speedrun runs decode unguarded: a tool-call gate
  (`--tool-call-gate P`, a call only where its raw probability is at least P) and a loop guard
  (`--loop-guard`: exact and n-gram repeats; a loop while thinking closes the thinking and answers).
  `/v1/models` reports both. Also `--log-requests` (old logs pruned at start, `--log-requests-keep-days`,
  default 7), `--mlx-cache-limit-mb`, and `memory_mb` in `/v1/models`. A reply the loop guard cuts short
  ends with `finish_reason: "length"`.
- Zoro-1 Chat, a native macOS app for the MLX server (`inference/zoro-1/mlx/app`): streaming chat with decode
  stats, images, agent tools (files, a per-chat terminal, screenshots, UI control), decision-model tool
  routing, reply checks with one rewrite, `/decide`, compaction near the context limit, saved chats, and a
  self-contained bundle / DMG build.
- `trida-mlx-datamine` (the app's chats and request logs → scrubbed candidate training data: preference
  pairs, feedback, tool use, decisions, failures, trajectories) and `trida-mlx-simuser` (an LLM plays the
  user of the app to collect more trajectories, through a local, token-protected automation API).

- `README.ko.md`: a Korean README, linked from the English one (#57).
- README: a Models table with Hugging Face links for Zoro-1 and Zoro (formerly Trida2.0-4B), and
  sections on typed-decision joint training, image training, `--mode decide` serving, the backends
  (SGLang, vLLM, MLX) and the two getting-started notebooks (#57).
- Image+text training for Qwen3.5 vision-language checkpoints. Conversations under
  `--pack --multiturn` may carry image parts. The vision encoder is frozen and the
  projector trains at `--projector_lr_scale` × `--lr`. Images above
  `--max_image_pixels` are downscaled, and image tokens get Qwen's 3-D RoPE
  positions. `train/tools/compose_vl_checkpoint.py` attaches the base vision tower
  to a text-only checkpoint. `shape_for_sglang.py` passes VL checkpoints through.
  The FSDP ranks stay in lockstep when a batch has no images.
- Typed-decision training in [kev](https://github.com/jaredpalmer/kev)'s scheme
  (`train/data/decision.py`, `train/models/decision.py`, `--decision_data` / `--decision_only`). A pointer head on the clean causal
  stream answers typed questions (yes/no, pick one, score) over a state. kev's
  encoding, augmentation and loss are ported. It runs alone or interleaved with SFT,
  with `--decision_head_lr_scale`, `--decision_loss_weight` and `--decision_micro_per_step`.
  `train/tools/export_kev_checkpoint.py` exports a checkpoint to kev's layout for
  `kev.benchmark` / `kev.serve`.
- `--fp32_master` (fp32 weights and AdamW state over bf16 compute; needed at small
  LRs) and `--adam_beta2`.

### Changed

- Zoro-1 Chat: agent mode is always on, and its toolbar toggle, setting and `/agent` command are gone.
  With it off, the model still wrote tool calls as text that nothing could run; tool routing already
  keeps tools out of replies that don't need them.
- `trida-mlx-server`: without `--served-model-name`, the model id is the model folder or repo name,
  lowercased (e.g. `zoro-1-mlx-q8`), instead of always `trida2.0-4b`. Zoro-1 Chat names a server it
  did not start by the folder that server loaded.
- Zoro-1 Chat reply review is one direct decision, with no reasoning pass. P(yes) below 0.2 fails and
  triggers the rewrite; 0.2–0.5 is shown as unsure and changes nothing. "Claims match results" runs only
  after an action tool or a failed or refused call, and is split into two questions: does the reply
  claim an action, and did the tools show it succeeded.
- MLX `POST /v1/decide` now defaults to `reasoning: false` instead of `"auto"`.
  Send `reasoning: "auto"` explicitly to retain confidence-based escalation. Both
  `true` and `"auto"` default to self-spec reasoning; `trida_mode: "causal"` overrides it.
- **nano-inference is now one stack per model.** `inference/` moved to `inference/zoro-1/` and serves
  only `trillionlabs/Zoro-1`:
  - the default model is `trillionlabs/Zoro-1` (gated) instead of `trillionlabs/Trida2.0-4B`
  - `self-spec` uses the model card's setting, `configs/zoro1_self_spec_b15_g8.yaml` (generation
    block 8, top-k 20), and `eval.py` defaults to top-k 20 to match
  - `--mode decide` defaults to the same checkpoint, since Zoro-1 carries `jev_head.pt`
  - `inference/` is no longer a Python package (`pyproject.toml` packages `train` only); the scripts
    keep their sibling imports, and paths across docs, CI and tooling point at `inference/zoro-1/`

- Training orchestration is split into model construction, distributed setup/sharding,
  data loaders, optimizer and checkpointing modules. Training-loop functions stay
  above `main()` in `train/train.py`; `main()` shows the stages directly. Model implementations live in `train/models/`;
  existing model import paths remain available.

- **Breaking:** typed-decision training now uses only `--decision_*` CLI flags,
  Python model arguments, and metrics. W&B config keys use `decision_*` too;
  update external job scripts, sweeps, and dashboard filters that use `jev_*`.
  Data/encoding lives in `train/data/decision.py`, head/loss in
  `train/models/decision.py`. Only the real pre-refactor imports (`train.jev` and
  `train.hf_block_diffusion*`) remain as shims until **0.2.0**. Checkpoint names,
  `jev_head.*` parameter keys and optimizer group order are unchanged.

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

### Removed

- `serve.py --mode diffusion` (Zoro-1 does not support pure diffusion decoding) and the third-party
  `sdar`, `llada2-0`, `llada2-1-speed` and `llada2-1-quality` modes with their configs.

### Fixed

- Zoro-1 Chat:
  - **Prompt cache across messages:** the tool router's empty reply placeholder was sent as an empty
    assistant turn, so each new message re-read the whole chat (about 80 s at 42k tokens).
  - **Agent loop:** the last step answers without tools instead of stopping at "Stopped after N agent
    steps". A call repeated in the same turn, or a URL already opened earlier in the chat, is not run again.
    After five web calls, the results say how many steps are left.
  - **Searches:** a search that uses none of the user's words gets a note to search with their words as
    written, since names were being romanized wrongly. `fetch_url` suggests Naver and namu.wiki for Korean
    subjects.
  - **Tool-call gate:** when the gate held back a call the model meant to make and the reply degenerated
    into `|` until the loop guard, the step is retried once with the gate relaxed. Saving to the knowledge
    graph now routes to the memory tools.
- `inference/zoro-1/mlx`: tool calls written without the `<tool_call>` wrapper (bare `<function=…>` blocks after a first wrapped call) are now parsed instead of dropped. A stray `</tool_call>` no longer shows up in the reply.
- Metrics snapshots are labelled with the ISO week in KST. The weekly run moved to Monday 07:17 KST
  (#54), which is still Sunday in UTC, so labelling by the UTC date named the first post-change run
  2026-W44 again and its PR (#71) would have overwritten the reviewed W44 snapshot. Every existing
  label is unchanged; that run is 2026-W45.
- The metrics population split is read from `docs/metrics/first-party.txt` instead
  of an API lookup. Organisation membership here is private, so neither
  `/collaborators` nor `author_association` can answer it from a workflow: both
  classified 32 of our own items as outside contributions when collected in Actions,
  while passing locally against a user token. Snapshots also record distinct author
  counts, which is what made the bug legible -- 32 external items from 1 author.
  `tools/check_first_party.py` fails CI when a listed login stops resolving.

### Security

- Decision requests never name files. `decide_request.parse` (both copies, SGLang and MLX) takes an image only
  in memory: a PIL image, base64 or a data URL. Reading a file moved to `decide_request.open_image(path)`,
  which Python callers call themselves, and the `allow_paths` flag every HTTP entry point had to remember to
  turn off is gone. Before, `/v1/decide` was safe only because both servers passed `allow_paths=False`
  (CodeQL: uncontrolled data used in a path expression).

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
