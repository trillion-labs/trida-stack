# Review validation

## System One integration and H100 rerun (2026-11-16)

Integrated the maintainer's six commits through `8fdf816` and `origin/main`
`75d07f7` (including #94). The adapter now posts `{model, state, questions}` to
`/v1/systemone`; it preserves response usage and reasoning metadata, and still
rejects unfinished, empty or trimmed reasoning without normalizing probabilities.
The upstream `typesafe` provider does not perform these strict completion checks,
so the small transport adapter remains. Questions, states, policies, references,
dataset revisions and scoring code are unchanged. The probability validation
formula and tolerance are unchanged.

`cut` remains a closed set: `length` / `empty` / `state_limit`. All unfinished
passes use `length`; `reasoning_stats.finish_reason` retains `abort`, `unknown`,
or the actual runtime reason. Both CUDA and MLX now default an omitted reasoning
budget to the configured cap (4,096 by default), matching the direction of #97.
The benchmark's explicit 2,048-token default and 8,192-token ceiling are unchanged.

**H100:** job 5341 on `diffusion-compute-001`, one H100 80 GB, stock SGLang
0.5.23 / torch 2.14.1+cu130, BF16, the same local checkpoint used in the earlier
runs: `/mnt/vast/trillion/hyungguk/trida-jev-eval/runs/rl-head-i11/iter_300`.
The head SHA-256 remains
`1e406f44fd66191cc16bbddf02213182a0e8403ee98ae971fedf8f72a294d9db`.
This is a local checkpoint run, not an independently downloaded release check.

- HTTP: all three answer types matched the deprecated alias exactly (after the
  documented score-legend representation change); the alias retained its
  deprecation header. Missing model, empty questions and budget 8,193 returned
  System One's 422 `detail` list. A one-token reasoning request returned
  `cut: "length"`, direct fallback answers, and `usage.output_tokens: 1`.
- Actual benchmark: pinned invoice dataset
  `6beeb2d2acd65c086c835022f5f4d7434114cafc`, complete `ap_00111`, all policies,
  per-question mode, one worker. Context 32,768, row limit 16,384, state limit
  15,744 (the required 640-token gap), explicit reasoning budget/cap 8,192.
  The previous 16,384/16,384 server split is now rejected; the original benchmark
  state and questions were not shortened. Every outgoing state and question was
  compared with the pinned upstream builders before inference. The nine saved
  requests from the earlier 8K run also match exactly in state, question and budget.
- Direct: all 21 calls succeeded in 6.46 s. Reasoning: all 21 succeeded in 53.60 s,
  each with `finish_reason: "stop"`, no `cut`, no trimming and reasoned answers.
  Previously looping `price_basis` completed in 1,697 tokens this time. Sampling
  settings were unchanged; this single successful rerun does not establish
  repeatability or resolve the model's earlier looping behavior.
- Consensus agreement: exact action-set 0/3, primary action 3/3 in both modes.
  Reasoning exact agreement was also 0/3 against OpenAI labels and 2/3 against
  Anthropic labels. This is a valid scored reasoning case, not a full evaluation
  or a claim of benchmark-wide accuracy.

Original state/question fingerprint (sorted JSON, SHA-256):
`b235a449c205bed06c585c175ea4659798c0bfcae7d5a3eba44d9ef8c624a526`.
Raw requests, responses, scores, source/head hashes and logs remain outside Git at
`/mnt/nvme_cache/trillion/wonsuk/pr89-systemone-20261116/`.

**Regression checks:** clean Linux Python 3.10 / CPU torch (CI-style): **589 passed,
69 skipped**; adapter/setup: **21 passed**; Apple Silicon MLX decision/server:
**56 passed, 1 skipped**; focused CPU decision tests: **117 passed**. Ruff, shell
syntax, shared-request copy and supply-chain pin checks pass. Initial local full
suite failures were macOS's old Bash (`mapfile` unavailable); an exploratory run
inside the SGLang environment also failed GravityOCR runtime-specific tests. The
clean Linux CPU environment above is the full regression result; no production
runtime packages were altered to make it pass.


## 8K reasoning / 16K decision-input run (2026-11-13)

Revision `d10db54` adds opt-in CUDA server limits while retaining the original
defaults. Local checks passed: **187 CPU tests (2 skipped), 20 adapter/setup tests,
43 MLX decision tests**, lint, action pins, and supply-chain checks.

Ran original case `ap_00111` with per-question mode, one worker, BF16 on one H100,
the same local checkpoint as below, context 32,768, reasoning budget/ceiling 8,192,
and decision-state/row limits both 16,384. All 21 questions and three policies
were selected; actual submitted states and questions were checked for exact
equality with the original benchmark inputs. Strict adapter checks stayed enabled.

**Still no valid scored reasoning case.** Eight of nine submitted questions
completed with `finish_reason: stop`, no trimming/cut, and `answers_source:
reasoned`. `price_basis` consumed all 8,192 reasoning tokens and returned
`cut: length` with direct fallback answers. The adapter rejected that response,
and upstream cancelled remaining queued questions after the failure. Case time
was 63.26 seconds, excluding server startup. Zero agreement scores reflect the
failed execution rather than answer accuracy.

A token-level audit confirmed complete state and reasoning retention for all
eight accepted responses. The rejected partial reasoning would occupy a
13,231-token decision row, below 16,384; this is a hypothetical encoding, since
the server correctly used direct answers for the unfinished generation. Its
37,193-character trace repeatedly revisited the price-basis interpretation before
the budget ran out. This run was blocked by generation not finishing, not by the
new decision-row limit. Raw requests, responses, scores, and audit were retained
outside Git. The test server and allocation were stopped afterward.

Earlier per-question attempts with the original decision limits also failed:
2,048 tokens accepted 11/12 submitted calls before `line_0_scope` hit the budget;
4,096 accepted 8/9 before `price_basis` hit the budget. Neither completed all 21.

## 32K real-model run (2026-11-13)

Ran revision `bfc550b` on one H100 80GB with stock SGLang 0.5.23,
torch 2.14.1+cu130, BF16 weights, server context 32,768, and the same local
`rl-head-i11/iter_300` checkpoint used in Guks's test. Its `jev_head.pt` SHA-256 is
`1e406f44fd66191cc16bbddf02213182a0e8403ee98ae971fedf8f72a294d9db`.
This identifies the tested local copy; the full released backbone was not
independently compared with Hugging Face.

Inputs, questions, policies, saved references, and upstream scoring were unchanged.
The normal wrapper ran the first four invoice cases at a 2,048-token reasoning
budget. Two additional smoke cases were selected through upstream's existing
`plan(case_ids=...)` API for their shorter state or fewer questions, not to estimate
benchmark accuracy.

| Cases | Reasoning budget | Result |
|---|---|---|
| `ap_00000`, `ap_00001`, `ap_00003` | 2,048 | HTTP 400: decision rows still exceed 8,192 tokens |
| `ap_00002` | 2,048 | Context check passed; adapter rejected `cut: "length"` |
| `ap_00271` | 2,048 | Adapter rejected `cut: "length"` |
| `ap_00271` | 4,096 | HTTP 400: reasoning-added state makes `line_0_scope` exceed the row limit |
| `ap_00111` | 2,048 and 4,096 | Adapter rejected `cut: "length"` at both budgets |

**No case produced a valid completed reasoning result.** Upstream counts errors
as wrong and emitted zero agreement values; these reflect execution failures,
not evidence of answer quality. The larger context resolves the observed prompt-budget refusal, but does
not establish a successful reasoning evaluation within the remaining limits.
An earlier parallel smoke attempt was interrupted by test-server cleanup; its
transport error is excluded above and the case was rerun in its own allocation.
All three test allocations and their server processes were stopped afterward.

## H100 results and context guidance (documented 2026-11-13)

[Guks reported CUDA results on 2026-11-12](https://github.com/trillion-labs/trida-stack/pull/89#issuecomment-6815251583)
at `4be0443`: one H100, stock SGLang 0.5.23, torch 2.14.1+cu130, and a local
Zoro-1 checkpoint whose `jev_head.pt` matched the release. All seven API checks
passed, including reasoning completion, length/state-limit fallbacks, trimming,
and invalid budgets. These are reviewer-reported results, not a local rerun.

For the first four invoice cases, direct mode ran `ap_00002`; the other three
exceeded decision-row limits. Reasoning mode ran none: the same three failed,
and `ap_00002` needed a 21,306-token prompt plus a 2,048-token reasoning budget,
exceeding the default 16,384-token context.

The README and wrapper help now recommend server `--context-length 32768` for
invoice reasoning. This fits that observed prompt plus budget, keeps benchmark
inputs unchanged, and does not change the 7,551-text-token state / 8,192-token
decision-row limits. A successful scored real-model reasoning run with the larger
context remains unverified. (The licence exception and the #83 fixes were settled later;
see "Maintainer review (2026-11-16)" below.)

## Review fixes (2026-11-12)

Rebased onto `origin/main` (`a02da55`, including #88).

- Full CPU suite: **180 passed, 2 skipped** (transformers 5.x).
- Offline adapter/setup suite: **20 passed**.
- MLX decision and decision-extension suites: **43 passed**, using tiny random
  checkpoints, including PyTorch head parity. This is not a Zoro-1 model evaluation.
- Lint, shell syntax, action pins, and supply-chain checks passed.

CUDA and MLX now share reasoning-state fitting. CPU tests cover states already at
the 7,551-text-token cap, partial-tail fitting, image-token reservations, and the
actual encoded rows retaining the conclusion. Both servers return direct answers
with machine-readable `cut` values when reasoning is unfinished, empty, or cannot
fit. The adapter rejects these fallbacks and `reasoning_trimmed: true` responses.
The earlier empty-answers-on-truncation behavior described below is historical.

Intake IDs are now 0011–0015, leaving 0009–0010 to #72. The proposed local opt-in
for unverified datasets was later approved as a maintainer exception (2026-11-16, below).

CUDA/Zoro-1 GPU inference remains pending; no GPU node is available. Luke's seven
announced #83 commits were not available on #83's frozen head (`e0a8c60`) or the
replacement branch when checked (compared and settled on 2026-11-16, below).

## Rebase validation (2026-11-11)

Rebased onto `origin/main` (`dcd5163`), preserving the `inference/zoro-1/` layout.
Full CPU suite: **144 passed, 2 skipped**. Adapter/setup suite: **18 passed**.
Lint, shell syntax, action pins, and supply-chain checks passed. No GPU inference
was rerun for this rebase.

## Tests rerun on 2026-11-10

- Full CPU suite: **119 passed, 2 skipped** (transformers 5.x).
- Offline adapter/setup suite: **18 passed**.
- Ruff, shell syntax, action pins, and supply-chain pin checks passed.

The CPU tests exercise the actual `DecideEngine.adecide` and HTTP handler with a fake
SGLang generation backend. They cover normal completion, a real `finish_reason:
length`, rejection before generation for excessive budgets, and prompt/context
accounting. On truncation, the response contains `cut: "length"`, empty `answers`,
and `reasoning_stats.finish_reason: "length"`. The adapter rejects this response,
as well as missing completion metadata. CUDA inference was not run.

## Original Q8 reasoning pilot (2026-11-06)

The server was **MLX**, not `serve_decide`: commit
`7ef348e69e36811655d0bc83f03b4498fa1d667c` from the MLX integration worktree.
That version accepts the `{"request": ...}` envelope at `/v1/decide`.
The checkpoint was the locally converted Q8 version of
`trillionlabs/Zoro-1@ab05e6c025deb0eed102f42b736dc1589848e6c9`.

The previously audited customer-service case was
`access-2fa-old-number-flight/t0`, node `triage`, with a 2,048-token budget:

- `reasoning_stats.finish_reason` was `"stop"`, not `"length"`.
- No `cut` marker; `reasoned.answers` contained 11 answers.
- The returned reasoning text contained 5,208 characters.
- All three policy action sets matched the saved consensus references; 41.21 seconds.

These are the audit values recorded during the run, not a fresh replay. The full
pilot payload is no longer available in the temporary checkout. The terminal token
was not retained, so these values establish a reported normal stop, not independent
proof of a literal `</think>` token. This pilot did not validate CUDA truncation;
the fake-backend tests above now cover that missing path. Customer-service replay now requires explicit `--allow-unverified-datasets`
consent for local evaluation; its license remains unverified.

## Invoice failures

Reproduced on CPU on 2026-11-10 using the exact historical encoder revision above,
the pinned Zoro-1 tokenizer, and the pinned invoice workflow/data. No model was
loaded. These failures occur during encoding, before inference:

```text
ap_00000 RequestError: question 'line_0_kind' is too long: 719 tokens of instructions and options with a 7552-token state (at most 8192 per question); shorten it
ap_00001 RequestError: question 'line_0_scope' is too long: 1482 tokens of instructions and options with a 7552-token state (at most 8192 per question); shorten it
```

No inputs were shortened. The invoice dataset is Apache-2.0; license evidence and
the maintainer exception for the other three datasets are in the
[intake record](../../docs/oss/intake/0011-workflowevals.md).

## Row-budget review follow-up (2026-11-13)

Reasoning fitting now reserves the longest question's encoded suffix, including
options and delimiters, before selecting a conclusion-preserving reasoning tail.
Regression tests cover a 3,000-character state and 1,200-character question, with
and without image tokens, and the HTTP response: successful trimmed reasoning
instead of HTTP 400. Strict benchmark checks still reject that trimmed response.
Server configuration requires at least the default 640-token gap between state
and row caps; the longer-input example now uses 15,744 / 16,384. Questions larger
than the gap may need a lower state cap. Existing H100 observations above retain
the original settings; they have not been rerun or reinterpreted as passing scores.

MLX now enforces the configured reasoning cap (default 4,096, opt-in up to 8,192),
including auto mode. Local verification: 191 CPU tests passed (2 skipped), 20
adapter/setup tests and 55 MLX decision/server tests passed (1 skipped), plus Ruff.
Benchmark inputs, reference answers, and strict completion checks are unchanged.
License-exception approval and the disposition of Luke's seven #83 commits were
maintainer decisions at the time; both are settled in the next section.

## Maintainer review (2026-11-16)

**Licence exception approved.** Luke Kim (@usik-luke-trl, maintainer) approved local
evaluation of the three datasets without a stated licence (customer service, security
incidents, agent-trace observability) behind `--allow-unverified-datasets`. Scope: the
pinned revisions are downloaded to the user's own cache only when the flag is passed;
nothing is committed, bundled or redistributed; their licence stays recorded as
unverified (미확인). Re-check when upstream states a licence, or by 2027-05-16. The
record is in [intake 0011](../../docs/oss/intake/0011-workflowevals.md) and both
registers.

**The seven #83 commits, compared with this branch:**

| #83 commit | What it fixed | Here |
|---|---|---|
| `dd5e515` | an aborted reasoning pass was decided on as if it had finished | ported in `7696ad6`: any finish other than `stop` is a cut with direct answers |
| `11c73c4` | long reasoning lost its conclusion to the encoder's state cut | covered by `fit_reasoning` in the shared request contract (keeps the end, reports `reasoning_trimmed`) |
| `1abe57f` | CI token write-capable and persisted while setup runs third-party code | ported in `12763e6` (`permissions: contents: read`, `persist-credentials: false`) |
| `f0c884f` | `run.sh` arguments could override the pinned dataset revision and model | covered: `run.sh` refuses `--dataset-revision` and requires `--model trida:…` |
| `a6a3869` | empty reasoning reported as a misleading "did not complete: stop" | covered: `cut: "empty"` and the adapter's message |
| `fb41554` | the reasoning budget was validated even without reasoning | covered: only the reasoning pass checks the budget |
| `3165c47` | no CHANGELOG / COMPLIANCE entries | ported in `5fd236b`, rewritten for the current contract |

Checks after these commits: 555 CPU tests passed (69 skipped) on Python 3.10 with CPU
torch as in CI; 20 adapter/setup tests passed from a fresh clone; Ruff, shellcheck and
pre-commit on all files pass. No GPU inference was rerun.
