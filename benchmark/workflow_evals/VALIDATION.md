# Review validation

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
context remains unverified. Maintainer license approval and Luke's unpublished
commits also remain pending.

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
for unverified datasets awaits maintainer approval; no approval is claimed.

CUDA/Zoro-1 GPU inference remains pending; no GPU node is available. Luke's seven
announced #83 commits were not available on #83's frozen head (`e0a8c60`) or the
replacement branch when checked. They still need to be republished for comparison.

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
the pending status of the other three datasets are in the
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
License-exception approval and the disposition of Luke's seven #83 commits remain
maintainer decisions; this change does not grant approval or supersede that work.
