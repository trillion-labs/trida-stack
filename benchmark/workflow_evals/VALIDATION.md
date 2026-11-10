# Review validation

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
[intake record](../../docs/oss/intake/0009-workflowevals.md).
