# Decision strategy and parity follow-up

One H100 80GB, 2026-11-17. **Two-runner pooling was fastest in these probes.**
A one-model worker extension can also reuse prefixes with chunked prefill,
without modifying vLLM core. Neither result establishes strict SGLang parity.
The PR still uses sequential, cache-off capture; the experimental alternatives
below have not been applied to the server.

## Fixed conditions

The comparison reused the earlier short-text, image and long-text request
fixtures, each containing three typed questions. Encoded row lengths were
45/58/63, 86/96/90 and 7566/7579/7584. Input, expanded-row and pointer-head hashes
were checked across implementations. The local checkpoint was the one used in
the earlier H100 reviews, not an independent verification of the released Hub
weights. Its head SHA-256 is
`1e406f44fd66191cc16bbddf02213182a0e8403ee98ae971fedf8f72a294d9db`.

The vLLM environment was 0.32.0 / torch 2.13.0 / transformers 5.19.0, BF16,
eager execution, one worker, 8192-token context and four request slots. Each
strategy ran in a fresh process on the same GPU, with a total memory budget of
70%; two-runner pooling split that budget equally. Sampling, fixtures and the
original absolute probability gate **< 0.01**, including selected-option
agreement, were unchanged.

## A/B results

Seconds, median of five samples after the first request and one warm-up.
Four-client mix means the fixed short/image/long/short requests completing
together. Cache-on results came from a separate follow-up, not an interleaved
statistical performance trial.

| Strategy | Short | Image | Long | Four-client mix | Peak GPU GiB |
|---|---:|---:|---:|---:|---:|
| One model, sequential prefixes, cache off | 0.691 | 0.750 | 1.689 | 2.569 | 56.18 |
| One model, two parallel prefixes, cache off | 0.668 | 0.653 | 1.674 | 2.647 | 56.18 |
| One model, four parallel prefixes, cache off | 0.357 | 0.433 | 1.681 | 3.303 | 55.91 |
| Two models, full-row pooling, cache off | **0.187** | **0.234** | 0.777 | **1.404** | 55.33 |
| One model, sequential prefixes, cache on, batch 2048 | 0.645 | 0.726 | 0.780 | 1.516 | 54.61 |
| One model, four parallel prefixes, cache on, batch 2048 | 0.384 | 0.387 | 0.753 | 1.782 | 54.62 |
| One model, four parallel prefixes, cache on, batch 8192 | 0.371 | 0.396 | **0.737** | 1.795 | 55.38 |

Loading once was the main reason for moving away from pooling. The A/B does
not justify that choice on performance alone: pooling was about 3.7x faster
on short input and 2.2x on long input than sequential cache-off capture.
Native cache reuse largely removed the long-input penalty, but small requests
still paid for multiple prefix readouts/RPCs. Four-prefix submission improved
short requests and did not improve this mixed workload.

Each backbone load used 8.56 GiB of weights, versus 17.12 GiB for two runners.
The similar total memory readings above include reserved caches under the same
70% budget; they do not mean two models cost the same memory as one. Full-row
pooling also returned 58,186,240 hidden-state values to the CPU for each long
request. SGLang latency was not remeasured in this comparison.

## Cache, concurrency and cancellation

The experiment enabled native prefix caching and chunked prefill through engine
arguments. An opt-in worker extension skipped intermediate chunks and captured
only the final prompt position, retaining ticket, length, row, finiteness and
worker/model checks. It wrapped the same `add_requests` and `sample` boundaries;
no installed vLLM source files were edited.

A cold 7562-token prefix at batch budget 2048 executed as six chunks, with
computed/scheduled pairs:
`[0,1584], [1584,1584], [3168,1584], [4752,1584], [6336,1056], [7392,170]`.
There was exactly one final readout. Subsequent prefixes reused **7392 tokens**;
487,872 cache-hit tokens were observed over the long-case warm-up and repeats.
The 8192-budget run also chunked at the native cache-alignment boundary.

The HTTP probes exercised live generation plus capture, concurrent distinct
requests, stream cancellation and admitted decision cancellation. The two
2048-budget runs additionally observed cancellation during partial prefill,
then a successful next request. Final unfinished requests, tickets, bindings
and capture failures were zero, and GPU memory returned to zero after all runs.
These are resource/routing results, not blanket numerical passes: four-prefix
runs exceeded 0.01 on some post-cancellation responses, and concurrent capture
could differ from its solo answer by more than 0.01. Pooling's mixed answers
matched the earlier solo vLLM reference at API precision in this fixture set.

## Where the probability difference appears

A separate audit used the same fixtures and server decision policy, called
directly through `adecide`. Each vLLM profile ran 50 requests across solo,
within-question parallel and four-request mixed phases. Four profiles produced
**2200 audited vectors**, all passing:

- Bitwise equality with the actual tensor passed to native `model.compute_logits`
  in the **same forward**, and with an independently indexed query-last vector.
- Native expanded-prompt hash, final token ID and position matching the intended
  prefix; no next generated token or intermediate chunk was read.
- Request/ticket and worker/model identity checks. Audit failures: **zero**.

Rescoring the saved vectors through the same head and temperature reproduced
the recorded API probabilities exactly at four-decimal precision. Computing
that head in FP64 rather than FP32 changed probabilities by at most
**1.48e-6**. The larger differences were already present in the backbone vectors.

A cache-off precision control kept the checkpoint, inputs and vLLM/torch
versions fixed. FP16 changed runtime precision only, not the saved weights or
production configuration:

| Precision | Parallel-prefix variation vs solo | Four-request variation vs solo | Maximum hidden coordinate change |
|---|---:|---:|---:|
| BF16 | 0.0103 | 0.0103 | 0.2578125 |
| FP16 | 0.0025 | 0.0026 | 0.03125 |

Both precisions were repeatable in the solo phase. For example, the short
request's middle `amount` probability changed from 0.6305 to 0.6408 under BF16
batching, versus 0.6310 to 0.6284 under FP16. The lower variability under FP16
supports a precision/runtime explanation rather than a capture-index error.
It does not prove that BF16 rounding is the only cause: dtype can change kernel
selection, and no full-backbone FP32 oracle or layer-by-layer audit was run.

SGLang 0.5.23 / torch 2.14.1 / transformers 5.19.0 also changed on identical
cold/warm inputs: maximum probability differences were 0, 0.0040 and **0.0120**
for short/image/long cases. On the long case, `route.review` changed from
0.8568 to 0.8688. Its raw hidden states changed too. This path does not use our
vLLM extension. The torch-version and kernel/cache differences remain confounders
in cross-engine attribution.

## Limits that remain

- **Strict parity still fails.** Across the audit runs, the maximum gap against
  the **unchanged earlier SGLang reference** was 0.0199 for each BF16 profile and
  0.0157 for FP16. Against both newly collected SGLang cold/warm results, the
  maxima were 0.0239 and 0.0157. All selected options/directions agreed; that was
  not used to waive the 0.01 probability gate. The earlier reviewer's separate
  0.0222 five-option case was not reproduced by this three-fixture experiment
  and remains a documented limitation.
- The audit found no readout error in these observations, but does not prove
  every input correct or exclude an upstream kernel/cache defect. Neither
  changing dtype nor widening tolerance is proposed as a parity fix.
- These are small synthetic probes, not full WorkflowEvals scores or a new
  completed scored-reasoning run. Image-encoder reuse, long-duration stability,
  distributed workers and CUDA graphs remain unverified here. The current
  server's chat-starvation limitation remains documented in the README.
- Existing `Event loop is closed` output-handler errors remained at process
  teardown. Exit codes were zero and requests/GPU memory drained; clean graceful
  process shutdown has not been established.
- Experiments remain separate from this PR's serving implementation. Whether to
  retain this experimental backend, adopt pooling or pursue cached capture is
  still a review/design decision. None meets #60's strict probability-equivalence
  criterion yet.

## Validation record

The A/B harness CPU checks passed 5 tests; cache/chunk plus existing vLLM
regressions passed 59; the independent readout audit plus existing regressions
passed 56. Before/after hashes of eight installed vLLM core files matched in
the cache and origin-audit runs. Inputs, comparison functions and tolerances
were retained throughout. The latest serving-code CI passed 678 CPU tests
(69 skipped), and local MLX regressions passed 56 (2 skipped).
