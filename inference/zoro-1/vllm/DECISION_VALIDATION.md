# Decision serving validation

## Environment

2026-11-13, one H100 80GB on `diffusion-compute-000`, allocated through Slurm.
Stock vLLM **0.32.0**, torch **2.13.0**, transformers **5.19.0**, BF16, no
quantization or diffusion plugin. The reference
uses stock SGLang **0.5.23**. The branch starts at `ef46bd5` on `origin/main`.

Both backends read the same local checkpoint used for the earlier #89 H100
checks, `/mnt/vast/trillion/hyungguk/trida-jev-eval/runs/rl-head-i11/iter_300`.
Its `jev_head.pt` SHA-256 is
`1e406f44fd66191cc16bbddf02213182a0e8403ee98ae971fedf8f72a294d9db`.
This was not a fresh download or an independent comparison of all weights with
the released Hugging Face checkpoint.

## Hidden-state feasibility

Slurm job 5107 compared the same encoded rows for one `noul`, one `choice`, and
one `score` question. Rows were 45, 58, and 63 tokens. vLLM returned one
unnormalised hidden state per token; option-end and decision positions were
compared with SGLang's prefix/last-state implementation through the same head.

- All three selected options agreed.
- Maximum absolute probability difference: **0.003980**.
- Minimum hidden-state cosine similarity: **0.999856**.
- Repeating the same vLLM batch produced identical selected hidden states.
- The predeclared probability tolerance was **0.01**, absolute.

The initial asynchronous multi-row implementation changed probabilities by up
to **0.0103** when concurrent HTTP requests changed batch composition. The initial
decision runner therefore used fixed one-row batches. The serial fallback
still does; bounded concurrency is now available with the numerical limits
reported below. Both paths use eager execution, no prefix caching, and no
chunked prefill.

## Single generation engine

Slurm job **5121** loaded one stock generation runner with
`trida_worker_extension.DecisionWorkerExtension`. The worker captured **33**
last-prompt hidden states across all three question types on short text, an
image, and long rows. Every capture reported the same worker PID and model
instance identity. The log recorded one **8.56 GiB** model load.

- Every reported answer probability matched the old pooling implementation
  exactly at the API's four-decimal precision.
- Ordinary greedy generation returned identical token IDs before and after
  captures. Arming then abandoning a capture did not change generation.
- No second model, pooling runner, or vLLM source patch was used.

The server now uses this one-runner design. Captures read the hidden vector
before the unchanged vocabulary projection; each prefix generates one discarded
token to run the normal generation path. Ticket and call-count checks reject
missing, stale, or ambiguous vectors. Ordinary generation and capture share a bounded execution gate, including
native iterator cleanup on cancellation. The original exclusive path remains
available with `--max-concurrency 1`.

## Cross-backend probability parity: still open

Job **5117** compared the final HTTP server's answers with SGLang on the same
checkpoint and requests. All categorical choices and yes/no directions agreed.
The **0.01 absolute option-probability tolerance did not pass every case**:

| Input | Encoded row lengths | Maximum absolute probability difference |
|---|---|---|
| Short text | 45, 58, 63 | 0.0092 |
| Red image | 86, 96, 90 | 0.0022 |
| Long state at the encoder limit | 7,566, 7,579, 7,584 | **0.0120** |

Job **5119** then used vLLM's unnormalised LAST pooling on K+1 prefixes per
question, matching SGLang's readout strategy. Every reported probability was
identical to vLLM's full-row result at the API's four-decimal precision. The
prefix strategy now lets the server share one generation engine; it did not
remove this cross-engine discrepancy.
Both engines honor this checkpoint's FP32 GDN-state setting. The remaining
cross-engine numerical difference has not been isolated to a specific kernel.

The functional API and hidden-state extraction are implemented, but #60's
probability-equivalence acceptance criterion remains open. These small probes
do not establish parity across arbitrary checkpoints or requests.

## CPU checks

`pytest -q`: **206 passed, 2 skipped**. Ruff, the supply-chain pin check, and
`git diff --check` passed. New tests cover hidden-state alignment and finiteness,
option ordering, image placeholder construction, complete/truncated/empty/oversize
reasoning, context and budget validation, sampling metadata, capture tickets,
single-model startup, bounded concurrency, reordered native-row routing,
ignoring late cancelled tickets, serial fallback, and cancellation cleanup
(including repeated cancellation).
The HTTP budget test also catches a main-branch handler that previously ignored
`max_reasoning_tokens`. The CPU tests do not import vLLM or load a model.

## Initial HTTP end-to-end checks (two runners)

Slurm job **5115** ran the actual vLLM HTTP server with both runners, an
8,192-token context, and a 0.35 GPU memory fraction per runner. It passed:

- All three question types on short text, a long state reaching the encoder's
  state limit, and a synthetic red image with image-specific questions.
- Two concurrent copies of a request: maximum probability difference **0.0**.
- A one-token reasoning budget: exactly one completion token, `cut: "length"`,
  `answers_source: "direct"`, and the original direct answers.
- A complete reasoning request: the equation `2 + 2 = 4` finished in **19 tokens**,
  `finish_reason: "stop"`, no trimming, and `answers_source: "reasoned"`.
- Missing request and out-of-range budget rejection with HTTP 400.
- Nonstreaming chat and SSE chat ending with `[DONE]`.

These are synthetic transport and integration checks, not WorkflowEvals scores.
Raw artifacts are under
`/mnt/vast/trillion/wonsuk/issue60-vllm-20261113` on the GPU cluster.

## Earlier HTTP end-to-end checks (one serial runner)

Slurm job **5122** repeated the HTTP checks with the worker extension and **one
generation runner**, at an 8,192-token context and memory fraction **0.5**.
The log contains one 8.56 GiB model load. Short text, image, and long-input
answers were identical to the two-runner API at its four-decimal precision.

The run also passed concurrent decision/chat requests and a real SSE client
disconnect followed by another decision. The subsequent decision completed in
**0.76 seconds**, with the original direct answers. Concurrent duplicate
decisions again had a maximum probability difference of **0.0**.

The one-token reasoning request returned `cut: "length"` and direct fallback.
The 256-token equation request exhausted its budget on this run and was also
correctly returned as direct fallback; it was not a completed reasoned answer.
The earlier two-runner run's 19-token completion is not claimed for this run.
No prompts, sampling parameters, or budgets were tuned to force a completion.

HTTP rejection of invalid requests and both chat modes still passed. This
single-engine change preserves the SGLang parity limitation reported above.

## Earlier bounded concurrency experiment (before HTTP integration)

Slurm job **5124** tested a separate experimental worker extension on the same
H100/checkpoint/vLLM 0.32.0 environment. It wrapped two V2 worker-runner methods:
`add_requests` binds a pre-armed ticket from `SamplingParams.extra_args` to the
native request ID; `sample` uses `InputBatch.req_ids`, `logits_indices`, and
`cu_num_logits_np` to select only that request's hidden vector. Normal generation
is not captured. The original methods still run unchanged after the observation.
No vLLM source files were patched. Before/after checksums of the model runner,
input batch, sampling parameters, and AsyncLLM source files matched.

The experiment used one generation worker, eager execution, no prefix cache,
no chunked prefill, no speculation, and `max_num_seqs=4`. Its results were:

| Submitted workload | Largest actual scheduler batch | Result |
|---|---:|---|
| 1 generation + 1 capture | 2 | Passed |
| 1 generation + 1 long-input capture | 2 | Passed |
| 3 generations + 1 image capture | 4 | Passed |
| 2 generations + 2 captures | 4 | Passed |
| 4 captures | 3 | Routing passed; a four-capture batch was not observed |

There were **18 successful captures**, five observed generation/capture mixed
batches, and two batches containing multiple captures. Every result had one
capture call, the expected prompt length, and its own native request binding.
The worker and model identity stayed constant. The final capture store and ID
map both contained **zero** entries.

Cancelling a running generation was followed by successful captures in **0.16
seconds**. Two cancelled capture trials cleaned up their stored vectors; both
had already observed one model call before cleanup, so they do not prove
cancellation at every point inside a GPU forward. A deliberately late request
for a cleared ticket was ignored and did not recreate its capture entry.
A separate CPU check covered reordered batch rows and ignoring ordinary decode.

Concurrent hidden states were not bit-identical to the sequential reference:
minimum cosine similarity was **0.999684**, and the largest absolute coordinate
difference was **0.12890625**. These are hidden-vector measurements, not answer
probability errors. This experiment establishes request-routing feasibility
within the tested workloads; it does not establish probability parity, HTTP
integration, long-duration stability, or support for graphs/distributed workers.

The production serial extension, engine, and launcher were unchanged (their
checksums also matched). Experimental sources, CPU check, scheduler trace,
checksums, and `report.json` are retained under
`/mnt/vast/trillion/wonsuk/issue60-vllm-20261113/concurrency`.

## HTTP concurrency integration

The application now shares one bounded gate between generation and capture,
with four slots by default and `--max-concurrency 1` retaining the serial path.
Concurrent captures use the request-indexed worker extension described above.
HTTP decision disconnects cancel native work and await ticket cleanup; streaming
chat closes its native iterator before releasing its slot. The SGLang handler
keeps its previous behavior; disconnect monitoring is enabled by the vLLM launcher.

Slurm job **5126** passed the actual FastAPI/uvicorn HTTP workload at concurrency
four, using one 8.56 GiB model load. Actual scheduler batches were two for one
chat plus one capture, and four for three chats plus one capture and for two
chats plus two captures. Maximum probability differences from sequential HTTP
answers were **0.0**, **0.0083**, and **0.0031**, respectively. The two simultaneous
decisions in this first run used identical inputs.

Job **5127** strengthened the last case with distinct text and image decisions.
Both HTTP responses succeeded and matched their sequential references at API
precision, but their capture prefills did not overlap: the largest mixed batch
was **three**, not the test's required four. That batch-size assertion failed,
so this job did not reach the cancellation checks. This is not evidence for
four simultaneous image/text capture rows.

Job **5128** observed a four-row batch with two distinct text decisions and
two chats (maximum probability difference **0.0076**). Its strengthened
cancellation assertion failed because the HTTP request was cancelled while its
ticket was armed but before native admission: one pending ticket was cleared,
with zero admitted cancellations. The final store and binding map were empty
and capture failures were zero. The subsequent test waits for an actual native
binding with only the target decision active, avoiding this admission race.

Slurm job **5129** passed the final HTTP checks, including two distinct text
decisions with separate question IDs. Actual mixed batches were **2, 4, and 4**
for the three workloads above; their maximum probability differences from each
request's sequential reference were **0.0, 0.0083, and 0.0076**.

- SSE client cancellation was followed by a successful decision in **0.73 s**.
- An HTTP decision was disconnected only after its native binding was observed,
  while two chats remained active. One admitted capture with an observed model
  call was cleared; a subsequent decision completed in **2.85 s**. This does not
  prove interruption at every point inside a GPU forward.
- The final capture store and native-ID map were empty, with **zero** capture
  failures. Worker/model identity stayed constant and the log recorded one
  **8.56 GiB** weight load.
- Short, image, and long sequential answers matched the earlier serial HTTP
  references at API precision. One-token reasoning correctly returned direct
  fallback; invalid bodies and budgets returned HTTP 400.
- Before/after hashes of the four monitored vLLM source files matched. No core
  patch was needed. These are bounded integration checks, not a throughput,
  long-duration reliability, or cross-backend probability-parity claim.

The HTTP harness, reports, and logs are retained under
`/mnt/vast/trillion/wonsuk/issue60-vllm-20261113/http-concurrent`.

## Reproduce the hidden-state probe

Use separate vLLM and SGLang environments, pointing both at the same local
checkpoint. Each process exits before the next backend loads the model:

```bash
PROBE=inference/zoro-1/vllm/tools/probe_decision_hidden_states.py
VLLM_PLUGINS='' "$VLLM_PYTHON" "$PROBE" worker "$CKPT" --output /tmp/decision-probe
"$SGLANG_PYTHON" "$PROBE" sglang "$CKPT" --output /tmp/decision-probe
"$SGLANG_PYTHON" "$PROBE" compare-worker "$CKPT" --output /tmp/decision-probe
```

The probe writes the exact token rows, selected hidden states, probabilities,
and `worker-report.json`. The original pooling experiment remains available
through the `vllm` and `compare` modes. This is a synthetic integration check, not benchmark accuracy
or proof of equivalence for every request.

## Operational limits

One backbone now serves decisions, reasoning, and chat. The default memory
fraction is **0.5 for the one engine**. The prototype's `--no-generation` option
was removed along with the pooling runner. Execution currently requires one
worker, eager mode, no prefix cache, no chunked prefill, and no speculation.
Generation and capture share a bounded execution limit of four by default
(`--max-concurrency 1` retains the serial path; 2 is also supported). Streaming
chat occupies one slot until completion or cancellation. Concurrent capture
requires the unsharded V2 runner. K+1 prefix passes add compute, and BF16
probabilities may vary with batch composition. Graphs, distributed workers,
higher concurrency, and long-duration stability remain unvalidated.

On this node the optional FlashInfer sampler failed to compile because its CUDA
headers and compiler were incompatible. The server defaults to vLLM's PyTorch
sampler, without changing temperature, top-k, or top-p. No model tuning or
benchmark-input changes were made.
