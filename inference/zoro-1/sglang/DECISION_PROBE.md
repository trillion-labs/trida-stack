# Self-spec decisions: H100 validation

Issue [#81](https://github.com/trillion-labs/trida-stack/issues/81).
The server is available experimentally through `serve_decide.py --backend
self-spec`. It supports direct, reasoned, and auto decisions plus self-spec chat
from one engine. Parity is not yet accepted. The earlier readout experiments and
the current implementation's validation are recorded below.

## What works

One `HybridDiffusionSelfSpec` generation engine can return normalized prompt
hidden states for the existing pointer head. The checkpoint is loaded once;
the head runs on CPU. Reference and candidate engines run in separate processes,
one after the other. Weights, head, and request inputs are unchanged.

`decision_readout.patch` applies **after** `two_stream_diffusion.patch`, on the
documented HybridDiffusion base `6ca547a`. It carries last-position states through
the DLLM logits path and into the request response. Only results explicitly
marked by the algorithm's pure causal-prefill branch are accepted.

The active outer result handler is in
`managers/scheduler_output_processor_mixin.py`. The similarly named method in
`dllm/mixin/scheduler.py` is shadowed by Python's inheritance order. The probe
connects the active handler and the fast decode loop to the same capture helper.

Prefixes are submitted **serially**. Each request generates and discards one
token, preserving the existing generation/cleanup path. This does not establish
prefill-only termination, concurrent capture, or self-spec chat/decision mixing.

## Results

H100, BF16, eager execution, one worker. Same local checkpoint used for the
earlier H100 checks:
`/mnt/vast/trillion/hyungguk/trida-jev-eval/runs/rl-head-i11/iter_300`.
This is not an independently verified download of the released HF model.
Reference: stock SGLang 0.5.23 / PyTorch 2.14.1. Candidate: the pinned diffusion
fork and public patch / PyTorch 2.9.1+cu128.

The synthetic short and long cases each contain noul, choice, and score
questions. Long question rows contain 7,566–7,584 tokens. Token-prompt SHA256 and
head-file SHA256 must match before comparing probabilities. The acceptance
tolerance remains **0.005**, including the original
`mlx/tests/compare_decisions.py` checks on returned answers.

| Case | Maximum probability difference | Result |
|---|---:|---|
| Short text | 0.004221 | Probabilities pass; derived confidence differs by 0.0060, so full answer parity fails. |
| Long text | 0.009739 | Probability and full answer parity fail. |
| Image | Not a valid comparison | The DLLM backend defaults to text-only; the probe reports unsupported. |

All choice winners agreed. Three repeats produced identical text probabilities.
The final candidate run made 66 serial prefix requests (two text cases, three
repeats, eleven prefixes each). Image cases were explicitly skipped as unsupported.
The tested runs completed capture and cache flush without an idle leak error;
this is a bounded smoke check, not a long-duration stability claim.

An earlier image run silently used the text-only backend and showed a 0.774630
probability difference. That value is **not** valid image parity evidence.
The probe now checks the runtime's multimodal setting before image inference.
Enabling image support and validating its processor/position path is separate
work.

The stock reference and diffusion fork use different runtime/kernel versions,
and the reference batches prefixes while this candidate serializes them. The
initial text differences were not attributed to any single cause. Do not widen the
tolerance or claim #81 is complete based on this experiment.

## Same-fork causal control

Follow-up H100 job **5143** runs the fork's causal mode and self-spec mode in
separate processes using the same shaped checkpoint, environment, FlashInfer
backend, eager execution, serial prefixes, and three repeats. Causal mode omits
`dllm_algorithm`, just like `serve.py --mode causal`. It uses the fork's
unmodified full-prefill hidden-state API and selects the final returned state;
self-spec uses the experimental last-state path. Both feed the same CPU head.

A stock causal **serial** control also ran to separate batching effects from
the old stock-batched reference. The input file's SHA256 was unchanged before
and after the job. Prompt and head hashes matched, and **0.005 remained the
strict tolerance** throughout.

| Comparison | Short text max probability difference | Long text max probability difference |
|---|---:|---:|
| Fork causal vs fork self-spec | 0.008859 | 0.006207 |
| Stock batched vs fork causal | 0.004891 | 0.009699 |
| Stock serial vs fork causal | 0.004055 | 0.010497 |
| Stock batched vs stock serial | 0.008607 | 0.007142 |

The two fork modes produced identical results across their respective three
repeats, but differ from each other before the pointer head: normalized hidden
states have minimum cosine similarities of 0.999899 (short) and 0.999912 (long),
RMS differences around 0.0104, and maximum coordinate differences of 0.125 and
0.1875. Thus pinning the runtime version alone does not restore parity.

Readout audit job **5145** checked all 22 prefixes in one repeat:

- The token ID at the actual selected prefill endpoint equals the request's
  final token ID; the actual full sequence length equals the prompt length.
- Only the algorithm's pure causal-prefill result is exported. The source
  backbone applies final normalization before returning to the logits processor
  (`models/qwen3_5.py`); the readout receives that normalized tensor.
- Converting the selected BF16 states to FP32 Python lists preserves their
  values exactly. Audited outputs and saved hidden states are bitwise identical
  to the earlier self-spec run.

These checks found no endpoint-selection or export error. The observed
differences are consistent with numerical differences in forward execution,
including causal vs DLLM attention/GDN/cache paths and batch shapes. Even stock
SGLang exceeds the tolerance when only the prefix submission layout changes.
This does **not** identify a particular kernel or rule out every forward-path
bug; no precision sweep, no-cache control, or layer-by-layer bisect was run.
Strict probability and full-answer parity still fail. Images remain explicitly
unsupported in both fork controls.

All inference, strict comparisons, input integrity checks, and readout audits
completed. Jobs 5143 and 5145 exited 0 because their harnesses retain expected
comparison rejections; the comparison JSON still reports `passed: false`.
There were no scheduler exceptions or idle memory-leak errors in these runs.
Artifacts are under `same-fork/` and `same-fork/audit/` in the H100 task directory.

## Experimental server and independent validation

`self_spec_decide_engine.py` reuses the existing request encoder, CPU pointer
head, formatting, chat template, and HTTP routes. The optional versioned
`decision_readout.patch` replaces the prototype overlay. The server checks its
version before loading weights. No backend source or weights are vendored.

- Direct decisions use serial K+1 prefix requests, discarding one generated
  token per prefix to preserve the tested KV cleanup path.
- `reasoning: true` and `"auto"` generate with the self-spec engine. Auto first
  scores directly and escalates when the lowest top probability is below the
  requested threshold. Both CUDA engines share the reasoning orchestration;
  CUDA and MLX share conclusion fitting and cut/source normalization. Completed
  reasoning is fitted to the state budget. Incomplete/empty/unfittable reasoning
  keeps the direct answer with the canonical `cut` and `answers_source` fields.
- All native calls share an admission lock, including streaming and abort
  acknowledgement. Native aborts are retried across tokenization/admission
  races. Failure to acknowledge within 30 seconds prevents further inference
  until the server is restarted.
- Images are rejected. The pinned decoder's sampling is global, so chat
  settings conflicting with the YAML are rejected rather than silently ignored.
  The shipped YAML uses temperature 1, top-p 0.95, top-k 20 and no presence penalty.

### 40 independent cases: provisional profile failed

H100 job **5147** used 40 new synthetic requests generated by
`tools/parity_cases.py`, with 120 questions and 345 option probabilities. These
cover eight domains, English/Korean, several option counts, singleton choices,
and four long contexts around 6.5K tokens. They are not accuracy benchmarks or
LLM-judged examples. The cases were frozen before either backend ran; input and
token hashes matched, and no inputs were edited after observing results.

The opt-in `--profile self-spec-bf16` uses **absolute probability tolerance
0.02**. The default `strict` profile and the shared MLX comparator still use
0.005. The opt-in profile checks each answer's derived confidence/score against
its own probabilities and reports their differences; it does not apply 0.02
indiscriminately to ordinal scores or normalized confidence. Choice changes
away from a reference margin of at most `2 * atol` fail the profile.

**38/40 cases passed. The profile as a whole failed.**

| Metric | Observed |
|---|---:|
| Mean absolute option probability difference | 0.004969 |
| Maximum option probability difference | 0.020536 |
| `ambiguous-23` maximum | 0.020536 |
| `support-33` maximum | 0.020484 |
| Maximum confidence difference | 0.0295 |
| Maximum ordinal score difference | 0.0355 |

One choice winner changed (`routing-21`, reference top-two margin 0.002114).
Two score distribution modes also changed near ties. Top-probability thresholds
crossed 0.5 in `inventory-02` and 0.7 in `support-33`; this can change auto
escalation. Derived answer fields remained consistent with their probabilities.
No tolerance was raised again to make these observations pass. The 40-case
measurement compares the readout backend through the probe; HTTP behavior is
covered separately below.

Reproduce by generating a case file once, running reference/candidate with that
same file, then selecting the profile explicitly:

```bash
python tools/parity_cases.py cases.json
# Run the reference and candidate commands below with --requests cases.json.
STOCK_PYTHON tools/probe_decision_prefill.py compare --profile self-spec-bf16 \
  --reference reference.json --candidate candidate.json --output comparison.json
```

### Actual HTTP smoke: passed

H100 job **5149** completed with one BF16 engine, one eager worker, and 100
native calls. Through real loopback HTTP it checked direct short/long decisions,
image/invalid-option rejection, auto skip at threshold 0, escalation at threshold
1, one-token reasoning fallback, a completed reasoned decision, chat, mixed
clients under serial admission, and streaming.

The completed reasoning case used `{a: 2, b: 1}` and a noul comparison question.
It generated 454 characters within a 512-token budget, closed thinking, and
returned a scored `reasoned` answer. This is not the earlier WorkflowEvals
invoice case or evidence of broad model reasoning quality.

SSE was closed after native text generation began, and a reasoned HTTP request
was disconnected after native admission. The subsequent direct decisions took
1.019 and 1.036 seconds respectively. Final native request count was zero,
cache flush succeeded, and no idle memory-leak/scheduler errors were logged.
The first HTTP attempt (5148) failed on a tokenizer API return-type difference;
reasoning now renders the template as text and tokenizes it like the chat path.

Artifacts: `validation40/` and `http/` under the H100 task directory. The HTTP
harness, immutable case file, full results, and job logs are retained there.
These are bounded checks; long-duration stability, images, concurrent native
readout, distributed execution, and zero-token completion remain unvalidated.
The implementation remains **experimental and does not close #81's parity gate**.
Local verification after the PR review: 205 CPU tests passed, two skipped;
53 MLX decision/server tests passed, one skipped. Changed Python files
passed Ruff, the supply-chain check passed, and the patch applied cleanly to the
pinned backend after the two-stream patch.

### Review fixes: reasoning token boundaries

If the combined reasoning prompt has no generation space, or an explicitly
requested budget does not fit, the server now returns the computed direct
answer with `cut` instead of HTTP 400. Completed reasoning is checked against
the actual encoder's escaped state-token count, including the state delimiter,
before it can be reported as a `reasoned` answer. The shared request contract
and its MLX copy remain byte-identical; encoding and token limits are unchanged.
Twelve CPU regression cases cover these boundaries, including the exact-fit
state boundary that must remain accepted.

H100 HTTP job **5158** passed the existing smoke checks and the new context
regression: three 6,012-token decision rows fit individually, while the combined
18,075-token reasoning prompt exceeded the unchanged 16K server context. The
response was HTTP 200 with the direct answer and `cut`, without starting native
reasoning. The real tokenizer also confirmed a control-token case whose raw
state count is 2,009 but whose actual escaped state count is 10,010.
Completed reasoning/re-scoring, mixed requests, and cancellation recovery still
passed. After 109 native calls, no requests remained and cache flush succeeded;
SSE and decision cancellation recovery each took about 1.03 seconds. No idle
memory-leak or scheduler errors were logged.

The original smoke inputs and frozen 40-case input files had identical SHA256
before and after this run. Strict tolerance remains 0.005 and the opt-in
probability tolerance remains 0.02. The 40-case parity measurement was not
rerun for these reasoning-only fixes; its two failures remain unresolved.
New run artifacts are in `review-fixes/` under the same H100 task directory.

## PR review: shared contract and readiness

The request helpers adopt #89's `fit_reasoning` and limit validation directly;
benchmark code from that PR is not imported. `SelfSpecDecideEngine` no longer
implements its own `adecide`: it inherits the causal engine's shared policy and
only replaces native generation/readout operations. MLX uses the same torch-free
`finish_reasoning` helper. `cut` is `length`, `empty`, or `state_limit`,
`answers_source` is `direct` or `reasoned`, and `reasoning_trimmed` records
conclusion fitting. Optional `cut_detail` identifies context/row limits. Native
row-limit errors after reasoning still fall back to the direct answer.

All three implementations default to a 4096-token reasoning cap (768 for boolean
reasoning, the configured cap for auto). CUDA and MLX expose
`--reasoning-token-limit` up to 8192. `usage.input_tokens` sums the decision rows
actually scored, including both passes, and `usage.output_tokens` uses native
reasoning counts even on fallback. These are logical API counters, not physical
K+1-prefill totals. This supplies #94's usage fields without adding its separate
`/v1/systemone` endpoint here. #89 and #94 remain independently open PRs; their
eventual handler merges still require the usual rebase/integration checks.

Abort-acknowledgement failure still prevents further inference, and `/health`
now returns 503 with `ok: false`. CPU regression tests force that timeout and
check readiness, rather than only testing the initial healthy state. The main
launcher's `--decision-backend` help explicitly reports experimental status and
the incomplete 38/40 parity result.

### Latency and AR comparison

H100 job **5162** used the same checkpoint as above, BF16, one eager worker,
memory fraction 0.5, context 16384, and temperature 1 / top-p 0.95 / top-k 20.
A separate synthetic latency request was fixed before measurement: a 5500-word
`reference` archive, an invoice state, and the same three typed questions as the
earlier smoke request. Encoded rows are 5537, 5550, and 5555 tokens. This is a
new performance fixture; the original smoke and 40-case inputs were untouched.

**Chat and decisions do not overlap.** A streaming chat holds admission for its
entire generation/abort acknowledgement, and each question requires K+1 serial
prefills. The self-spec backend's first cold-cache direct call took **1.731 s**;
the next two warm calls took **1.559 and 1.064 s**. A completed 512-budget
reasoned call took **8.673 s**, generating 344 tokens (5.677 s reasoning wall
time). It includes direct and reasoned scoring passes.

Stock AR control job **5165** used the same checkpoint, request and sampling,
but the actual stock last-state API with batched prefixes (maximum 16 running
requests). Stock uses SGLang 0.5.23 / PyTorch 2.14.1 and its default attention
backend; self-spec uses the pinned diffusion fork / PyTorch 2.9.1 and FlashInfer.
Both have CUDA graphs and overlap scheduling disabled. Stock's first cold-cache
direct call took 10.022 s, including the first long-shape compilation; warm calls
took **0.133 and 0.117 s**. Cold numbers are not steady-state latency claims.

| Backend | Budget | Actual reasoning tokens | Reasoning wall s | Wall tok/s | Total HTTP s | Answer source / cut |
|---|---:|---:|---:|---:|---:|---|
| Stock AR | 128 | 128 | 5.287 | 24.2 | 5.654 | direct / length |
| Self-spec | 128 | 128 | 2.359 | 54.3 | 3.470 | direct / length |
| Stock AR | 512 | 38 | 1.195 | 31.8 | 1.991 | reasoned |
| Self-spec | 512 | 344 | 5.677 | 60.6 | 8.673 | reasoned |
| Stock AR | 1024 | 157 | 4.779 | 32.9 | 5.336 | reasoned |
| Self-spec | 1024 | 196 | 2.897 | 67.7 | 5.783 | reasoned |

Reasoning wall time includes rendering, tokenization, prefill, and generation;
this is not decode-only throughput. These are single observations per budget,
and sampling produced different lengths. **There is no general end-to-end
speedup claim:** stock warm direct decisions were much faster, and total time
depends on the generated length and serial readout costs. No interpretation of
these timings changes the failed parity result or model accuracy claims.

An earlier same-fork AR control in job 5162 exported full prefill states and
selected the last state on CPU. Its roughly 13 s warm direct latency is dominated
by full-state export and is not used as the serving comparison. The stock
FlashInfer attempt (5164) failed on incompatible CUDA compiler/header versions;
5165 used the previously working stock default path without changing the environment.

HTTP regression job **5166** passed the same smoke inputs plus canonical
cut/source/usage and budget-cap checks, completed reasoning, mixed clients,
and cancellation recovery (SSE 1.019 s, decision 0.995 s). It ended with zero
native requests and a successful cache flush. Readiness returned 503 when
poisoned in the latency checks; no scheduler/idle-leak errors appeared in the
successful runs. Artifacts, harnesses and the fixed latency request are in
`pr96-review/` under the H100 task directory. Input SHA256 checks passed for all
three fixtures before/after measurement. No benchmark inputs or tolerances were
changed, and the 40-case parity suite was not rerun for this contract change.

## Reproduce

Use separate stock-SGLang and diffusion environments as documented in the
parent inference README and this directory's README. In an isolated upstream
checkout, apply the public patch, then this experimental patch:

```bash
git apply /path/to/trida-stack/inference/zoro-1/sglang/two_stream_diffusion.patch
git apply /path/to/trida-stack/inference/zoro-1/sglang/decision_readout.patch
```

Prepare `cases.json` as a list of `{"name": "...", "request": {...}}` objects.
Use the same file and checkpoint for both commands. Run each engine in its own
process and environment, followed by the comparison:

```bash
PROBE=/path/to/trida-stack/inference/zoro-1/sglang/tools/probe_decision_prefill.py
STOCK_PYTHON "$PROBE" reference --checkpoint "$CKPT" --requests cases.json --output reference.json
DIFFUSION_PYTHON "$PROBE" candidate --backend /path/to/HybridDiffusion/eval \
  --checkpoint "$CKPT" --requests cases.json --repeats 3 --output candidate.json
STOCK_PYTHON "$PROBE" compare --reference reference.json --candidate candidate.json --output comparison.json
```

Replace the interpreter placeholders with executable paths. Comparison exits
nonzero for unsupported cases or parity failures. The backend path is checked
explicitly because the shared H100 environment prepends a different checkout
through a `.pth` file. Existing installations are not modified.

For the same-fork control, run `causal` and `candidate` with the diffusion
interpreter and identical arguments. Add `--save-states` to both, then compare
the generated JSON files; the comparator also reports raw hidden-state metrics
when the matching `.states.pt` files exist. A stock serial control uses
`reference --reference-layout serial --save-states`. These switches change
execution/observation only, not the request file or acceptance tolerance.

H100 artifacts: `/mnt/vast/trillion/wonsuk/issue81-selfspec-20261113/`.
Final candidate/comparison job: **5140**, after reference generation in **5131**.
The job exits 1 because strict parity fails, not because inference crashes.
The stored inputs, reference/candidate JSON, comparison JSON, and Slurm logs
remain outside the repository. Earlier unsuccessful diagnostics are retained
there; they are not successful parity tests.

Experimental reasoning/auto serving is implemented; strict parity remains unmet.
Image support,
zero-token cleanup, and mixed requests remain separate validation gates.
The patched backend retains the existing
PolyForm Noncommercial restriction; see `NOTICE`.
