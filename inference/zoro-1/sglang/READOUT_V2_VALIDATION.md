# Readout v2 H100 validation

2026-11-17, tested serving code `84e645d`. One H100 80GB, BF16, one eager
worker, context 16384, memory fraction 0.5 and the existing self-spec YAML.
The isolated backend was HybridDiffusion `6ca547a` with
`two_stream_diffusion.patch` followed by `decision_readout.patch` version **2**.
The runtime was the earlier Python 3.10 / torch 2.9.1+cu128 environment.
The source path and version guard were checked before loading the model.

**Request-level failure isolation, cancellation recovery, the readout deadline
and fail-closed health behavior passed. Probability parity is still not accepted.**
No serving-code changes were needed for these checks.

## Method and unchanged inputs

The short/long decision fixtures, earlier cancellation prompts and budgets,
checkpoint and frozen 40-case suite were retained. Input files were hashed
before and after the run. The frozen-suite SHA-256 was
`6a351f5fefbf9e7b64710474a001fa6c269d894fca952fc01e2bac4710768ae1`.

An isolated Python import hook injected one-shot faults **after a real GPU
forward**, changing only readout output metadata. It was not added to the
serving code or installed in the shared backend. Faults held a short barrier
so another valid HTTP decision was waiting on admission before the bad request
was released. Actual input IDs, model weights and probability gates were not
changed. `TRIDA_DECISION_READOUT_AUDIT=1` remained on.

The four metadata faults exercised the shared helper from the observed
`process_batch_result_dllm` path. The inline caller of the same helper was not
separately forced on GPU; its helper logic remains covered by CPU regression.
This is controlled fault injection, not a claim to reproduce a physical GPU hang.

## Failure isolation

| Injected metadata failure | Bad HTTP request | Readout export | Queued/next healthy request |
|---|---:|---:|---|
| Wrong final token ID | 500 | 0 states | Succeeded |
| Wrong prompt length | 500 | 0 states | Succeeded |
| Hidden-state batch mismatch | 500 | 0 states | Succeeded |
| Upstream readout-validation error | 500 | 0 states | Succeeded |

Each bad request was marked for abort. The same scheduler process continued;
there was no process-wide failure or endless retry. Queued and following
healthy answers had **zero probability difference** from the baseline under
the unchanged 0.005 recovery check. Native pending requests returned to zero.
The following decision took 0.987–1.006 seconds.

The production readout deadline stayed **30 seconds**, with a **30-second**
abort-drain window. A test-only 32-second scheduler delay caused an HTTP 500
at 32.198 seconds after the timed-out readout was drained. `/health` stayed
healthy, the next decision completed in 1.031 seconds, its probability delta
was zero and native pending requests were zero.

## Cancellation and slow consumers

Existing prompt content and token budgets were reused. Recovery time below
is the next full decision after cleanup, not just abort acknowledgement.

| Scenario | Next decision seconds | Probability difference | Native pending after recovery |
|---|---:|---:|---:|
| Direct decision cancelled after GPU prefill admission | 1.034 | 0 | 0 |
| SSE chat disconnected after emitted text | 1.017 | 0 | 0 |
| Nonstreaming chat disconnected while native work was active | 1.050 | 0 | 0 |
| Reasoning request disconnected at the existing 4096-token budget | 1.041 | 0 | 0 |
| Adapter stream left unread while native generation completed | 1.018 | 0 | 0 |

The last case kept the async stream consumer open without reading its remainder,
then verified a healthy HTTP decision could enter after native generation
finished. It tests adapter admission independently of the consumer; it is not
a claim to exhaust a real TCP send buffer. Cache flush before the final
unhealthy-engine test succeeded.

## Abort acknowledgement never arrives in time

A final, test-only 65-second scheduler delay exceeded both unchanged windows.
The readout request failed at **60.091 seconds**, the engine marked itself
failed, and **`/health` returned 503**. A subsequent decision was rejected with
HTTP 500 and **no new native request was submitted**. This is fail-closed
behavior requiring a restart, not automatic recovery. Native pending requests
were zero once the delayed backend returned. The engine was then shut down;
a fresh engine ran the frozen suite below.

## Frozen parity suite

The original comparator, reference, inputs and profiles were reused.

| Comparison | Tolerance | Passed | Overall result |
|---|---:|---:|---|
| Default strict | 0.005 | 0/40 | Failed |
| Explicit `self-spec-bf16` profile | 0.02 | 38/40 | Failed |
| New v2 candidate vs earlier candidate | Exact stored probabilities | 345/345 identical | No numerical change |

The same two opt-in failures remain: `ambiguous-23` at **0.0205364525** and
`support-33` at **0.0204840899**. Prompt/head hashes and row lengths matched.
No tolerance or fixture was changed to obtain a pass. This run does not close
#81 or provide a new completed scored-reasoning/accuracy result.

## Remaining limits and checks

- An initial **cold** first request exceeded the unchanged 30-second deadline
  and returned 500. That failed attempt was retained. After the kernel cache
  was populated, a fresh process's first request took 7.676 seconds and the
  checks above passed. Cold-start readiness/warm-up remains an operational
  limitation; this test did not add a startup warm-up or raise the timeout.
- The hard-stall case intentionally needs restart; normal cancellation and the
  shorter delayed acknowledgement recovered without one.
- These are bounded, single-worker tests with injected metadata failures.
  Long-duration operation, distributed execution and every scheduler path
  have not been established. Existing serial/text-only restrictions remain.
- **225 focused CPU regressions passed again.** The backend's four patched
  source files had identical hashes before/after the run, and both fixture
  files were unchanged. No test hook was left in the serving implementation.
- After both GPU processes exited, GPU memory returned to zero. The serving
  code's full CI and earlier MLX checks are recorded in the parent validation
  notes; this follow-up adds GPU functional evidence, not a speedup claim.
