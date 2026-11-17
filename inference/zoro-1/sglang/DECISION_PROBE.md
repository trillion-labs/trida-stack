# Self-spec decision validation

Experimental support for [#81](https://github.com/trillion-labs/trida-stack/issues/81).
One model serves text decisions, reasoning and chat. **Parity is not accepted.**
The frozen requests, comparison profiles and tolerances have not changed.

## Results and limits

Historical H100 checks used BF16 and one eager worker. The checkpoint was a local
Zoro-1 checkpoint, not a fresh independently verified Hub download. The stock
reference used SGLang 0.5.23 / PyTorch 2.14.1; the candidate used HybridDiffusion
`6ca547a` plus the two overlays / PyTorch 2.9.1+cu128.

| Check | Result |
|---|---|
| Original short text | Max probability gap 0.004221; confidence gap 0.0060, so full-answer parity failed |
| Original long text (7,566–7,584-token rows) | Max probability gap 0.009739; failed strict 0.005 |
| Same-fork causal vs self-spec | Short/long probability gaps 0.008859 / 0.006207; still failed strict 0.005 |
| Readout audit | All 22 prefix endpoints/lengths matched; BF16-to-FP32 export was exact |
| Frozen 40-case suite | 38/40 passed the explicit 0.02 profile; the profile as a whole **failed** |
| Images | Unsupported; no valid image parity result |

The 40 cases contain 120 questions and 345 probabilities. Mean absolute
probability difference was 0.004969; the largest was 0.020536. The two rejected
cases were `ambiguous-23` (0.020536) and `support-33` (0.020484). One near-tie
choice changed (`routing-21`, reference margin 0.002114); score modes and auto
thresholds also changed near ties. These are synthetic parity checks, not accuracy
benchmarks or LLM-judged examples. The default comparator remains **0.005**;
`--profile self-spec-bf16` explicitly selects **0.02** and checks derived answer
fields against their own probabilities. Neither threshold was raised to pass.

Historical HTTP checks covered direct/auto/completed reasoning, one-token fallback,
limits, streaming, usage and cancellation recovery around one second. Native
requests drained and cache flush completed. This does not establish long-duration
stability, distributed execution, concurrent readout, or zero-token termination.

## Latency observations

A fixed three-question fixture had rows of 5,537 / 5,550 / 5,555 tokens. Sampling
was temperature 1, top-p 0.95, top-k 20; memory fraction 0.5 and context 16,384.
Stock used batched last-state prefixes; self-spec used serial eager prefixes.
Runtime/kernel versions differed. Warm direct latency was 0.117–0.133 s on stock
versus 1.064–1.559 s on self-spec.

| Backend | Budget | Actual reasoning tokens | Reasoning wall s | Wall tok/s | Total HTTP s |
|---|---:|---:|---:|---:|---:|
| Stock AR | 128 | 128 | 5.287 | 24.2 | 5.654 |
| Self-spec | 128 | 128 | 2.359 | 54.3 | 3.470 |
| Stock AR | 512 | 38 | 1.195 | 31.8 | 1.991 |
| Self-spec | 512 | 344 | 5.677 | 60.6 | 8.673 |
| Stock AR | 1024 | 157 | 4.779 | 32.9 | 5.336 |
| Self-spec | 1024 | 196 | 2.897 | 67.7 | 5.783 |

These are single observations, including prompt preparation and prefill.
Sampled lengths differ, so this is **not a general end-to-end speedup claim**.

## Current safety and reasoning contract

See the [shared contract](../README.md#reasoning-contract) for cuts, token usage,
completion, auto thresholds and configured budgets. The canonical route is
`/v1/systemone`; `/v1/decide` remains a deprecated alias. Images return 422.
The native engine is serial and sampling follows the YAML. Capabilities and
reasoning sampling are reported to clients.

The version **2** readout overlay marks only whole causal-prefill results and
checks each endpoint before exporting a hidden state. Mismatches abort the affected
request rather than raising inside the scheduler. Readout waits have a 30-second
deadline; cancellation drains the native task. Abort cleanup retries for at most
30 seconds and marks `/health` unavailable if acknowledgement never arrives.
Generation owns streaming admission; a slow reader does not keep the lock after
native generation finishes. Disconnected nonstreaming chat is cancelled too.

These review fixes passed 225 focused CPU tests, 63 MLX tests (2 skipped),
and 21 benchmark adapter/setup tests. Both overlays applied to the pinned
source and parsed successfully. The historical GPU results above used the
earlier overlay. [Readout v2 H100 validation](READOUT_V2_VALIDATION.md) now covers
request-level failure isolation, cancellation, deadlines and fail-closed health.
All 345 frozen-suite probabilities matched the earlier candidate exactly;
strict 0.005 still failed and the explicit 0.02 profile remained 38/40.
The initial cold-start timeout is documented there. No result here changes the
two parity failures or closes #81. Latest main is integrated, including
#89 and the `decision_head.pt` / legacy `jev_head.pt` loading convention from #97.

## Reproduce

Build separate stock and diffusion environments using [the recipe](README.md).
Apply `two_stream_diffusion.patch`, then `decision_readout.patch`, to the exact
`6ca547a` checkout. Use `SGLANG_PYTHON` for the diffusion interpreter. The server
refuses older/missing readout versions. Both patch applications are checked in CI.

Run from `inference/zoro-1/sglang`, substituting the same local checkpoint for both:

```bash
python tools/parity_cases.py cases.json
STOCK_PYTHON tools/probe_decision_prefill.py reference --checkpoint CHECKPOINT \
  --requests cases.json --output reference.json
DIFFUSION_PYTHON tools/probe_decision_prefill.py candidate --checkpoint CHECKPOINT --backend DIFFUSION_EVAL \
  --requests cases.json --output candidate.json
STOCK_PYTHON tools/probe_decision_prefill.py compare \
  --reference reference.json --candidate candidate.json --output strict.json
STOCK_PYTHON tools/probe_decision_prefill.py compare --profile self-spec-bf16 \
  --reference reference.json --candidate candidate.json --output provisional.json
```

Use the same immutable case file, checkpoint and tokenizer for both processes.
Compare request/token/head hashes before interpreting probabilities. Keep input
and output artifacts outside the repository. `TRIDA_DECISION_READOUT_AUDIT=1`
adds exact endpoint/export checks. CPU tests use fake native engines and tiny MLX
fixtures; they do not substitute for GPU parity or performance measurements.
