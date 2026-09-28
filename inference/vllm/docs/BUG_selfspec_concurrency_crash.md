# Self-spec dies under concurrency: `operation not supported on global/shared address space`

Status: **root cause NOT established.** Characterised precisely, five mechanisms proposed and four
disproved. This is a handover, not a diagnosis.

## Repro

```
CKPT=.../qwen35-4b-flare-v6-2n-fullmask-bd8/step_2000
TRIDA_SELFSPEC_N=8  CL=14  THRESH=0.90  MAXSTEPS=8  --max-num-seqs 16
sweep_client.py --limit 48 --concurrency 2 --max_tokens 256 --no_sampling
```

Dies after 12 of 48 requests. **Deterministic**: six repeats, identical counts.
Script: `tools/experiments/bisect_crash.sh`, `tools/experiments/shape_probe.sh`.

## Fault

Under `CUDA_LAUNCH_BLOCKING=1` it names itself. Without it, an async illegal-access whose traceback
points at an unrelated call.

```
qwen3_5_diffusion.py:1901 _trida_gdn_forward_core
qwen3_5_diffusion.py:1829 _trida_gdn_selfspec_core
vllm/.../qwen_gdn_linear_attn.py:1377 _forward_core          <- the STOCK kernel, i.e. the fallback
vllm/.../fused_sigmoid_gating.py:242  fused_sigmoid_gating_delta_rule_update
triton/runtime/jit.py:761 run
RuntimeError: Triton Error [CUDA]: operation not supported on global/shared address space
```

## Established by measurement

| claim | evidence |
|---|---|
| Deterministic, not a race | 6/6 repeats identical (job 3904) |
| Requires cuda graphs | `--enforce-eager` clean 3/3; costs 3x throughput (122 vs 383 tok/s) |
| Not the compile cache | wiping it changes nothing, 3/3 still crash |
| Not an index overflow | bounds guards on `n_pad`, `sl.max()` vs `Rn`, and `num_decode` vs UVA capacity all stay SILENT through the crash |
| Not chunked prefill | `--no-enable-chunked-prefill` gives identical mixed-batch counts and the identical fatal shape |
| Mixed batches involved but NOT sufficient | C=4 sees 74 mixed batches and stays clean |
| The fallback DOES fire | "WARNING mixed batch" present; the fault is inside the stock kernel it delegates to |

### Concurrency pattern (unexplained)

| C | outcome | mixed-batch steps |
|---|---|---|
| 1 | clean | 0 (client never overlaps prefill with decode) |
| 2 | **crash** 12/48 | 66 |
| 4 | clean | 74 |
| 8 | **crash** 24/48 | 60 |

### The fatal step (`TRIDA_SHAPELOG=1`)

```
step=768  n=1 num_reqs=1 n_pad=1 tokens=15   k=[7]      normal decode (15 = canvas width)
step=769  n=1 num_reqs=1 n_pad=1 tokens=80   k=[7]      prefill mixed in
step=770  n=2 num_reqs=2 n_pad=2 tokens=137  k=[0,0]    DIES
```
Pure decode batches (15 tokens/request) ran 700+ times cleanly.

## Disproved hypotheses

Recorded so nobody re-walks them: a race on batch composition; a per-slot buffer overrun via the
`slot+1` convention; Triton pointer-alignment specialisation against a cached binary; a stale
compile-cache artifact; a Python fallback branch trapped inside a captured graph (our GDN core is a
registered splitting op, so it runs OUTSIDE the graph and the branch does execute).

## Known-wrong-by-design code this touches

`_trida_gdn_selfspec_core` falls back to the stock kernel on a mixed batch, with its own comment
saying "state persistence would be wrong -> warn loudly". Since Fix S the recurrent state lives in a
private ring and is NOT written back to the kv-cache, so the stock kernel reads state that was never
populated. Note that *fixing the crash by syncing ring->cache would produce silently WRONG output*,
because the stock path applies token-causal readout where the canvas needs block-end readout. The
correct fix is to handle mixed batches in our own path using per-request token offsets, running the
stock path on prefill rows and the fused kernel on canvas rows.

## E13 result (job 3908): the fallback's boolean metadata does NOT discriminate

Logged every fallback entry at layer 0 in a surviving run (C=4, 14 fallbacks, clean) and a crashing
run (C=2, 9 fallbacks, died after #9). Every entry in both runs has `num_decodes=0`,
`has_spec_masks=True`, and all three state-row tensors present. Only `num_prefills` differs (C=4
sometimes has 3), which is batch size, not a cause.

What it does establish: the stock kernel is handed our canvases classified as **speculative decodes**
(`has_spec_masks=True`, `num_decodes=0`) and will index `spec_state_indices_tensor` rows in the
kv-cache. Our ring never writes those rows. That is consistent with the fault but does not yet
explain why C=4 survives the same classification.

Next refinement (job 3910, `[FALLBACK-ROWS]`): print the actual row VALUES and `cache_rows`, and the
query-start-loc tensors. A row `>= cache_rows`, or row 0 (the reserved null block) where a real row is
expected, is the kind of thing that discriminates where booleans cannot.

## E13b result (job 3910): row VALUES do not discriminate either

| | crashing (C=2, fallback #9) | surviving (C=4, fallback #14) |
|---|---|---|
| cache_rows | 2144 | 2144 |
| spec rows | [57] | [185] |
| nonspec / prefill rows | [61] | [193, 197, 189] |
| spec_query_start_loc | [0, 15] | [0, 15] |
| nonspec_query_start_loc | [0, 122] | [0, 65, 136, 214] |

Every row is in bounds, none is the reserved null row 0, and both are structurally the same thing:
**one canvas classified as a spec decode plus N genuine prefills**. Note `num_decodes=0` throughout --
vLLM never sees our canvases as decodes.

**Conclusion of the diagnostic line: the fault is not in the fallback's inputs.** It is downstream, in
what the stock kernel does with kv-cache state rows our ring never wrote; whether that faults depends
on whatever stale contents those rows hold, which is why it is deterministic per configuration yet
differs between configurations. No further logging will shortcut this. The fix is the mixed-batch
path (Phase 2): route prefill rows to the stock path and canvas rows to the fused kernel using the
per-request offsets that `spec_query_start_loc` / `non_spec_query_start_loc` already provide.

## Next diagnostic (not another hypothesis)

Instrument the fallback itself: log every entry with the batch shape AND the state rows handed to the
stock kernel, then diff a C=2 entry against a C=4 entry. Those two look alike in shape and differ in
outcome; that difference is the bug.

## Scope of impact

- Losslessness at C=1 is **positively verified**: the E1/E2/E3 runs recorded zero mixed batches.
- Losslessness at C>1 is **unverified**, and 8.6% of steps at C=2 are mixed.
- Self-spec cannot serve above C=1 today. This blocks measuring the AR-vs-self-spec curve past the
  crossover at C=4 (see `selfspec-is-a-latency-win-not-throughput`).

---

# Throughput loop (separate from the crash): P0/P1 findings

**Target (gate-checked):** self-spec pays +2.13 ms/forward over AR at C=1, entirely GPU idle; our
kernels are 0.49 ms CHEAPER. The device waits on the host.

**P1 Fix 1 (shipped, lossless):** removed a per-forward `.item()` sync in prepare_attn via a numpy
mirror `sspec_started_np`. Syncs 1.62 -> 1.00 /fwd, wall +2.13 -> +2.00 ms. Byte-identical verified.

**Finding that redirects the loop:** syncs are NOT the bottleneck (they explain ~0.3 of 2.4 ms idle).
The GPU idles on **133 eager kernel launches/forward** (AR: 37). Attributed by `launch_sites.py`:
  prepare_attn      54.5 /fwd   (per-forward metadata: index buffers, causal buf, RR setup, fills)
  _selfspec_step    44.0 /fwd   (verify math: argmax/gather/cumprod/where/scatter on [nd, blk])
NOT the 24-layer GDN loop (that is one fused kernel). So Fix 2 = fuse/batch the many small tensor ops
in these two functions. Correctness-critical (touches verify), so it needs the full
measure->byte-identical->wall cycle per change.

Tools: attribute_overhead.sh + diff_profiles.py (per-fwd kernel diff, idle from bench wall),
sync_sites.py (sync->frame), launch_sites.py (launch->frame), host_attrib.py, verify_lossless.sh.
