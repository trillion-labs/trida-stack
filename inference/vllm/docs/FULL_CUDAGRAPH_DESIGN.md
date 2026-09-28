# FULL cuda-graph for two-stream diffusion — design & decision (2026-09)

Goal: move the GDN two-stream readout from *eager under PIECEWISE* to *captured
under FULL* so the whole forward is graphed. This doc records the exact
degeneration mechanism, the complete fix design, the measured payoff, and the
recommendation.

## Why FULL_AND_PIECEWISE currently degenerates (root cause, from serve log)

- The denoise step IS a uniform spec-decode batch (`num_draft_tokens=32,
  all_noisy=True, mode=denoise`) → vLLM captures ONE FULL graph at that size.
- **cuda-graph capture runs with `for_capture=True`**, so `prepare_attn` leaves
  the phase flag `"other"` → `_trida_gdn_forward_core` falls through to the
  **stock token-causal** GDN readout. The FULL graph therefore **bakes in the
  commit (token-causal) readout**.
- At replay, denoise and commit share one `BatchDescriptor` (no phase field), so
  both replay that stock-readout graph. Denoise needs the block-end (mode-2)
  readout → gets token-causal → **garbage/looping output**.

## The fix: one device-tensor-selected kernel (not run-both)

Key simplification found in the kernel: **commit == `causal_mode=2` with
`num_clean=T`** (all positions clean ⇒ all token-causal; the block-end phase-2
loop is empty). So `num_clean` alone parameterizes every readout:
`num_clean=1` → denoise (seed token-causal, masks block-end); `num_clean=T` →
commit (token-causal); `num_clean=0` → pure block-end. `causal_mode` is subsumed.

### (a) Unified Triton kernel (`block_causal_readout.py`)
- Drop `CAUSAL_MODE`/`NUM_CLEAN` `tl.constexpr`; add device pointers `num_clean`
  (int32[N]) and `persist` (int32[N]). Load per-seq near `cache_idx`.
- Single always-run recurrence loop (mode-independent), then **two predicated
  readout stores**: token-causal store masked by `t < num_clean`, block-end
  store masked by `t >= num_clean`. Both instructions always emitted; the device
  scalar only toggles the store mask → fixed instruction stream, graph-safe.
- Final-state store: keep the `tl.store(ht)` instruction always present (allocate
  `ht` at capture), mask the actual write by `persist`.

### (b) Graph-safe conditional persist
- `persist = is_encoder_phase[slots].int()` (already a device tensor — no host
  sync). Commit rows persist; denoise rows don't.
- ssm/conv persist via fixed-shape `torch.where(persist, final, cache)` masked
  scatter — same pattern as `_gdn_snapshot_restore`. Because that restore runs
  **eagerly and idempotently** each step, an unconditional-under-mask persist is
  safe: denoise's stray write is reverted next step; commit's is kept
  (`is_enc=True` ⇒ `snap_valid=False`).

### (c) Remove host syncs in `prepare_attn`
- Delete `all_noisy = bool((~causal).all().item())` (:829) and
  `max_query_len=...max().item()` (:854). Derive `num_clean`/`persist` device
  tensors directly from `causal = is_encoder_phase[slots]`
  (`num_clean = where(causal, T, 1)`), and pad `max_query_len` to the static
  canvas length.

### (d) Denoise → uniform-decode reclassification (the big one)
The canvas is currently scheduled as a **prefill chunk** (`num_prefills>0`), and
GDN's FULL capture is **decode-only**, so denoise never enters the capture set
cleanly. Present the canvas as a **spec-decode batch** (`spec_sequence_masks`
set, `num_decodes>0`, `num_prefills==0`) so GDN takes the multi-query decode
path (`fused_sigmoid_gating_delta_rule_update`) that IS decode-classified. This
touches runner metadata construction + sampler assumptions — a runner-level
change, not just the kernel.

## Risks (ranked)
1. **HIGH — commit numerics.** Swapping the stock `chunk_gated_delta_rule`
   commit for the recurrent kernel changes bf16 accumulation order. Model is
   precision-sensitive (diffusion already not lossless). Mitigation: run the
   ENTIRE path through the one recurrent kernel (single numeric regime); CPU-
   validate block-boundary state drift BEFORE GPU.
2. **HIGH — uniform-decode reclassification** is a runner-level surface
   (metadata + sampler), the true gate to decode-only FULL capture.
3. MEDIUM — conv-state decode (`causal_conv1d_update`) vs prefill path parity.
4. MEDIUM — predicated-store cost (smaller than run-both: recurrence runs once,
   only the cheap `q@S` store is duplicated). Measure vs PIECEWISE.
5. LOW-MED — `persist`/`snap_valid` must stay derived from one `is_encoder_phase`.
6. LOW — static/padded `T` under capture.

## Measured payoff (the decision input)
CL=32 (shipped), C=1, greedy 128 tok, clean code:
- PIECEWISE = **62.5 tok/s** (coherent)
- FULL_AND_PIECEWISE = **67.6 tok/s** (degenerate today; ~the coherent ceiling)
→ **~+8%** at single stream. FULL cuda-graph only saves kernel-launch overhead,
which is a small fraction at compute-bound CL=32 (the big-canvas fix already
banked the launch-overhead savings that made CL=3 look like 1.85×). It does NOT
close the SGLang gap (144) — that lives elsewhere (GDN kernel efficiency /
tokens-per-forward), not cuda-graph. Concurrency would amortize launch overhead
more, BUT diffusion serving is currently `max_num_seqs=1` (single-stream);
batched diffusion is itself a separate lift.

## Recommendation
**Shelve implementation; keep this design ready.** ~+8% single-stream is a poor
return on a change with two HIGH risks (commit numerics + runner-level
reclassification). Revisit if/when (a) batched diffusion serving becomes a goal
(the reclassification work is shared and concurrency makes FULL pay), or (b) the
GDN-kernel-efficiency gap to SGLang is closed first (making launch overhead a
larger relative share). Until then, PIECEWISE CL=32 (62.5 tok/s, coherent) is the
shipped path.
