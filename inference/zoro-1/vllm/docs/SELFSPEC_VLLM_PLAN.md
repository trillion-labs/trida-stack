# Self-spec (AR-Trust) on vLLM — port plan (2026-09-09)

Goal: bring FLARE's AR-Trust decoding (`HybridDiffusionSelfSpec` in the SGLang
reference) to the vLLM native plugin, so the fast, lossless mode runs on the
engine whose AR path already does 1,633 tok/s @C=8 (H100).

## Reference semantics (from `hybrid_diffusion_self_spec.py`, variant `bd_bidir_shift`)
- Canvas = `blk = 2N-1` slots, N = `gen_block_size`. One forward per cycle.
  - Verify round: `[t0(pending clean), spec_0..spec_{K-1}, MASK x (N-1)]`
  - Cold start:   `[t0, MASK x (2N-2)]`
- Attention inside the block: **clean rows 0..N-1 are CAUSAL** (see prefix + earlier
  clean slots only); **MASK rows N..blk-1 attend to everything** in the block. Prefix
  is causal as usual. => verify logits are exact AR logits (lossless).
- GDN: `causal_mode=2, num_clean=N` (clean slots token-causal readout, masks
  block-end readout), intermediate per-step states cached; after verification the
  state **after the accepted prefix** is scattered into the persistent cache
  (`commit_cached_intermediate_states_batch` -> `fused_mamba_state_scatter_with_mask`
  for ssm AND the conv window). Nothing is persisted by the forward itself.
- Greedy verify (shift readout: logit at slot i predicts slot i+1):
  accept spec_i iff `argmax(logit_i) == spec_i`, left to right.
  - reject at i: output = accepted specs + corrected token `ct = argmax(logit_i)`;
    next canvas = cold start `[ct, MASK...]` (`_force_next_token`).
  - all K accepted: `clean = argmax(logit_K)` (slot of last spec); new specs =
    argmax of the mask slots (shifted); output = specs + clean (N tokens); next
    canvas = `[clean, new_specs..., MASK x (N-1)]`.
  - cold start: output = `[t0, clean]` (t0 if not yet emitted) ; specs from masks.
- KV of MASK / rejected slots is freed after the forward; accepted slots keep KV.
- Tokens per forward = 1 + accepted specs (1..N). Speed vs AR = tok/fwd x (AR step / spec step).

## vLLM design
1. **Mode switch**: `decode_mode=selfspec`, `gen_block_size=N`; canvas_length = 2N-1.
   Every step is one forward; no separate commit pass; no denoise rounds.
2. **Canvas/state per slot**: `pending` token, `specs[K]`, `forced` flag, `t0_emitted`.
   Build draft_tokens = `[t0, specs, MASK...]` each step (fixed shape 2N-1).
3. **Attention**: vLLM FA3 has per-request causal/bidirectional only (PrefixLM
   ranges need FA4). Phase 1 uses **causal=True for the whole canvas**: clean rows
   exactly as the reference (lossless verify preserved); MASK rows lose the
   look-ahead to later masks (weaker drafts => lower acceptance than SGLang).
   Phase 2 option: Triton/Flex attention for the 8 attention layers to reproduce
   the reference mask exactly.
4. **GDN**: packed kernel (`fused_recurrent_block_causal_gated_delta_rule_packed`)
   with `causal_mode=2, num_clean=N`, `intermediate_states_buffer[R, N, HV, V, K]`,
   `cache_steps=N`, `output_final_state=False`. Conv: denoise-style work buffer
   (Fix B2), plus save the block's pre-conv rows per layer to rebuild the conv
   window of the accepted prefix. After the sampler decides `k` per request:
   fixed-shape masked gather of `intermediate[slot, k]` -> ssm cache row, and of
   the pre-conv rows -> conv cache row (24 layers, no host sync).
5. **Sampler** (GPU, fixed shape): argmax over canvas logits; `acc = cumprod(argmax[:K]==specs)`,
   `k = sum(acc)`; corrected/clean token; new specs from mask slots; `num_sampled`
   per request = accepted + 1 (+1 for un-emitted t0); write next canvas into
   `req_states.draft_tokens`; existing `_build_output` handles variable
   `num_sampled` (vLLM spec-decode output path). Rejected slots' KV are simply
   recomputed next step (positions beyond the committed length are reused).
6. **Diagnostics**: extend the JSONL tracer (accepted count histogram, cold-start
   rate, tok/fwd).

## Validation (same Slurm tooling)
1. **Lossless oracle**: greedy AR-Trust output must equal vLLM causal greedy output
   token-for-token -> `compare_identity.py` vs `runs/20260909_143259/vllm-causal-clean`
   (1319 items). Any mismatch = bug.
2. tok/fwd + accepted-specs histogram from the tracer; compare with SGLang self-spec
   (`runs/spec_20260909_170829`, g4/g8/g16) to quantify the causal-mask draft penalty.
3. Concurrency sweep C=1/4/8/16 vs vLLM AR (`sweep_client.py`).

## Effort / risks
- ~2-3 days implementation + validation cycles. Risks: (a) Phase-1 mask deviation lowers
  acceptance (measured against SGLang); (b) conv-window rebuild correctness (oracle catches
  it); (c) intermediate-state buffer memory (R x N x 2 MB x 24 layers; fine for R<=16, N<=8).
- Fix B (B1+B2 accepted, B3a pending full-set accuracy) carries over: the self-spec step
  reuses the packed kernel and the conv work buffer.

## Phase 2 (2026-09-10): step cost → CUDA graphs
Measured after S2/S3: GPU busy 5.1 ms/forward (GEMM 3.7, fused GDN 0.6, elementwise 0.4, sampler 0.2, attention 0.02),
host dispatch ~6.5 ms (156 eager launches + 32 piecewise graph launches). Path to the AR-step floor:
- S3 (done): fused kernel writes readout + pre-conv rows directly (−48 launches).
- S4: present the self-spec step as a uniform spec-decode batch (bonus token + 2N−2 drafts) with spec-classified
  attention metadata (inject `num_speculative_tokens=2N−2` for the GDN/FA builders; supply
  `num_decode_draft_tokens_cpu`/`num_accepted_tokens` via ModelSpecificAttnMetadata like upstream mamba_hybrid) so
  vLLM's FULL_AND_PIECEWISE captures the whole forward. Target: step ≈ 5–6 ms → ~2× vLLM AR at C=1.
- S5: single-kernel sampler (argmax + verify + window update) to drop the remaining ~59 eager launches.

### S4 outcome (2026-09-10)
Done and validated (commit a998a15): step 7.5 ms (fwd 6.16 / sampler 0.90 / snap 0.41), 30/30 identical to S3 under FULL,
C=1 248 tok/s vs vLLM AR 219. Two engine facts worth remembering: the scheduler hard-codes 0 sampled tokens/step for
diffusion models (shimmed to 1 in spec-shape), and `post_update` advances `num_computed_tokens` by `query_len − num_rejected`,
so prefill must report `num_rejected = 0` when it emits the seed as the bonus token.
Next: S5 single-kernel sampler (−0.9 ms), fold the snapshot into the fused kernel (−0.4 ms), then mask fidelity for +11–14% tok/fwd.
