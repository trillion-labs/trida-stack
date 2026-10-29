# vLLM diffusion profiling — findings & fixes (2026-09)

Root-cause investigation of "vLLM block-diffusion under-commits / degenerates at
large canvas," run as a research → profile → verify loop on node1 (8×H100),
checkpoint `qwen35-4b-flare-v6-2n/step_18000`. All numbers: C=1, greedy, 128
tokens, best of 3.

## TL;DR

The port was **not** a forward bug. The slow/degenerate behavior came from **one
stashed regression + two config values**. Fixing them:

| | tok/s | coherent |
|---|---:|:---:|
| vLLM diffusion **before** (CL=3, thr 0.95, run-both variant) | 15.9 | ✓ |
| vLLM diffusion **after** (CL=32, thr 0.90) | **63.6** | ✓ |
| vLLM diffusion **after** (CL=32, thr 0.80 — speed lever) | **77.7** | ✓ |

~4× faster and the large-canvas garbage is gone. Committed in the same change as
this doc.

## What was actually wrong (all config / regression — not the forward)

1. **The box was running the rejected "run-both readouts" variant** (`37c947f`),
   a FULL-cudagraph workaround that runs both GDN readout kernels every step and
   mask-selects. It halves throughput regardless of cudagraph mode. The first
   sweep's tok/s were ~2× depressed by it. Restored the clean committed forward.

2. **`max_denoising_steps` < canvas masks → premature forced-flush.** On the last
   allowed denoise step the gate does an *unconditional* flush (`transfer = m`),
   committing every still-masked position regardless of confidence. With
   `max_denoising_steps=16` on a 31-mask canvas, ~15 low-confidence positions got
   force-committed as garbage → the CL=32 "looping / degenerate" output. SGLang
   avoids this by setting `step_budget = block_size`. Fix: `max_denoising_steps >=
   masks` (we ship 64 for CL=32).

3. **`canvas_length=3` leaves the diffusion speedup on the table.** Big blocks are
   the throughput regime (SGLang's fast path is bd32). Measured (thr 0.90):
   CL=3 → 16, CL=8 → 41, CL=16 → 59, CL=32 → 64 tok/s.

4. **`confidence_threshold=0.95` was stricter than SGLang's 0.90** → fewer
   positions clear the gate per forward. Confidences cluster *just below* 0.90, so
   dropping to 0.80 commits the whole cluster for ~+20% tok/s (78) while staying
   coherent across diverse prompts — at a small quality risk (commits before the
   left context fully settles). Ship 0.90 (SGLang parity); 0.80 documented lever.

## Verified correct (against SGLang source + experiments) — NOT the gap

- **Commit gate is byte-identical to SGLang** `LowConfidenceShiftHybridDiffusion`:
  `transfer = conf > threshold` (parallel over all masked positions) + force-top-1
  fallback + last-step flush. There is **no schedule** (`num_transfer_tokens`) in
  either implementation — both are pure threshold.
- **GDN denoise readout already matches**: `causal_mode=2, num_clean=1` (seed reads
  its own token-causal state; masks read the block-end state).
- **Bidirectional within-block attention is engaging.** A/B: forcing denoise causal
  (`TRIDA_FORCE_CAUSAL_DENOISE=1`) *degrades* output and collapses the confident
  step from `[0.95,0.89,0.89,0.85,0.84]` (5 near-threshold) to `[0.98,0.47,…]` (1).
  So `causal=False` really does produce bidirectional attention on this H100/FA3
  build.
- **Confidence definition matches**: fp32 softmax-of-argmax, logit-shift (position
  `i` scored by logit `i-1`).

## Throughput matrix (all modes)

| Backend | Mode | Config | tok/s | Coherent |
|---|---|---|---:|:---:|
| SGLang | causal (AR-Trust) | cudagraph bs=1 | 209.6 | ✓ |
| SGLang | self-spec | b7_g4 | 243.9 | ✓ (lossless) |
| SGLang | diffusion bd4 | block_size 3, thr 0.90 | 55.7 | ✓ |
| SGLang | diffusion bd32 | block_size 31, thr 0.90 | 144.2 | ✓ |
| vLLM | causal (AR) | stock, cudagraph on | 207.9 | ✓ |
| vLLM | diffusion (before) | CL=3, thr 0.95, run-both | 15.9 | ✓ |
| vLLM | diffusion (after) | CL=16, thr 0.90 | 58.8 | ✓ |
| vLLM | diffusion (after) | **CL=32, thr 0.90** | **63.6** | ✓ |
| vLLM | diffusion (after) | CL=32, thr 0.80 | 77.7 | ✓ |

Note the naming off-by-one: SGLang `block_size=k` = `k+1` canvas positions
(carried-seed), so SGLang bd32 ≈ vLLM CL=32. Rows aligned by actual positions.

## Residual gap & how to close it

vLLM 63.6 vs SGLang bd32 144.2 tok/s ≈ **2.2× wall-clock** (NOT the "15× tokens/
forward" seen earlier — that was a cross-server counter artifact). The remainder
is **per-forward wall time**: vLLM runs the GDN op eager under PIECEWISE cuda-graph
while SGLang graphs more of the forward. Plus SGLang's marginally-higher-confidence
forward clears more of the near-threshold cluster.

Closing it requires the **FULL cuda-graph path**, which needs a single
`causal_mode`-tensor GDN kernel (denoise vs commit selected by a device tensor, not
a Python flag). The "run-both-readouts" shortcut was already tried and rejected
(halves throughput, not byte-identical). This is a larger lift; SGLang remains the
speed champion until then. vLLM is now a **fast, coherent, correctness-parity**
diffusion backend.

## Diagnostics (env-gated, inert by default)

- `TRIDA_DUMP_CONF=1` — per-position denoise confidence dump each commit round.
- `TRIDA_FORCE_CAUSAL_DENOISE=1` — force denoise full-attn causal (the A/B probe).

Raw artifacts on node1: `/path/to/{dump_conf,fix_test,ab_causal,thr_sweep,coh}_result.txt`.
