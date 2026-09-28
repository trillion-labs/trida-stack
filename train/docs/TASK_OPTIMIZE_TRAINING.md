# Task: optimize the training code

**Owner:** luke · **Status:** round 1 done 2026-09-10 (cheap wins measured, defaults updated); round 2 = kernel work · **Priority:** next

## Why

The FLARE two-stream trainer (`train/train.py` → `HFBlockDiffusionHybrid.forward_flare`) runs at ~8% MFU.
Every retrain (block-32 retrain, draft-aligned fine-tune, control runs) pays this 4–5× tax.

| Measured (job 3726, 16×H100, micro 1 × accum 2, packed 32k multi-turn, bd 8) | |
|---|---|
| tokens/step (16 GPUs × 2 × ~12.2k) | ~390k |
| throughput | 15–27k tok/s, avg ~20k (alternates per 5-step window) |
| step time | ~20 s |
| FLOPs/token (8N with checkpoint recompute, ×2 streams) | ~64 GFLOP |
| achieved per GPU | ~80 TFLOP/s ≈ 8% of 990 dense bf16 (4% if the 2nd stream is not credited) |
| hyungguk's v6-2n "nooff" runs | 17–26k tok/s — same code, same speed |
| interconnect | TCP vs RoCE/IB made no difference at this scale (compute-bound) |

Target: 35–45% MFU (typical dense-4B FSDP fine-tune) → 3–5× faster steps.

## Facts already established (do not re-derive)

- `fla` 0.5.1 fused `chunk_gated_delta_rule` IS bound (transformers 5.12 binds it even though
  `is_fast_path_available` is False). `causal_conv1d` and `flash_attn` are NOT installed in
  `/path/to/trida-stack/.venv`; the short conv runs the torch fallback.
- Attention: 8 layers, `flex_attention` via transformers' integration (it torch.compiles the flex call).
- Two-stream GDN: FLARE Triton kernels vendored in `train/block_gated_delta_rule/` (`block_train_method='auto'`
  → `chunk_wy_triton_fla_style` for block_size < 16, `chunk_refine` for ≥ 16). Conv: `block_train_conv`
  (`QWEN35_BLOCK_CONV_METHOD` auto → fla_batched | twostream).
- Gradient checkpointing on every decoder layer (`checkpoint(lambda x, l=layer: run(l, x))`), recomputes the full
  two-stream forward in backward.
- `--activation_offload` is 6× slower (job 3719) — keep off. `--compile_layers` / `--compile_mlp` / `--compile_glue`
  exist but are unused in the production recipe.
- Collator: `PackedMultiTurnCollator` packs 11 conversations/rank/step into ≤32k rows on the fly (num_workers 8).
- Never write into hyungguk's venv; build a separate env (`luke/env/…`) for kernel installs.

## Plan (ordered by expected payoff ÷ effort)

1. **Profile one step** (torch.profiler, 1 node is enough — FSDP over 8 ranks, same per-rank work). Attribute GPU time to:
   two-stream GDN fwd/bwd, flex attention, MLP/CE, conv fallback, checkpoint recompute; CPU time to collator + Python glue;
   idle gaps (the 27k/15k alternation). Deliverable: a table of % per bucket. ~1 h incl. sbatch.
2. **Cheap wins found by (1)**, in this likely order:
   - collator stalls → prefetch/pre-pack the dataset offline (rows written once, read by every run);
   - `torch.compile` of MLP + norms (`--compile_mlp`, shape-stable) — measure, not assume;
   - install `causal_conv1d` in a luke env → fused short conv;
   - selective checkpointing (checkpoint the GDN layers, not the MLP) if memory allows (49.5 GB used of 80).
3. **Kernel-level**: compare `block_train_method` variants for the bd in use (bd 8 pilot vs bd 32 production);
   FLARE's `_profiling.py` has hooks. Only if (1) shows GDN dominating.
4. **Interconnect**: `NCCL_IB_PKEY=1` on mlx5_2/3/4 (43 GB/s/card) or RoCE mlx5_0/1 — free once the probe job
   (3734) confirms; matters once per-step compute drops.
5. Re-measure tok/s and MFU; record in this file; update the sbatch defaults.

## Acceptance

- ≥2× tokens/s on the pilot recipe with bit-identical loss curve over 20 steps (same seed/data) vs baseline.
- No change to model math (two-stream semantics, masks, loss) — speed only.


## Round 1 results (2026-09-10, one node, 8×H100, `profile_1node.sbatch`, 2–3 profiled optimizer steps each)

Baseline canvas/bd 8: **31.25 s/step**, phases fwd 7.8 / bwd 22.2 / data 0.02 / optim 0.05. Self-CUDA shares:
two-stream GDN bwd 23%, NCCL all-gather 19% (waiting on stragglers, not bandwidth), matmuls 16%, GDN fwd 8%,
elementwise ~18%, flex-attn bwd 5%. CPU: 89% in `cudaStreamSynchronize`.

| variant (canvas bd 8 unless noted) | s/step | real tok/s (8 GPU) | note |
|---|---|---|---|
| baseline (buckets 2k..32k, pack 11) | 31.25 | ~13–14k | |
| `TRIDA_BLOCK_CONV_CHECK=0` (no host sync) | 31.2 | same | GPU-bound here; sync only mattered for bd 32 |
| `--fsdp_keep_params` | 31.1 | same | all-gather time is waiting, not bandwidth |
| `--compile_mlp` (bucketed shapes) | 34.3 | | recompiles per bucket → worse |
| all rows padded to 32k (BUCKETS=32768) | 30.8 | ~12–15k | NCCL all-gather share 18.9% → 13.0% (imbalance confirmed), padding eats it |
| + `PACK_EXAMPLES=20` (fuller rows) | 30.9 | **~15.5–16.5k** | rows 28–31k real tokens: **+10–15% throughput, free** |
| + `--compile_mlp` (static shapes, profiled after 6 warm-up steps) | **29.4** | **~17k** | −4.7% step time, peak mem 51.1 → 47.2 GB, loss identical to 4 decimals |
| + `--compile_glue` instead | 29.45 | ~17k | no gain over compile_mlp |
| GDN `TRIDA_GDR_CKPT_STRIDE=2` / `4` | 30.7 / 30.9 | | backward is not replay-bound |
| **random/bd 32 (production recipe)** check on → off, same node & data | **55.4 → 36.5** | | `.item()` in the conv cu_seqlens check blocked the CPU 55% of the time on the `chunk_refine` path |

Net for the pilot recipe: **31.25 → 29.4 s/step at +20% tokens/step ≈ 1.25× real-token throughput**.
Net for the production recipe: **≈1.5× from the sync removal alone**; packing + compile should stack (not yet measured together).

New defaults in `train/tools/draftalign/train_2node.sbatch`: `TRIDA_BLOCK_CONV_CHECK=0`, `BUCKETS=32768`,
`PACK_EXAMPLES=20`, `--compile_mlp` (`COMPILE_MLP=0` to disable). Caveat: `max_packed_rows=1` DROPS the
conversations that do not fit the first row — with pack 20 that is more data skipped per step (a data-efficiency
issue, not a speed one). Fix = carry leftovers to the next step in `PackedMultiTurnCollator` (round 2).

## Round 2 (not started): where the remaining 3× lives

MFU is still ~10%. GPU time is now: two-stream GDN kernels ~31% (the `fla_style_full` backward alone 23%),
matmuls 16%, elementwise 18%, NCCL wait 13%, flex-attn 5%. Options, by expected payoff:
1. Two-stream GDN backward kernel: it costs 2.9× its own forward; profile the Triton kernel (`_profiling.py` hooks),
   check tile sizes (`TRIDA_GDR_BWD_BV` 16/64), and the `chunk_wy_triton_improved` route vs `fla_style`.
2. Selective gradient checkpointing (skip recompute for MLP/attn, keep for GDN) — 47 GB peak leaves ~30 GB.
3. Length-balanced packing across ranks (all-reduce the bucket) instead of padding everything to 32k.
4. Elementwise fusion outside the MLP (norms, gating, l2norm in the GDN wrapper) via compile or Triton.
5. Collator carry-over (data efficiency).
