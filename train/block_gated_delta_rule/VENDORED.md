# Provenance: HybridDiffusion / FLARE (arXiv 2606.01774)

> These kernels are **no longer vendored** in this repo — they are fetched by
> `fetch_kernels.sh` (see `README.md` in this directory). This file records the
> provenance and license of what that recipe assembles.

Source: https://github.com/yuchen-zhu-zyc/HybridDiffusion @ commit `6ca547a`, subpath `torchtitan/models/qwen3_5/model/block_gated_delta_rule/`.
Two-stream Gated-DeltaNet + ShortConv Triton kernels for block-diffusion training.
License: PolyForm Noncommercial 1.0.0 (see LICENSE.HybridDiffusion). Research/internal use only.
Standalone deps: torch, triton, einops, fla (flash-linear-attention), causal_conv1d.
