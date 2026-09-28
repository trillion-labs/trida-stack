# block_gated_delta_rule — two-stream Gated-DeltaNet kernels (fetch recipe)

The two-stream Gated-DeltaNet + ShortConv Triton kernels used by
`train/hf_block_diffusion_hybrid.py` (`forward_flare`) are **not vendored in this
repo**. They are PolyForm-Noncommercial 1.0.0 (upstream) — see
`LICENSE.HybridDiffusion` — so this Apache-2.0 repo ships a *recipe* to assemble
them from their public upstream, exactly like `inference/sglang/`.

## What you get

`fetch_kernels.sh` clones the pinned upstream, copies the kernel package into this
directory, and applies `trillion_mods.patch` (our block-causal / two-stream
readout modifications to three files: `chunk_fla_style_wy.py`, `convolution.py`,
`fused_recurrent_state.py`).

## Recipe

```bash
bash train/block_gated_delta_rule/fetch_kernels.sh
```

Equivalent by hand:

```bash
git clone https://github.com/yuchen-zhu-zyc/HybridDiffusion
cd HybridDiffusion && git checkout 6ca547a   # "Initial public release"
cp torchtitan/models/qwen3_5/model/block_gated_delta_rule/*.py \
   <repo>/train/block_gated_delta_rule/
cd <repo>/train/block_gated_delta_rule && patch -p1 < trillion_mods.patch
```

## License

The fetched kernels and `trillion_mods.patch` derive from HybridDiffusion and are
**PolyForm Noncommercial 1.0.0** (`LICENSE.HybridDiffusion`, `NOTICE.HybridDiffusion`).
They are noncommercial and are not covered by this repo's Apache-2.0 license.
