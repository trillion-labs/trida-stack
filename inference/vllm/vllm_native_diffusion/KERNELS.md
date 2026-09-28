# Native-diffusion kernels — provenance & recipe

`qwen3_5_diffusion.py` (this vLLM plugin's diffusion model) needs
`block_causal_readout.py`, the two-stream block-causal Gated-DeltaNet **readout**
kernel. That file is **not vendored here**: it is a near-verbatim port of the
sglang two-stream backend's
`eval/sglang/srt/layers/attention/block_gdn/fused_recurrent.py`, which derives
from flash-linear-attention and HybridDiffusion and is **PolyForm Noncommercial
1.0.0** — not covered by this repo's Apache-2.0 license.

## How to obtain it

1. Build the sglang two-stream backend via `inference/sglang/` (clone pinned
   upstream `yuchen-zhu-zyc/HybridDiffusion@6ca547a`, apply
   `two_stream_diffusion.patch`).
2. Copy `.../block_gdn/fused_recurrent.py` to
   `inference/vllm/vllm_native_diffusion/block_causal_readout.py`.
3. Repoint the two FLA helper imports to vLLM's bundled `vllm.model_executor`
   FLA so it runs in the vLLM venv (the only change from the sglang source).

Until this file is present the `--native-diffusion` vLLM path will not import;
the sglang backend (`inference/serve.py`) is the supported serving path.
