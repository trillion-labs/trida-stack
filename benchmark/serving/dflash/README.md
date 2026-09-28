# DFlash vs self-spec vs AR — cross-engine throughput comparison

External baseline for the trida self-spec (AR-Trust) work: the public DFlash drafter for the stock
Qwen3.5-4B target (`z-lab/Qwen3.5-4B-DFlash`, 6-layer block-diffusion drafter, Apache-2.0) measured on
the same GPUs and protocol as our self-spec sweeps, on both engines.

- `serve_vllm.sh` — vLLM 0.27.x: `ar` (stock), `dflash` (built-in `method: dflash`, `num_speculative_tokens` = block),
  `selfspec` (our plugin, `TRIDA_SELFSPEC_N` = block)
- `serve_sglang.sh` — recent upstream SGLang: `ar`, `dflash` (`--speculative-algorithm DFLASH`)
- `bench_job.sh` — one-node Slurm job: one server per GPU, then `sweep_client.py` (64 GSM8K prompts, fixed 512 output
  tokens, greedy, C = 1/4/8/16). `SMOKE=1` = 10 prompts, C=1, prints two outputs.
- The sweep client is `inference/vllm/tools/comprehensive_run/sweep_client.py` (shared with the self-spec sweeps).

Targets differ (stock Qwen3.5-4B for DFlash, trida for self-spec), so compare each method's speedup over its own AR
on the same engine, plus absolute tok/s side by side. Block ladder: 4 / 8 / 16 (DFlash) and N = 4 / 8 / 16 / 32 (self-spec).
Box paths: models `luke/models/dflash/`, envs `luke/env/{vllm-uv27,sglang-dflash}`, results `luke/runs/2026-09-11_dflash/`.
