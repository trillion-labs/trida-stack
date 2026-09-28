# BFCL v4 eval for Trida-7B

Runs the [Berkeley Function-Calling Leaderboard v4](https://github.com/ShishirPatil/gorilla)
on `trillionlabs/Trida-7B-Preview`.

## Why this shape

BFCL drives local models through an **OpenAI-compatible `/v1/completions`** endpoint (its `OSSHandler`
normally spins up vLLM/sglang). This track predates the serving work in `inference/`: it was built when
stock vLLM/sglang could not load a `trust_remote_code` block-diffusion arch, and it serves the model
directly with `model.generate(...)` on **one GPU (~15.5 GB bf16)** instead. `inference/` now serves this
architecture on both backends (vLLM plugin, SGLang recipe) and is the path to use for anything new; the
shim below is kept because the published BFCL numbers were produced with it. So we:

1. Serve Trida with a tiny OpenAI shim (`serve_trida_openai.py`) that calls `model.generate(...)`.
2. Run **one replica per GPU** (data-parallel) behind a round-robin `router.py` (throughput).
3. Point BFCL at the router (`REMOTE_OPENAI_BASE_URL` + `--skip-server-setup`).
4. Use a **native-FC handler** (`trida_handler.py`, a subclass of BFCL's `QwenFCHandler`) — Trida's
   `<tool_call>{...}</tool_call>` format is identical to Qwen's.

## Files
- `serve_trida_openai.py` — single-GPU OpenAI `/v1/completions` + `/v1/models` server around Trida.
- `router.py` — round-robin router over the per-GPU replicas.
- `trida_handler.py` — `TridaFCHandler` (BFCL handler; symlinked into the BFCL package).
- `register_bfcl.sh` — installs BFCL + registers Trida (`model_config.py` + handler symlink). Idempotent.
- `run_eval.sh` — launches the pool, runs `bfcl generate` + `bfcl evaluate`, collects scores.
- `bfcl_eval.sbatch` — 1 node × 8 GPU on `a GPU node` (local only; not tracked).

## Run

> **Note:** `*.sbatch` launchers are cluster-specific and are **not tracked** in git (see `.gitignore`). Recover one with `git log --all --diff-filter=D -- <path>` then `git show <rev>:<path>`, or use the tracked `run_eval.sh` / `torchrun` path below.

```bash
# 0) one-time: install BFCL + register Trida into it
bash benchmark/bfcl_v4/register_bfcl.sh

# 1) full run on 8 GPUs, after the training job frees a GPU node:
sbatch --dependency=afterany:<train_jobid> benchmark/bfcl_v4/bfcl_eval.sbatch
#    (or interactively on a free GPU node:)  bash benchmark/bfcl_v4/run_eval.sh

# quick smoke (1 GPU, a few ids):
NUM_GPUS=1 CATEGORIES=simple_python bash benchmark/bfcl_v4/run_eval.sh
```

**Categories:** default `non_live,live,multi_turn,memory` = "all scoring except `web_search`"
(web_search needs a `SERPAPI_API_KEY`). Override with `CATEGORIES=...`.

**Outputs:** `benchmark/bfcl_v4/result/` (raw generations) and `benchmark/bfcl_v4/score/`
(`data_overall.csv` + per-category), via `BFCL_PROJECT_ROOT`.

## Environments (three isolated venvs; `register_bfcl.sh` builds the last two)
- `.venv` — the trainer (transformers 5.x). **Not used here.**
- `.venv-serve` — the Trida inference server. **transformers 4.57.1** + torch 2.11 (cu128): the
  released `trillionlabs/Trida-7B-Preview` remote code is written for tf 4.x and breaks on 5.x
  (`config.pad_token_id` was removed) — so the server is pinned to 4.57.1.
- `.venv-bfcl` — the `bfcl` CLI (BFCL's own deps).

## Notes
- Generation params match `benchmark/scripts/eval_trida.sh` (`threshold=0.9`, `block_size=32`,
  `small_block_size=8`, `top_p=0.95`, `mask_id=128012`, `stop=128001`). Tune via env in `run_eval.sh`.
- A diffusion model over the full scoring set is slow — expect a multi-hour run even at DP=8.
- `.env` supplies `HF_TOKEN` (model download) and optionally `WANDB_API_KEY`.
