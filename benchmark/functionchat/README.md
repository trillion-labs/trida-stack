# FunctionChat-Bench (Trida diffusion-LM eval)

Kakao's [FunctionChat-Bench](https://github.com/kakao/FunctionChat-Bench) — Korean tool-use /
function-calling, scored by an LLM-as-judge. Wired to serve the Trida block-diffusion model over an
OpenAI-compatible HTTP endpoint, mirroring `benchmark/bfcl_v4/`.

## Why this shape

Trida is a `trust_remote_code` block-diffusion arch that vLLM/sglang can't serve, but it fits on one
GPU. We serve it with the **shared stack** in `benchmark/serving/`: N per-GPU replicas of
`serve_trida_openai.py` behind `router.py` (round-robin + failover). FunctionChat talks OpenAI
**`/v1/chat/completions`** with `tools`; Trida ships no chat template, so the server builds a
Qwen-style ChatML prompt and parses `<tool_call>` blocks into OpenAI `tool_calls` server-side.

FunctionChat needs no package patching (unlike BFCL): it takes the target model purely via CLI
(`--model inhouse --base_url … --served_model_name …`).

## Venvs

| venv | role |
|------|------|
| `.venv-serve` (repo root) | Trida inference server — transformers 4.57.1 (released remote code needs 4.x) |
| `<FCBENCH_DIR>/.venv` | FunctionChat's own deps (openai, click, mistralai, vertexai, qwen_agent) |

## Run

> **Note:** `*.sbatch` launchers are cluster-specific and are **not tracked** in git (see `.gitignore`). Recover one with `git log --all --diff-filter=D -- <path>` then `git show <rev>:<path>`, or use the tracked `run_eval.sh` / `torchrun` path below.

```bash
# once: build venvs (idempotent)
bash benchmark/functionchat/register_functionchat.sh

# full eval (1 node x 8 GPU)
sbatch benchmark/functionchat/fcbench_eval.sbatch
#   or interactively on a GPU node:
NUM_GPUS=8 bash benchmark/functionchat/run_eval.sh

# cheap *scored* smoke: one subset, first 2 raw records (FC_LIMIT truncates the input)
SUBSETS="dialog" FC_LIMIT=2 NUM_GPUS=1 bash benchmark/functionchat/run_eval.sh
```

Env knobs: `NUM_GPUS`, `MODEL` (default `trillionlabs/Trida-7B-Preview`), `SUBSETS`
(`dialog singlecall common`), `FC_LIMIT` (truncate raw input to N records for a quick scored run),
`PORT`, `BLOCK_SIZE`, `THRESHOLD`, `FCBENCH_DIR`. Note: FunctionChat's own `--sample` flag is an
unimplemented stub that skips scoring, so this harness uses `FC_LIMIT` instead.

## Judge

LLM-as-judge is FunctionChat's own `config/openai.cfg` (OpenRouter `gpt-4.1`) — reused as-is; ensure
its `api_key` is valid. `run_eval.sh` passes `--is_batch False` so the judge's `base_url` is honored
(batch mode would hit `api.openai.com` and ignore OpenRouter).

## Output

Scores are collected into `benchmark/functionchat/output/<model>/` (gitignored), keyed
`FunctionChat-<model>.eval_score.json` (per-category pass rates for singlecall / dialog /
calldecision). Replica/router logs: `replica_*.log`, `router.log` here.
