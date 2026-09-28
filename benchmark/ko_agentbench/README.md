# Ko-AgentBench (Trida diffusion-LM eval)

HF KREW's [Ko-AgentBench](https://huggingface.co/datasets/huggingface-KREW/Ko-AgentBench) — Korean
tool-calling agent benchmark across 7 capability dimensions (L1–L7: single call, selection,
sequential, parallel, error-handling, efficiency, long-context memory). Wired to serve the Trida
block-diffusion model over an OpenAI-compatible endpoint, mirroring `benchmark/bfcl_v4/`.

## Why this shape

Trida is served by the **shared stack** in `benchmark/serving/` (per-GPU `serve_trida_openai.py` +
`router.py`). Ko-AgentBench calls the model through **LiteLLM's OpenAI provider**: set
`OPENAI_API_BASE` to the router and use `--model openai/<name>` (zero code change). The server
handles `/v1/chat/completions` with `tools` and parses `<tool_call>` into OpenAI `tool_calls`.
Task tools are **cache-backed** (`--cache-mode read`) so no external API keys are needed.

Two stages: **run** trajectories (`run_benchmark_with_logging.py`) then **evaluate**
(`evaluate_model_run.py`). Ko-AgentBench is **sequential** (one model call at a time) → one replica
suffices (`NUM_GPUS=1`).

## Venvs / judge

| venv | role |
|------|------|
| `.venv-serve` (repo root) | Trida inference server — transformers 4.57.1 |
| `<KOAB_DIR>/.venv` (uv) | Ko-AgentBench — python 3.10, litellm |

Judge defaults to `openrouter/openai/gpt-4.1-mini` — the `openrouter/` prefix routes scoring to
OpenRouter (via `KOAB_DIR/.env`'s `OPENROUTER_API_KEY`), so it does **not** hit the local Trida
endpoint. `OPENAI_API_BASE` is scoped to the run stage only.

## Run

> **Note:** `*.sbatch` launchers are cluster-specific and are **not tracked** in git (see `.gitignore`). Recover one with `git log --all --diff-filter=D -- <path>` then `git show <rev>:<path>`, or use the tracked `run_eval.sh` / `torchrun` path below.

```bash
# once
bash benchmark/ko_agentbench/register_ko_agentbench.sh

# full eval (1 GPU)
sbatch benchmark/ko_agentbench/koab_eval.sbatch
#   or interactively on a GPU node:
NUM_GPUS=1 bash benchmark/ko_agentbench/run_eval.sh

# smoke: just L1
LEVELS="L1" bash benchmark/ko_agentbench/run_eval.sh
```

Env knobs: `NUM_GPUS`, `MODEL` (default `trillionlabs/Trida-7B-Preview`), `LEVELS`, `SERVED`
(LiteLLM model label, no `gpt-5` substring), `JUDGE`, `PORT`, `BLOCK_SIZE`, `THRESHOLD`, `KOAB_DIR`.

## Output

The evaluation report is collected into `benchmark/ko_agentbench/reports/openai_<served>_<date>/`
(gitignored): `evaluation_report.json` (summary + per-level metrics), `.csv`, `.md`.

## Note

Sequential throughput (one replica). Running levels in parallel processes to use more GPUs is a
possible future optimization but complicates the per-run timestamp dir that `--date` resolves.
