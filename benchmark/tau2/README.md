# τ²-bench (tau2) — Trida diffusion-LM eval

[sierra-research/tau2-bench](https://github.com/sierra-research/tau2-bench) — multi-turn tool-use
agent benchmark with a **simulated user**. An orchestrator runs a turn-based conversation between an
LLM **agent** (customer-service rep following a domain policy, calling tools) and an LLM
**user simulator**, against a stateful **environment** (DB + tools). Domains: `airline` (50),
`retail` (117), `telecom` (thousands), `banking_knowledge` (RAG), `mock` (dev). Metric: per-task
`reward ∈ {0,1}` → aggregate `pass^k` / average reward.

## Why this shape

- **Agent = Trida** over the **shared serving stack** (`benchmark/serving/`): N per-GPU replicas of
  `serve_trida_openai.py` behind `router.py`. tau2 talks OpenAI `/v1/chat/completions` with `tools`
  via LiteLLM's `openai/` provider; the server builds ChatML server-side and returns OpenAI
  `tool_calls`. tau2 runs `--max-concurrency` conversations in parallel, so the data-parallel pool
  is used (`CONCURRENCY` defaults to `NUM_GPUS`).
- **User simulator = OpenRouter** (`--user-llm openrouter/…`, `OPENROUTER_API_KEY`) — kept separate
  from the agent so only the agent hits the local server (agent routing lives in `--agent-llm-args`
  `api_base`, not global env).
- tau2-bench is a stock upstream checkout (no patching); only the run harness lives here.

## Venvs / keys

| venv | role |
|------|------|
| `.venv-serve` (repo root) | Trida inference server — transformers 4.57.1 |
| `<TAU2_DIR>/.venv` (uv) | tau2-bench — python 3.12, LiteLLM |

- `OPENROUTER_API_KEY` (repo `.env`) — required, for the user simulator.
- `OPENAI_API_KEY` — only needed for tasks whose `reward_basis` includes NL assertions (tau2's judge
  is hardcoded to `gpt-4.1`); airline/retail/telecom are mostly DB/action-graded.

## Run

> **Note:** `*.sbatch` launchers are cluster-specific and are **not tracked** in git (see `.gitignore`). Recover one with `git log --all --diff-filter=D -- <path>` then `git show <rev>:<path>`, or use the tracked `run_eval.sh` / `torchrun` path below.

```bash
# once: clone + sync tau2-bench (into TAU2_DIR, default /path/to/tau2-bench) + serve venv
bash benchmark/tau2/register_tau2.sh

# full eval (1 node x 8 GPU)
sbatch benchmark/tau2/tau2_eval.sbatch

# smoke: mock domain, 2 tasks
DOMAINS="mock" NUM_TASKS=2 NUM_GPUS=1 bash benchmark/tau2/run_eval.sh
```

Env knobs: `NUM_GPUS`, `MODEL` (default `trillionlabs/Trida-7B-Preview`), `DOMAINS`
(`airline retail telecom`), `NUM_TRIALS` (1), `CONCURRENCY` (=`NUM_GPUS`), `NUM_TASKS`/`TASK_IDS`
(cap scope — telecom has thousands of tasks), `USER_LLM` (`openrouter/openai/gpt-4o-mini`), `SERVED`,
`PORT`, `BLOCK_SIZE`, `THRESHOLD`, `TAU2_DIR`.

## Output

Per domain, `benchmark/tau2/results/trida_<domain>/results.json` (gitignored) — full simulations
with `reward_info` per task/trial. The run prints `sims / avg_reward / pass@1` per domain; use
`uv run tau2 view` in `TAU2_DIR` for the interactive report.
