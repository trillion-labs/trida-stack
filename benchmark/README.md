# Benchmarking

Single-turn benchmarks run through **[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)**
(MIT). The two **agentic** tracks — **SWE-bench** and **Terminal-Bench (Hard)** — use their own
harnesses (scaffolds here).

## Layout

```
benchmark/
├── tasks/            # Trida lm-eval task YAMLs (gsm8k-trida)
├── serving/          # shared OpenAI-compatible Trida server + router + pool.sh (used by the tool-use benches)
├── bfcl_v4/          # Berkeley Function-Calling Leaderboard v4 (tool use)
├── functionchat/     # FunctionChat-Bench (Korean tool use; LLM-judge)
├── ko_agentbench/    # Ko-AgentBench (Korean agent, L1–L7; LLM-judge)
├── tau2/             # τ²-bench (multi-turn tool use w/ simulated user; pass^k)
├── swe_bench/        # agentic software-engineering benchmark (scaffold)
└── terminal_bench/   # agentic terminal benchmark — Hard subset (scaffold)
```

### Default model

The tool-use benches default to the public
**[trillionlabs/Trida-7B-Preview](https://huggingface.co/trillionlabs/Trida-7B-Preview)** checkpoint
(loaded with `trust_remote_code`, transformers 4.x). Point them at another checkpoint with
`MODEL=<hf-id-or-local-path>`.

### Tool-use / agent benchmarks (OpenAI-served)

These tool-use benches run against an OpenAI-compatible endpoint. The checkpoint can be served
through the SGLang or vLLM backends in `inference/` (see the root README), but the benches here use
the self-contained **shared serving stack** in `serving/` — N per-GPU replicas of
`serve_trida_openai.py` (`.venv-serve`, transformers 4.57.1; a plain `trust_remote_code` HF server
for the released remote-code checkpoint) behind `router.py` (round-robin + failover), launched by
`pool.sh`. The server exposes `/v1/completions` (BFCL) and `/v1/chat/completions` with server-side
`<tool_call>`→OpenAI-`tool_calls` parsing (FunctionChat / Ko-AgentBench / tau2). Each benchmark dir
has a `register_*.sh` (build venvs), `run_eval.sh`, and a local (untracked, see `.gitignore`) `.sbatch`; see each dir's README. tau2 also
uses an OpenRouter **user simulator**; its `--max-concurrency` maps parallel conversations onto the
replica pool.

## Benchmark catalog

| Benchmark | Type | Measures | Availability | License |
|-----------|------|----------|--------------|---------|
| **GSM8K** | Reasoning (math, 4-shot) | accuracy + tokens/s | Runnable | MIT |
| **HumanEval** | Code gen (0-shot) | pass@k | Runnable † | MIT |
| **MBPP / sanitized** | Code gen | Python problems | Runnable † | CC-BY-4.0 |
| **MMLU** | Knowledge | 57-subject accuracy | Runnable † | MIT |
| **GPQA** | Knowledge | graduate Q&A | Runnable † | CC-BY-4.0 |
| **MATH / Minerva** | Reasoning | competition math | Runnable † | MIT |
| **IFEval** | Instruction following | verifiable adherence | Runnable † | Apache-2.0 |
| **BFCL v4** | Tool use (function calling) | per-category accuracy | Runnable (`bfcl_v4/`) | Apache-2.0 |
| **FunctionChat-Bench** | Korean tool use (LLM-judge) | per-category pass rate | Runnable (`functionchat/`) | Apache-2.0 |
| **Ko-AgentBench** | Korean agent (L1–L7, LLM-judge) | per-dimension metrics | Runnable (`ko_agentbench/`) | Apache-2.0 |
| **τ²-bench (tau2)** | Multi-turn tool use w/ simulated user | reward / pass^k | Runnable (`tau2/`) | MIT |
| **SWE-bench (Verified)** | **Agentic software** | resolve real GitHub issues | Scaffold (`swe_bench/`) | **MIT** |
| **Terminal-Bench (Hard)** | **Agentic terminal** | hard terminal tasks in Docker | Scaffold (`terminal_bench/`) | **Apache-2.0** (runs untrusted code, Docker) |

† Available through lm-evaluation-harness; only GSM8K has a Trida task + driver scripted in this
repo (`tasks/gsm8k-trida.yaml`). **Scaffold** = integration stub, not yet runnable.

## Running the lm-eval benchmarks

The Slurm/driver scripts and the vendored dInfer harness that previously wrapped these runs are not
part of this branch. The repo still ships the Trida lm-eval task at `tasks/gsm8k-trida.yaml`; run it
with a separately-installed
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) pointed at a served
Trida endpoint (see `inference/` for serving). Only GSM8K has a Trida task scripted here; the other
single-turn benchmarks use lm-evaluation-harness's own task definitions.

## Agentic tracks

- **SWE-bench** — [`swe_bench/README.md`](swe_bench/README.md). Resolve GitHub issues (patch +
  tests). Scaffold: documents integration + licensing, stubs the runner.
- **Terminal-Bench (Hard)** — [`terminal_bench/README.md`](terminal_bench/README.md). Agent
  operating a real terminal in Docker, on the hard subset. Scaffold.

Both need an agent loop wrapped around a served Trida endpoint (`serving/`) and run
their tasks in **sandboxed Docker** (SWE-bench MIT, Terminal-Bench Apache-2.0; the operational
caveat is untrusted-code execution, not licensing).
