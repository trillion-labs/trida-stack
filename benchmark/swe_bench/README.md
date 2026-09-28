# SWE-bench — Agentic Software Benchmark (Scaffold)

**Status: placeholder / scaffold.** This directory defines the *shape* of a SWE-bench
integration for the Trida diffusion model and documents its licensing. It does **not** yet run
a full evaluation — `run_swebench.py` is a stub with the integration points marked `TODO`.

## What SWE-bench measures

[SWE-bench](https://github.com/princeton-nlp/SWE-bench) evaluates a model as an **agentic
software engineer**: given a real GitHub issue + repository snapshot, the model must produce a
code patch that makes the project's failing tests pass. Variants:

- **SWE-bench Verified** (recommended) — 500 human-validated, solvable instances.
- **SWE-bench Lite** — 300 lightweight instances for quick iteration.
- **SWE-bench** (full) — ~2,294 instances.

Unlike GSM8K/HumanEval (single-turn), SWE-bench requires an **agent loop**: read repo → locate
code → propose patch → run tests → iterate. A diffusion LLM therefore needs an agentic scaffold
(e.g. an SWE-agent-style controller) wrapped around `trida/inference` generation — that scaffold is the
main `TODO` below.

## Licensing & runtime

- **SWE-bench (harness + `SWE-bench_Verified` dataset): MIT.**
- Operational caveat (not a licensing one): task instances are built from real third-party GitHub
  repositories, so running them pulls and executes third-party code. Run from the official dataset
  in the provided **Docker sandbox**; if you ever redistribute derived artifacts, check the
  originating repos' terms.
- Evaluation executes untrusted repo code + tests — **run in an isolated/sandboxed container**.

## Integration shape

```
swe_bench/
├── README.md           # this file
├── requirements.txt    # swebench harness deps (not installed here)
└── run_swebench.py     # STUB: predictions -> swebench harness; TODOs for the agent loop
```

## How to make it real (checklist)

1. `pip install swebench` (MIT) and pull `SWE-bench_Verified` from Hugging Face.
2. Build an agentic controller that drives Trida via `trida/inference` (propose → apply patch → test).
3. Emit predictions in SWE-bench format: `{instance_id, model_name_or_path, model_patch}`.
4. Run the official evaluation harness inside a sandboxed Docker environment.
5. Report `% resolved` on SWE-bench Verified next to the lm-eval metrics in `../README.md`.

See `run_swebench.py` for the exact integration points.
