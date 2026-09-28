# Terminal-Bench (Hard) — Agentic Terminal Benchmark (Scaffold)

**Status: placeholder / scaffold.** Defines the integration shape + licensing for evaluating
Trida as an agent operating in a real terminal. `run_terminal_bench.py` is a stub.

## What it measures

[Terminal-Bench](https://www.tbench.ai/) ([github](https://github.com/laude-institute/terminal-bench))
evaluates an AI agent's ability to **accomplish real tasks in a sandboxed terminal/Docker
environment** — installing software, manipulating files, running builds, debugging, etc. Each
task ships with a container setup and a programmatic success check.

- **Terminal-Bench Hard** is the difficult subset (the tasks current frontier agents fail most
  often) — the target track requested here.

Like SWE-bench, this requires an **agent loop**: the model issues shell commands, observes
output, and iterates until the task's verifier passes. Trida must be wrapped in an agentic
controller (a "terminus"-style harness) driving generation from `trida/inference`.

## Licensing

- The **Terminal-Bench harness** is **Apache-2.0**.
- **Tasks execute arbitrary commands and pull third-party software inside Docker** — an
  untrusted-execution concern (not a licensing one). Run only in the provided sandbox.
- Task definitions themselves are authored for the benchmark, but they install external
  packages whose licenses vary.

## Integration shape

```
terminal_bench/
├── README.md
├── requirements.txt        # terminal-bench harness deps (not installed here)
└── run_terminal_bench.py   # STUB: registers Trida as an agent; TODOs for the command loop
```

## How to make it real (checklist)

1. `pip install terminal-bench` and ensure Docker is available.
2. Implement an agent adapter that maps Trida generations → shell actions (observe → act loop).
3. Run the harness against the **terminal-bench-hard** task set in its sandbox.
4. Report **% resolved** on the Hard subset next to SWE-bench in `../README.md`.

See `run_terminal_bench.py` for the integration points.
