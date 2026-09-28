"""Terminal-Bench (Hard) runner for Trida — SCAFFOLD / PLACEHOLDER.

Shows where an agentic terminal evaluation plugs into this repo. Does NOT implement the
agent command-loop or invoke the (Docker-sandboxed) harness.

Pipeline:
    terminal-bench-hard task set
        -> agent adapter (Trida issues shell commands, observes output, iterates)  [TODO]
        -> harness runs each task in Docker + checks success                       [TODO]
        -> report % resolved on the Hard subset

License note: harness is open-source (verify Apache-2.0); tasks execute untrusted software in
Docker. Run only in the provided sandbox. See README.md.
"""
from __future__ import annotations

import argparse

MODEL_NAME = "trida-7b-preview"


def build_agent(device: str = "mps"):
    """TODO: return an agent that wraps Trida generation as a terminal actor.

    The agent receives the task prompt + terminal state and must emit shell commands,
    observe stdout/stderr, and decide the next action until done. Drive generation via
    the served OpenAI-compatible endpoint for smoke tests.
    """
    raise NotImplementedError("Trida terminal agent adapter not implemented (scaffold).")


def run(task_set: str = "terminal-bench-hard", limit: int | None = None) -> None:
    """TODO: invoke the terminal-bench harness against `task_set` in its Docker sandbox.

    Roughly:
        from terminal_bench import Harness
        harness = Harness(task_set=task_set, agent=build_agent())
        results = harness.run(limit=limit)
        print(results.resolved_rate)
    """
    raise NotImplementedError("Wire up the terminal-bench harness in a Docker sandbox.")


def main() -> None:
    p = argparse.ArgumentParser(description="Terminal-Bench Hard runner for Trida (scaffold).")
    p.add_argument("--task-set", default="terminal-bench-hard")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()
    print("[scaffold] Terminal-Bench integration is not yet implemented. See README.md.")
    print(f"[scaffold] Would run task_set={args.task_set} (limit={args.limit}) in a Docker "
          f"sandbox with a Trida-driven terminal agent.")


if __name__ == "__main__":
    main()
