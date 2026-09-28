"""SWE-bench runner for the Trida diffusion model — SCAFFOLD / PLACEHOLDER.

This is an intentionally incomplete integration point. It shows *where* an agentic
SWE-bench evaluation plugs into this repo and what it must produce, without yet
implementing the agent loop or invoking the (heavy, sandboxed) evaluation harness.

Pipeline:
    dataset (SWE-bench Verified)
        -> agent loop (drives Trida via serving/ to propose a patch)   [TODO]
        -> predictions.jsonl  {instance_id, model_name_or_path, model_patch}
        -> swebench.harness evaluation (sandboxed Docker)              [TODO]
        -> report % resolved

License note: the `swebench` harness is MIT; task instances derive from third-party
repos under mixed licenses. Run evaluation only in an isolated/sandboxed container.
See README.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


MODEL_NAME = "trida-7b-preview"


def load_dataset(name: str = "princeton-nlp/SWE-bench_Verified", split: str = "test"):
    """TODO: load via `datasets.load_dataset(name, split=split)`.

    Returns an iterable of instances with at least: instance_id, repo, base_commit,
    problem_statement, test_patch.
    """
    raise NotImplementedError(
        "Install `datasets` + `swebench`, then load the SWE-bench dataset here. "
        "Gated by the compliance decision on task-instance licensing (see README)."
    )


def solve_instance(instance: dict) -> str:
    """TODO: the agentic core.

    Wrap the diffusion model (see ../../serving/) in an agent controller that:
      1. checks out `repo` at `base_commit`,
      2. reads `problem_statement`, locates relevant files,
      3. generates a candidate patch (unified diff) via Trida,
      4. applies it, runs the repo's tests, and iterates.
    Must return a unified-diff string (the `model_patch`).
    """
    raise NotImplementedError("Agentic patch-generation loop not implemented (scaffold).")


def build_predictions(dataset, out_path: Path) -> Path:
    """Write predictions.jsonl in the format the SWE-bench harness expects."""
    with out_path.open("w") as f:
        for inst in dataset:
            patch = solve_instance(inst)
            f.write(json.dumps({
                "instance_id": inst["instance_id"],
                "model_name_or_path": MODEL_NAME,
                "model_patch": patch,
            }) + "\n")
    return out_path


def evaluate(predictions_path: Path, dataset_name: str) -> None:
    """TODO: invoke the official harness inside a sandbox, e.g.:

        python -m swebench.harness.run_evaluation \\
            --dataset_name {dataset_name} \\
            --predictions_path {predictions_path} \\
            --max_workers 4 --run_id trida-swebench
    """
    raise NotImplementedError("Wire up swebench.harness.run_evaluation in a sandboxed container.")


def main() -> None:
    p = argparse.ArgumentParser(description="SWE-bench runner for Trida (scaffold).")
    p.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    p.add_argument("--split", default="test")
    p.add_argument("--limit", type=int, default=None, help="cap instances for a smoke test")
    p.add_argument("--out", type=Path, default=Path("predictions.jsonl"))
    args = p.parse_args()

    print("[scaffold] SWE-bench integration is not yet implemented. See README.md.")
    print(f"[scaffold] Would: load {args.dataset}:{args.split}, run agent loop, "
          f"write {args.out}, then evaluate with the swebench harness.")
    # dataset = load_dataset(args.dataset, args.split)
    # if args.limit: dataset = list(dataset)[: args.limit]
    # preds = build_predictions(dataset, args.out)
    # evaluate(preds, args.dataset)


if __name__ == "__main__":
    main()
