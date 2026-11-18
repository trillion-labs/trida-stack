#!/usr/bin/env python3
"""Guard the supply-chain records against drift.

The two-stream kernels are PolyForm-Noncommercial and are fetched at build time, so no
dependency scanner can see them. We therefore describe them by hand in
`docs/oss/sbom/external-components.json`, which is merged into every generated SBOM.

A hand-written SBOM fragment is only trustworthy if something stops it from going stale.
That is this script. It fails if the pinned upstream commit recorded anywhere in the
repository disagrees with the one the fetch script actually uses, or if a fetched
noncommercial kernel file is tracked by git (a `.gitignore` entry only hides it; a
moved directory or a `git add -f` gets past that silently).

Run: python tools/check_supply_chain.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FETCH_SCRIPT = ROOT / "train" / "block_gated_delta_rule" / "fetch_kernels.sh"

# Files that quote the pinned commit. If you add another, add it here too.
QUOTING_FILES = [
    "README.md",
    "COMPLIANCE.md",
    "train/README.md",
    "train/block_gated_delta_rule/README.md",
    "train/block_gated_delta_rule/VENDORED.md",
    "inference/zoro-1/sglang/README.md",
    "inference/zoro-1/sglang/DECISION_PROBE.md",
    "inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md",
    "docs/supply-chain/asset-register.csv",
    "docs/supply-chain/README.md",
    "tools/kernel-checksums.txt",
]

# A HybridDiffusion short commit: 7+ hex chars. Deliberately narrow so we do not flag
# unrelated hashes; we only look at lines that also mention the upstream or the pin.
COMMIT_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
CONTEXT_RE = re.compile(r"HybridDiffusion|HYBRIDDIFFUSION_COMMIT|git checkout", re.I)


# Where the fetch recipes put PolyForm-NC code. Matched by name, not by full path, so a
# copy left behind by a directory move is caught too.
NONCOMMERCIAL_FILENAMES = {"block_causal_readout.py"}           # KERNELS.md (vLLM readout)
NONCOMMERCIAL_PY_DIRS = ("train/block_gated_delta_rule/",)       # fetch_kernels.sh copies *.py here


def tracked_noncommercial(paths: list[str]) -> list[str]:
    """The paths in `paths` that are fetched noncommercial kernel code and must never be committed."""
    return [p for p in paths
            if p.rsplit("/", 1)[-1] in NONCOMMERCIAL_FILENAMES
            or (p.startswith(NONCOMMERCIAL_PY_DIRS) and p.endswith(".py"))]


def git_tracked() -> list[str]:
    return subprocess.run(["git", "ls-files"], cwd=ROOT, check=True, capture_output=True,
                          text=True).stdout.splitlines()


def authoritative_commit() -> str:
    """The commit fetch_kernels.sh actually checks out. This is the source of truth."""
    text = FETCH_SCRIPT.read_text()
    m = re.search(r'HYBRIDDIFFUSION_COMMIT:-([0-9a-f]{7,40})', text)
    if not m:
        sys.exit(f"FAIL: could not read the pinned commit from {FETCH_SCRIPT}")
    return m.group(1)


def main() -> int:
    pinned = authoritative_commit()
    problems: list[str] = []

    committed = tracked_noncommercial(git_tracked())
    if committed:
        print("PolyForm-Noncommercial kernel code is tracked by git; it must stay a fetched recipe:\n")
        for p in committed:
            print(f"  {p}")
        print("\nRemove it from the index (git rm --cached <path>); the fetch recipe recreates it locally.")
        return 1

    for rel in QUOTING_FILES:
        path = ROOT / rel
        if not path.exists():
            problems.append(f"{rel}: listed as quoting the pin, but the file is missing")
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if not CONTEXT_RE.search(line):
                continue
            for found in COMMIT_RE.findall(line):
                # Short-vs-long forms are fine as long as one prefixes the other.
                if not (found.startswith(pinned) or pinned.startswith(found)):
                    problems.append(
                        f"{rel}:{lineno}: quotes commit {found!r}, "
                        f"but fetch_kernels.sh pins {pinned!r}"
                    )

    if problems:
        print(f"Supply-chain records disagree with the pinned commit ({pinned}):\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nThe fetch script is the source of truth. If you bumped the pin, update the "
            "files above AND re-record tools/kernel-checksums.txt (run tools/verify_fetch.sh "
            "to see the new hashes) -- otherwise the integrity check silently passes on "
            "the old contents."
        )
        return 1

    print(f"OK: pinned commit {pinned} is consistent across {len(QUOTING_FILES)} files "
          f"and the recorded checksums.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
