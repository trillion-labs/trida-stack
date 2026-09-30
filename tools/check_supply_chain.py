#!/usr/bin/env python3
"""Guard the supply-chain records against drift.

The two-stream kernels are PolyForm-Noncommercial and are fetched at build time, so no
dependency scanner can see them. We therefore describe them by hand in
`docs/oss/sbom/external-components.json`, which is merged into every generated SBOM.

A hand-written SBOM fragment is only trustworthy if something stops it from going stale.
That is this script. It fails if the pinned upstream commit recorded anywhere in the
repository disagrees with the one the fetch script actually uses.

Run: python tools/check_supply_chain.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FETCH_SCRIPT = ROOT / "train" / "block_gated_delta_rule" / "fetch_kernels.sh"
SBOM_FRAGMENT = ROOT / "docs" / "oss" / "sbom" / "external-components.json"

# Files that quote the pinned commit. If you add another, add it here too.
QUOTING_FILES = [
    "README.md",
    "COMPLIANCE.md",
    "train/README.md",
    "train/block_gated_delta_rule/README.md",
    "train/block_gated_delta_rule/VENDORED.md",
    "inference/sglang/README.md",
    "inference/vllm/vllm_native_diffusion/KERNELS.md",
    "docs/oss/LICENSE_POLICY.md",
    "docs/oss/ASSET_INVENTORY.csv",
    "docs/oss/intake/2026-09-30-hybriddiffusion.md",
    "docs/oss/intake/2026-09-30-sglang.md",
]

# A HybridDiffusion short commit: 7+ hex chars. Deliberately narrow so we do not flag
# unrelated hashes; we only look at lines that also mention the upstream or the pin.
COMMIT_RE = re.compile(r"\b([0-9a-f]{7,40})\b")
CONTEXT_RE = re.compile(r"HybridDiffusion|HYBRIDDIFFUSION_COMMIT|git checkout", re.I)


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

    # The SBOM fragment must describe the same commit as a component version.
    fragment = json.loads(SBOM_FRAGMENT.read_text())
    for comp in fragment["components"]:
        if "HybridDiffusion" not in comp["name"]:
            continue
        version = comp["version"]
        if not (version.startswith(pinned) or pinned.startswith(version)):
            problems.append(
                f"docs/oss/sbom/external-components.json: component {comp['name']!r} "
                f"is version {version!r}, but fetch_kernels.sh pins {pinned!r}"
            )

    if problems:
        print(f"Supply-chain records disagree with the pinned commit ({pinned}):\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nThe fetch script is the source of truth. If you bumped the pin, update the "
            "files above -- including the SBOM fragment, or the generated SBOM will lie "
            "about which noncommercial code is in the build."
        )
        return 1

    print(f"OK: pinned commit {pinned} is consistent across {len(QUOTING_FILES)} files "
          f"and the SBOM fragment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
