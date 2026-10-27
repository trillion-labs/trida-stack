#!/usr/bin/env python3
"""Fail when a GitHub Action is referenced by a tag instead of a commit SHA.

This repository's supply-chain story is that third-party code is pinned to a commit
and CI fails when the pin drifts (tools/check_supply_chain.py, tools/verify_fetch.sh).
The workflows themselves were the exception: they ran actions from moving tags.

A tag is a pointer, not a version. Upstream can move it, and a compromised account
moves it silently -- which is how tj-actions/changed-files was used to exfiltrate
secrets from thousands of repositories in March 2025. The actions that build our SBOM
and sign our release attestation are exactly the ones a reader is trusting, so they
are the last place a moving reference belongs.

Local actions (./path) and reusable workflows in this repository are not pinned: they
are this repository's own code and move with it by design.

Usage:  python tools/check_action_pins.py
"""

from __future__ import annotations

import pathlib
import re
import sys

WORKFLOWS = pathlib.Path(".github/workflows")
# `uses: owner/repo@ref` or `uses: owner/repo/path@ref`, with an optional trailing comment.
USES = re.compile(r"^\s*-?\s*uses:\s*(?P<ref>[^\s#]+)")
SHA = re.compile(r"^[0-9a-f]{40}$")


def main() -> int:
    if not WORKFLOWS.is_dir():
        print(f"{WORKFLOWS}: not found", file=sys.stderr)
        return 1

    problems: list[str] = []
    checked = 0

    for path in sorted(WORKFLOWS.glob("*.yml")) + sorted(WORKFLOWS.glob("*.yaml")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            m = USES.match(line)
            if not m:
                continue
            ref = m.group("ref")
            # This repository's own code: a local action or a reusable workflow.
            if ref.startswith("./") or ref.startswith(".github/"):
                continue
            checked += 1
            if "@" not in ref:
                problems.append(f"{path.name}:{lineno}: {ref} has no ref at all")
                continue
            action, _, version = ref.rpartition("@")
            if not SHA.match(version):
                problems.append(
                    f"{path.name}:{lineno}: {action} is pinned to '{version}', not a commit SHA"
                )

    if problems:
        print("Actions must be pinned to a full 40-character commit SHA.\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nResolve the tag and keep the human-readable version in a comment:\n"
            "  gh api repos/<owner>/<repo>/git/ref/tags/<tag> --jq .object.sha\n"
            "  - uses: owner/repo@<sha>  # v1.2.3"
        )
        return 1

    print(f"OK: {checked} action reference(s), all pinned to a commit SHA.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
