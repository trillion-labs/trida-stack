#!/usr/bin/env python3
"""Verify every handle in CODEOWNERS still resolves.

CODEOWNERS ignores unknown handles silently -- no error, no warning, no annotation.
A rule keeps looking correct while enforcing fewer reviewers than it names.

This is not hypothetical here. Two owners renamed within one week
(usik-luke -> usik-luke-trl, hongjoonahn-tri -> hongjoonahn-trl) and GitHub does not
redirect the old handle: the API returns 404. Both breakages were found by accident.

Needs network and a token with read access. Run: python tools/check_codeowners.py
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
CANDIDATES = [".github/CODEOWNERS", "CODEOWNERS", "docs/CODEOWNERS"]
HANDLE = re.compile(r"@([A-Za-z0-9][A-Za-z0-9-]*(?:/[A-Za-z0-9._-]+)?)")


def gh(path: str):
    req = urllib.request.Request(
        f"https://api.github.com{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "trida-stack-codeowners",
            **({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
               if os.environ.get("GITHUB_TOKEN") else {}),
        },
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def resolves(handle: str) -> tuple[bool, str]:
    if "/" in handle:  # @org/team -- needs org scope, which a default token lacks
        return True, "team (not checked)"
    try:
        gh(f"/users/{handle}")
        return True, "user"
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False, "404 -- renamed or deleted"
        return True, f"unverified (HTTP {e.code})"
    except Exception as e:  # network trouble must not read as a dead handle
        return True, f"unverified ({e})"


def main() -> int:
    path = next((ROOT / c for c in CANDIDATES if (ROOT / c).is_file()), None)
    if path is None:
        print("no CODEOWNERS file; nothing to check")
        return 0

    handles = sorted({m for line in path.read_text().splitlines()
                      if line.strip() and not line.lstrip().startswith("#")
                      for m in HANDLE.findall(line)})
    if not handles:
        print(f"{path.relative_to(ROOT)}: no owners declared")
        return 0

    dead = []
    for h in handles:
        ok, note = resolves(h)
        print(f"  {'OK  ' if ok else 'DEAD'}  @{h}  ({note})")
        if not ok:
            dead.append(h)

    if dead:
        print(
            f"\n{len(dead)} handle(s) in {path.relative_to(ROOT)} do not resolve: "
            + ", ".join("@" + d for d in dead)
            + "\n\nCODEOWNERS ignores these silently, so the rules naming them enforce "
              "fewer reviewers than they appear to. Update the file; GitHub does not "
              "redirect a renamed handle."
        )
        return 1

    print(f"\nOK: {len(handles)} owner handle(s) resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
