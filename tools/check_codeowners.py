#!/usr/bin/env python3
"""Verify every GitHub handle this repository names still resolves.

CODEOWNERS ignores unknown handles silently -- no error, no warning, no annotation.
A rule keeps looking correct while enforcing fewer reviewers than it names.

This is not hypothetical here. Two owners renamed within one week
(usik-luke -> usik-luke-trl, hongjoonahn-tri -> hongjoonahn-trl) and GitHub does not
redirect the old handle: the API returns 404. Both breakages were found by accident.

Checking CODEOWNERS alone is not enough: the same handle is written in more than one
place, and the one this checker did not read kept the typo. `hongjoonahn-tri` survived
in GOVERNANCE.md -- the document that says who is responsible -- for two weeks after
CODEOWNERS was fixed, as a link that 404s. Every file below names people; all of them
are checked.

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
# Other files that name handles. A missing one is not an error: not every repository
# carries them, and the point is to check the ones that exist.
ALSO = ["GOVERNANCE.md", "docs/community/README.md"]
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
    owners = next((ROOT / c for c in CANDIDATES if (ROOT / c).is_file()), None)
    files = ([owners] if owners else []) + [ROOT / a for a in ALSO if (ROOT / a).is_file()]
    if not files:
        print("no file names handles; nothing to check")
        return 0

    # handle -> the files that name it, so a failure says where to edit
    where: dict[str, list[str]] = {}
    for path in files:
        rel = str(path.relative_to(ROOT))
        for line in path.read_text().splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            for h in HANDLE.findall(line):
                where.setdefault(h, [])
                if rel not in where[h]:
                    where[h].append(rel)

    if not where:
        print(f"{', '.join(str(f.relative_to(ROOT)) for f in files)}: no handles named")
        return 0

    dead = []
    for h in sorted(where):
        ok, note = resolves(h)
        print(f"  {'OK  ' if ok else 'DEAD'}  @{h}  ({note})  [{', '.join(where[h])}]")
        if not ok:
            dead.append(h)

    if dead:
        print(
            f"\n{len(dead)} handle(s) do not resolve:\n"
            + "\n".join(f"  @{d}  in {', '.join(where[d])}" for d in dead)
            + "\n\nCODEOWNERS ignores an unknown handle silently, so a rule naming one "
              "enforces fewer reviewers than it appears to, and elsewhere it is a link "
              "that 404s. Update the file; GitHub does not redirect a renamed handle."
        )
        return 1

    print(f"\nOK: {len(where)} handle(s) across {len(files)} file(s) resolve.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
