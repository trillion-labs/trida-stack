#!/usr/bin/env python3
"""Fail when a login in docs/metrics/first-party.txt no longer resolves on GitHub.

The roster decides which metrics series an item lands in, so a typo or a renamed
account silently moves someone's work into the external series -- the one that is
supposed to measure outside contributions. GitHub does not redirect renamed logins,
so a rename reads as a stranger.

Same shape as tools/check_codeowners.py. Network errors are reported as unverified
rather than failing the build.

Usage:  GITHUB_TOKEN=... python tools/check_first_party.py
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import urllib.error
import urllib.request

ROSTER = pathlib.Path(__file__).resolve().parent.parent / "docs/metrics/first-party.txt"


def logins() -> list[str]:
    return [
        line.strip()
        for line in ROSTER.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def resolves(login: str) -> bool | None:
    req = urllib.request.Request(f"https://api.github.com/users/{login}")
    req.add_header("Accept", "application/vnd.github+json")
    if token := os.environ.get("GITHUB_TOKEN"):
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            json.load(r)
        return True
    except urllib.error.HTTPError as e:
        return False if e.code == 404 else None
    except Exception:
        return None


def main() -> int:
    names = logins()
    if not names:
        print(f"{ROSTER}: no accounts listed", file=sys.stderr)
        return 1

    missing, unverified = [], []
    for name in names:
        state = resolves(name)
        mark = {True: "OK   ", False: "404  ", None: "?    "}[state]
        print(f"  {mark} @{name}")
        if state is False:
            missing.append(name)
        elif state is None:
            unverified.append(name)

    if missing:
        print(
            "\nThese logins do not exist. GitHub does not redirect renamed accounts, so a\n"
            "rename reads as a stranger and moves that person's work into the external\n"
            "series -- the one measuring outside contributions.\n",
            file=sys.stderr,
        )
        for name in missing:
            print(f"  @{name}", file=sys.stderr)
        return 1

    if unverified:
        print(f"\n{len(unverified)} login(s) could not be checked (network); not failing.")
    print(f"\nOK: {len(names)} first-party login(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
