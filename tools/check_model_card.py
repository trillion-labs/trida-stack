#!/usr/bin/env python3
"""Check that COMPLIANCE.md still describes the published models correctly.

The repository makes claims about models it does not contain. Those claims drift
silently: trillionlabs/Trida2.0-4B went public on 2026-09-19 and five files still told
users to authenticate for two weeks, because nothing was watching. It was later renamed
trillionlabs/Zoro, and the served default became the gated trillionlabs/Zoro-1, while this
check still watched the old id.

For every model in MODELS, this compares its row in COMPLIANCE.md section 4b against
Hugging Face's live metadata:

  * the licence column matches the card's licence
  * the access column (public / gated / private) matches the repository's visibility

Needs network. Run: python tools/check_model_card.py
"""

from __future__ import annotations

import json
import pathlib
import re
import sys
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parent.parent
COMPLIANCE = ROOT / "COMPLIANCE.md"

# The derived checkpoints this repository serves or publishes. Each needs a section 4b row.
MODELS = (
    "trillionlabs/Zoro-1",   # served by inference/zoro-1/
    "trillionlabs/Zoro",     # published as trillionlabs/Trida2.0-4B, which now redirects here
)

# What the licence column should say, keyed by what HF reports.
LICENCE_LABEL = {
    "apache-2.0": "Apache-2.0",
    "other": "`other` — terms not yet published",
}

# | kind | [org/name](url) ... | licence | access | verified |
ROW_RE = re.compile(r"^\|\s*(?:base|derived)\s*\|\s*\[([^\]]+)\]\([^)]*\)[^|]*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|",
                    re.M)


def model_rows(text: str) -> dict:
    """{model id: {"licence", "access"}} from the section 4b table."""
    return {m.group(1): {"licence": m.group(2).replace("**", "").strip(), "access": m.group(3).strip().lower()}
            for m in ROW_RE.finditer(text)}


def access(data: dict) -> str:
    if data.get("private"):
        return "private"
    return "gated" if data.get("gated") else "public"     # HF reports gated as False, "auto" or "manual"


def check(model_id: str, data: dict, rows: dict) -> list:
    """-> the ways section 4b disagrees with `data` (HF metadata) for `model_id`."""
    row = rows.get(model_id)
    if row is None:
        return [f"{model_id}: no row in COMPLIANCE.md section 4b"]
    problems = []
    hf_licence = ((data.get("cardData") or {}).get("license") or "unspecified").lower()
    expected = LICENCE_LABEL.get(hf_licence, hf_licence)
    if row["licence"] != expected:
        problems.append(f"{model_id}: licence mismatch: Hugging Face reports {hf_licence!r} (expect "
                        f"{expected!r}), section 4b says {row['licence']!r}")
    if row["access"] != access(data):
        problems.append(f"{model_id}: access mismatch: Hugging Face reports {access(data)!r} "
                        f"(private={data.get('private')}, gated={data.get('gated')}), section 4b says "
                        f"{row['access']!r}")
    return problems


def fetch(model_id: str) -> dict:
    req = urllib.request.Request(f"https://huggingface.co/api/models/{model_id}",
                                 headers={"User-Agent": "trida-stack-supply-chain"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit(f"FAIL: {model_id} is not publicly readable (HTTP 401), but "
                     f"COMPLIANCE.md describes it as published. Update section 4b.")
        sys.exit(f"FAIL: could not reach the Hugging Face API for {model_id}: {e}")
    except Exception as e:  # network blips should not read as a licence change
        sys.exit(f"SKIP-WORTHY: could not reach the Hugging Face API ({e})")


def main() -> int:
    rows = model_rows(COMPLIANCE.read_text())
    problems = []
    for model_id in MODELS:
        data = fetch(model_id)
        found = check(model_id, data, rows)
        problems += found
        if not found:
            print(f"OK: {model_id} reports licence "
                  f"{((data.get('cardData') or {}).get('license') or 'unspecified')!r}, "
                  f"access {access(data)!r}; COMPLIANCE.md section 4b agrees.")

    if problems:
        print("Model card check FAILED\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nThe model repository is the source of truth. Update COMPLIANCE.md section 4b "
            "(and any README that describes model access) to match it."
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
