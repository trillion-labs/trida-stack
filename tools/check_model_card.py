#!/usr/bin/env python3
"""Check that COMPLIANCE.md still describes the published model correctly.

The repository makes claims about a model it does not contain. Those claims drift
silently: trillionlabs/Trida2.0-4B went public on 2026-09-19 and five files still told
users to authenticate for two weeks, because nothing was watching.

This compares COMPLIANCE.md section 4b against Hugging Face's live metadata:

  * declared licence matches
  * visibility matches (public vs private/gated)

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
MODEL_ID = "trillionlabs/Trida2.0-4B"
API = f"https://huggingface.co/api/models/{MODEL_ID}"

# What COMPLIANCE.md 4b should say, keyed by what HF reports.
LICENCE_LABEL = {
    "apache-2.0": "Apache-2.0",
    "other": "`other` — terms not yet published",
}


def fetch() -> dict:
    req = urllib.request.Request(API, headers={"User-Agent": "trida-stack-supply-chain"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.load(r)
    except urllib.error.HTTPError as e:
        if e.code == 401:
            sys.exit(f"FAIL: {MODEL_ID} is not publicly readable (HTTP 401), but "
                     f"COMPLIANCE.md describes it as published. Update section 4b.")
        sys.exit(f"FAIL: could not reach the Hugging Face API for {MODEL_ID}: {e}")
    except Exception as e:  # network blips should not read as a licence change
        sys.exit(f"SKIP-WORTHY: could not reach the Hugging Face API ({e})")


def main() -> int:
    data = fetch()
    card = data.get("cardData") or {}
    hf_licence = (card.get("license") or "unspecified").lower()
    private = bool(data.get("private"))
    gated = bool(data.get("gated"))

    text = COMPLIANCE.read_text()
    m = re.search(r"\|\s*derived\s*\|.*?Trida2\.0-4B.*?\|\s*(.+?)\s*\|", text)
    if not m:
        sys.exit("FAIL: could not find the derived-model row in COMPLIANCE.md section 4b")
    claimed = m.group(1).replace("**", "").strip()

    expected = LICENCE_LABEL.get(hf_licence, hf_licence)
    problems = []

    if claimed != expected:
        problems.append(
            f"licence mismatch: Hugging Face reports {hf_licence!r} (expect the table to "
            f"read {expected!r}), COMPLIANCE.md section 4b says {claimed!r}"
        )

    if (private or gated) and "public" in text.lower().split("## 4b.")[1][:1200]:
        problems.append(
            f"visibility mismatch: HF reports private={private} gated={gated}, but "
            f"section 4b describes the model as published"
        )

    if problems:
        print(f"Model card check FAILED for {MODEL_ID}\n")
        for p in problems:
            print(f"  {p}")
        print(
            "\nThe model repository is the source of truth. Update COMPLIANCE.md section 4b "
            "(and any README that describes model access) to match it."
        )
        return 1

    print(f"OK: {MODEL_ID} reports licence {hf_licence!r}, private={private}, "
          f"gated={gated}; COMPLIANCE.md section 4b agrees.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
