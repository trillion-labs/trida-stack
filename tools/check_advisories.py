#!/usr/bin/env python3
"""Gate vulnerability findings against the approved exception list.

Scanners run in report mode; this is what actually fails the build. That split is
deliberate: a scanner wired to `continue-on-error` is, in the guideline's words,
"관리가 아니라 무시" -- so the scan reports, and the policy decides.

Fails when:
  * a finding has no entry in docs/supply-chain/exceptions.md
  * an entry's `expires` date has passed
  * an entry is missing a required field

Usage:
    python tools/check_advisories.py --pip-audit audit.json
    python tools/check_advisories.py --grype grype.json
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXCEPTIONS = ROOT / "docs" / "supply-chain" / "exceptions.md"
REQUIRED = ("id", "component", "status", "reason", "approver", "expires")


def load_exceptions() -> tuple[dict, list[str]]:
    """Parse the yaml fence out of exceptions.md."""
    text = EXCEPTIONS.read_text()
    fences = re.findall(r"```yaml\n(.*?)```", text, re.S)
    if not fences:
        sys.exit(f"FAIL: no yaml code fence found in {EXCEPTIONS}")
    if len(fences) > 1:
        sys.exit(
            f"FAIL: {EXCEPTIONS} has {len(fences)} yaml fences. Entries outside the "
            f"first one would be silently ignored; keep them all in one fence."
        )
    m = type("M", (), {"group": staticmethod(lambda _: fences[0])})
    try:
        import yaml
    except ImportError:
        sys.exit("FAIL: pyyaml is required (pip install pyyaml)")
    entries = yaml.safe_load(m.group(1)) or []

    problems, by_id = [], {}
    today = dt.date.today()
    for e in entries:
        eid = e.get("id", "<no id>")
        for field in REQUIRED:
            if not e.get(field):
                problems.append(f"{eid}: missing required field '{field}'")
        expires = e.get("expires")
        if isinstance(expires, dt.date) and expires < today:
            problems.append(
                f"{eid}: exception expired on {expires}. Re-review it and set a new "
                f"date, or fix the finding -- do not extend it without re-reading."
            )
        if e.get("status") not in ("affected", "not_affected"):
            problems.append(f"{eid}: status must be 'affected' or 'not_affected'")
        if e.get("status") == "affected" and not e.get("mitigation"):
            problems.append(f"{eid}: status 'affected' requires a 'mitigation'")
        if eid in by_id:
            problems.append(f"{eid}: duplicate entry")
        by_id[eid] = e

    # Entries sitting outside the fence are invisible to this gate, which is exactly
    # the kind of silent failure an exception list must not have.
    in_fence = {e.get("id") for e in entries}
    for stray in re.findall(r"^- id:\s*(\S+)", text, re.M):
        if stray not in in_fence:
            problems.append(f"{stray}: entry is outside the yaml fence and is ignored")
    return by_id, problems


def findings_from_pip_audit(path: pathlib.Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    out = []
    for dep in data.get("dependencies", []):
        for v in dep.get("vulns", []):
            out.append((v["id"], dep.get("name", "?")))
            for alias in v.get("aliases", []):
                out.append((alias, dep.get("name", "?")))
    return out


def findings_from_grype(path: pathlib.Path) -> list[tuple[str, str]]:
    data = json.loads(path.read_text())
    out = []
    for match in data.get("matches", []):
        vuln = match.get("vulnerability", {})
        name = match.get("artifact", {}).get("name", "?")
        out.append((vuln.get("id", "?"), name))
        for rel in match.get("relatedVulnerabilities", []):
            out.append((rel.get("id", "?"), name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pip-audit", type=pathlib.Path)
    ap.add_argument("--grype", type=pathlib.Path)
    args = ap.parse_args()

    exceptions, problems = load_exceptions()

    findings: list[tuple[str, str]] = []
    if args.pip_audit and args.pip_audit.exists():
        findings += findings_from_pip_audit(args.pip_audit)
    if args.grype and args.grype.exists():
        findings += findings_from_grype(args.grype)

    # One finding can surface under several ids (PYSEC / GHSA / CVE aliases).
    # Treat it as covered if ANY of its ids is excepted.
    seen: dict[str, set[str]] = {}
    for vid, comp in findings:
        seen.setdefault(comp, set()).add(vid)

    uncovered = []
    for comp, ids in sorted(seen.items()):
        if not (ids & set(exceptions)):
            uncovered.append((comp, sorted(ids)))

    if uncovered:
        problems.append("")
        problems.append("Findings with no approved exception:")
        for comp, ids in uncovered:
            problems.append(f"  {comp}: {', '.join(ids)}")
        problems.append("")
        problems.append(
            "Fix the dependency, or add an entry to docs/supply-chain/exceptions.md "
            "with a reason, mitigation, approver and expiry."
        )

    if problems:
        print("Advisory gate FAILED\n")
        print("\n".join(problems))
        return 1

    total = sum(len(v) for v in seen.values())
    print(
        f"OK: {total} advisory id(s) across {len(seen)} package(s); "
        f"all covered by {len(exceptions)} approved exception(s), none expired."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
