#!/usr/bin/env python3
"""Collect the monthly metrics snapshot.

Three metrics to start, per the 성과관리 guideline's starter model:

  M-001  첫 사람 응답시간   median and p90 of (first_human_response_at - created_at)
  M-002  미응답률          share of items with no human response yet
  M-003  SLA 달성률        share answered within the threshold
  C-001  관심 신호          stars / forks / watchers

  M-101  첫 사람 응답시간(내부)  the same measurement over first-party items, reported as
                              its OWN series -- never merged into M-001..M-003. No
                              internal no-response RATE: see "not_a_rate" in the output.

Rules the guideline is explicit about, and which this implements:

  * Bots and the author's own comments are not responses (4.2).
  * Items that were never answered vanish from a response-time sample, so 미응답률
    is reported alongside it, never instead of it (4.2).
  * Median and p90, never the mean -- an average hides long-abandoned items (3.3).
  * Missing is not zero. A metric with no measurable population is recorded as
    null with status "no_data", not 0 (6.4).
  * Every value carries a data-quality status: good / partial / estimated / invalid (6.4).
  * `open_issues_count` includes PRs, so it is not named "issues" (6.3).
  * Stars are an interest signal, not evidence of use (1.3) -- labelled as such.

Writes metrics/snapshots/<period>.json, where a period is YYYY-MM-A (1st-15th)
or YYYY-MM-B (16th-end).

Usage:  GITHUB_TOKEN=... python tools/collect_metrics.py [--repo owner/name] [--period YYYY-MM]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import math
import statistics
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
DEFAULT_REPO = "trillion-labs/trida-stack"

# Provisional, not a committed target. The guideline puts a new project in a 4-8 week
# observation period where data quality is confirmed and targets are NOT yet set (3.3).
SLA_BUSINESS_DAYS = 2


def gh(path: str, params: dict | None = None) -> list | dict:
    url = f"{API}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "trida-stack-metrics",
        **({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"} if os.environ.get("GITHUB_TOKEN") else {}),
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def paginate(path: str, params: dict | None = None) -> list:
    out, page = [], 1
    while True:
        batch = gh(path, {**(params or {}), "per_page": 100, "page": page})
        if not batch:
            break
        out.extend(batch)
        if len(batch) < 100:
            break
        page += 1
        if page > 20:  # guard against runaway pagination
            break
    return out


def is_bot(user: dict | None) -> bool:
    if not user:
        return True
    return user.get("type") == "Bot" or user.get("login", "").endswith("[bot]")


def parse(ts: str | None) -> dt.datetime | None:
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile. int(n*q)-1 silently under-reports whenever n*q is not a
    whole number -- for n=5 it returns the 4th value, not the 5th -- and p90 exists to
    surface neglected outliers, so under-reporting defeats its purpose."""
    ordered = sorted(values)
    rank = max(1, math.ceil(q * len(ordered)))
    return ordered[rank - 1]


def business_hours(start: dt.datetime, end: dt.datetime) -> float:
    """Elapsed hours excluding weekends. Public holidays are NOT excluded -- recorded
    as a limitation rather than silently approximated."""
    if end <= start:
        return 0.0
    hours, cur = 0.0, start
    while cur < end:
        nxt = min(cur + dt.timedelta(hours=1), end)
        if cur.weekday() < 5:
            hours += (nxt - cur).total_seconds() / 3600
        cur = nxt
    return hours


def first_human_response(repo: str, number: int, author: str, is_pr: bool) -> dt.datetime | None:
    """Earliest comment or review by a human who is not the author."""
    candidates: list[dt.datetime] = []
    for c in paginate(f"/repos/{repo}/issues/{number}/comments"):
        if is_bot(c.get("user")) or c.get("user", {}).get("login") == author:
            continue
        if (t := parse(c.get("created_at"))):
            candidates.append(t)
    if is_pr:
        for r in paginate(f"/repos/{repo}/pulls/{number}/reviews"):
            if is_bot(r.get("user")) or r.get("user", {}).get("login") == author:
                continue
            if (t := parse(r.get("submitted_at"))):
                candidates.append(t)
    return min(candidates) if candidates else None


def default_period(now: dt.datetime) -> str:
    """ISO week, e.g. 2026-W42.

    Weekly during the observation period, monthly once a 90-day baseline exists. The
    guideline puts a new project in a 4-8 week observation period whose purpose is
    confirming data quality rather than hitting a target, and a quantity you check
    monthly takes a quarter to show you the collection is wrong. The snapshot is
    cumulative over the repository's whole history either way; the cadence sets only
    how often that cumulative figure is written down and reviewed.
    """
    iso = now.isocalendar()
    return f"{iso.year}-W{iso.week:02d}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("METRICS_REPO", DEFAULT_REPO))
    ap.add_argument("--period", default=None,
                    help="YYYY-MM-A or YYYY-MM-B (default: derived from today)")
    ap.add_argument("--out", default="metrics/snapshots")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    period = args.period or default_period(now)

    repo = gh(f"/repos/{args.repo}")

    # The metric exists to measure the EXTERNAL contributor experience. First-party PRs
    # that a maintainer opens and merges are not neglected contributions, and counting
    # them produces a 100% no-response rate that means nothing (guideline 4.2, appendix B:
    # split by 신규/기존 기여자). Split the population and report external separately.
    # First-party is decided from a hand-maintained list, not an API lookup. Both API
    # routes fail from a workflow because organisation membership here is private:
    # /collaborators shows GITHUB_TOKEN only DIRECT collaborators, and
    # author_association reports MEMBER only when the membership is visible. Collected
    # in Actions, both called 32 of our own items outside contributions -- from a single
    # account. A reviewer can check eight names; nobody can check a permission-dependent
    # API answer after the fact.
    roster = pathlib.Path(__file__).resolve().parent.parent / "docs/metrics/first-party.txt"
    insiders = {
        line.strip()
        for line in roster.read_text().splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not insiders:
        raise RuntimeError(f"{roster} lists no first-party accounts")

    # Every issue/PR ever opened. The repo is young; when it is not, window this.
    items = paginate(f"/repos/{args.repo}/issues", {"state": "all", "filter": "all"})
    items = [i for i in items if not is_bot(i.get("user"))]

    external_authors: set[str] = set()
    internal_authors: set[str] = set()
    waits, unanswered, within_sla, measured = [], 0, 0, 0
    internal_total = 0
    internal_waits, internal_unanswered = [], 0
    per_item = []
    for i in items:
        num, author = i["number"], i.get("user", {}).get("login", "")
        if author in insiders:
            internal_authors.add(author)
            # Measured as its OWN series, never folded into the external one. The external
            # metric answers "what is it like to contribute here"; mixing first-party items
            # into it produces a 100% no-response rate that means nothing (4.2). But
            # discarding them entirely leaves nothing measured at all while the project has
            # no outside contributors yet, and an unanswered first-party PR is still a real
            # operational signal -- #4 sat 138 business hours before anyone replied.
            internal_total += 1
            resp_i = first_human_response(args.repo, num, author, "pull_request" in i)
            if resp_i is None:
                internal_unanswered += 1
                per_item.append({"number": num, "population": "internal", "responded": False})
            else:
                bh_i = business_hours(parse(i.get("created_at")), resp_i)
                internal_waits.append(bh_i)
                per_item.append({"number": num, "population": "internal",
                                 "responded": True, "business_hours": round(bh_i, 2)})
            continue
        external_authors.add(author)
        created = parse(i.get("created_at"))
        resp = first_human_response(args.repo, num, author, "pull_request" in i)
        if resp is None:
            unanswered += 1
            per_item.append({"number": num, "population": "external", "responded": False})
            continue
        bh = business_hours(created, resp)
        waits.append(bh)
        measured += 1
        if bh <= SLA_BUSINESS_DAYS * 24:
            within_sla += 1
        per_item.append({"number": num, "population": "external", "responded": True,
                         "business_hours": round(bh, 2)})

    total = len(items) - internal_total   # external population only

    def metric(value, status, **extra):
        return {"value": value, "status": status, **extra}

    # Missing is not zero: with no population, the value is null and the status says why.
    if total == 0:
        note = (f"no external issues or PRs yet ({internal_total} first-party items "
                f"excluded -- see population)")
        resp_time = metric(None, "no_data", note=note)
        unans = metric(None, "no_data", note=note)
        sla = metric(None, "no_data", note=note)
    else:
        resp_time = (metric(None, "no_data", note="no item has received a human response")
                     if not waits else
                     metric({"median_business_hours": round(statistics.median(waits), 2),
                             "p90_business_hours": round(percentile(waits, 0.90), 2),
                             "n": len(waits)},
                            "good" if len(waits) >= 10 else "partial",
                            note=None if len(waits) >= 10 else f"sample of {len(waits)}; too small for a trend"))
        unans = metric(round(unanswered / total * 100, 1), "good",
                       numerator=unanswered, denominator=total)
        sla = (metric(None, "no_data", note="no measurable responses")
               if measured == 0 else
               metric(round(within_sla / measured * 100, 1),
                      "good" if measured >= 10 else "partial",
                      numerator=within_sla, denominator=measured,
                      threshold_business_days=SLA_BUSINESS_DAYS,
                      note="threshold is PROVISIONAL -- observation period, no committed target"))

    if internal_total == 0:
        internal_block = {
            "M-101_first_human_response": metric(None, "no_data", note="no first-party items"),
        }
    else:
        internal_block = {
            "M-101_first_human_response": (
                metric(None, "no_data", note="no first-party item has a human response")
                if not internal_waits else
                metric({"median_business_hours": round(statistics.median(internal_waits), 2),
                        "p90_business_hours": round(percentile(internal_waits, 0.90), 2),
                        "max_business_hours": round(max(internal_waits), 2),
                        "n": len(internal_waits)},
                       "good" if len(internal_waits) >= 10 else "partial",
                       note=None if len(internal_waits) >= 10
                       else f"sample of {len(internal_waits)}; too small for a trend")),
        }

    snapshot = {
        "period": period,
        "measured_at": now.isoformat(),
        "repository": args.repo,
        "definitions": "docs/metrics/README.md",
        "observation_period": True,
        "observation_note": (
            "Public since 2026-09-28. The guideline puts a new project in a 4-8 week "
            "observation period confirming data quality rather than setting targets, and "
            "wants a 90-day baseline before targets are committed. No target is in force."
        ),
        "metrics": {
            "M-001_first_human_response": resp_time,
            "M-002_no_response_rate_pct": unans,
            "M-003_sla_attainment_pct": sla,
            "C-001_interest_signals": metric(
                {"stars": repo.get("stargazers_count"),
                 "forks": repo.get("forks_count"),
                 "watchers": repo.get("subscribers_count")},
                "good",
                note="interest signals, NOT evidence of use (guideline 1.3)"),
            "C-002_open_issues_including_prs": metric(
                repo.get("open_issues_count"), "good",
                note="GitHub's open_issues_count includes PRs (guideline 6.3)"),
        },
        "internal_series": {
            "scope": "first-party items (docs/metrics/first-party.txt)",
            "why": ("Reported separately, never merged into M-001..M-003. The external "
                    "series is the contributor-experience metric; this one is the "
                    "maintainer-responsiveness metric. Keeping both means a period with no "
                    "outside contributors still produces a measured number instead of a "
                    "row of nulls that cannot be told apart from not measuring at all."),
            **internal_block,
            "items": internal_total,
            "authors": len(internal_authors),
            "responded": len(internal_waits),
            "closed_without_an_outside_response": internal_unanswered,
            "not_a_rate": ("Deliberately a count, not a percentage. Most first-party items "
                           "are a maintainer opening and merging their own work, which never "
                           "waited for anyone -- expressing that as a no-response RATE "
                           "produces a large number that reads as neglect and means nothing. "
                           "M-002 is the rate that carries meaning, over external items."),
        },
        "population": {
            "scope": "external contributors only",
            "external_items": total,
            # Items alone hid a classification bug: 2026-W43 reported 28 external
            # items that came from a single account of our own. A distinct-author
            # count makes that visible at a glance.
            "external_authors": len(external_authors),
            "internal_items_excluded": internal_total,
            "responded": measured,
            "unanswered": unanswered,
        },
        "limitations": [
            "Business hours exclude weekends but NOT public holidays.",
            "Bots are excluded by account type and a [bot] login suffix; a human account "
            "used for automation would not be caught.",
            "Response means a comment or review by someone other than the author. A fast "
            "formal reply does not imply resolution (guideline appendix B).",
            "First-party is read from docs/metrics/first-party.txt. Adding someone "
            "there moves their items between series, so the list is part of the "
            "measurement and changes to it belong in the period review.",
            "Snapshots before 2026-10-26 split the population from an API lookup a "
            "workflow token cannot answer, and are not comparable.",
        ],
        "items": per_item,
    }

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / f"{period}.json"
    path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n")
    print(f"wrote {path}")
    print(json.dumps(snapshot["metrics"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
