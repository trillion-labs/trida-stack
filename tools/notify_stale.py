#!/usr/bin/env python3
"""Alert the chat channel about questions nobody has answered.

The guideline is explicit that piping every commit, PR and comment into chat makes
people stop reading, and lists what IS worth sending. "48시간 이상 담당자가 없는
질문" is on that list, and it is the one item that cannot be a GitHub-native
notification -- it is the absence of an event, not an event.

It also backstops the promotion discipline. If a maintainer answers in chat and never
records it on GitHub, the item keeps showing up here until someone does.

Posts to Discord or Slack depending on which webhook URL is set, so the channel
decision does not have to be made before this is written.

Usage:
    GITHUB_TOKEN=... DISCORD_WEBHOOK_URL=... python tools/notify_stale.py
    GITHUB_TOKEN=... SLACK_WEBHOOK_URL=...   python tools/notify_stale.py
    python tools/notify_stale.py --dry-run
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API = "https://api.github.com"
DEFAULT_REPO = "trillion-labs/trida-stack"
STALE_HOURS = 48


def gh(path: str, params: dict | None = None):
    url = f"{API}{path}" + ("?" + urllib.parse.urlencode(params) if params else "")
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github+json",
        "User-Agent": "trida-stack-community",
        **({"Authorization": f"Bearer {os.environ['GITHUB_TOKEN']}"}
           if os.environ.get("GITHUB_TOKEN") else {}),
    })
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def is_bot(user) -> bool:
    return (not user) or user.get("type") == "Bot" or user.get("login", "").endswith("[bot]")


def parse(ts):
    return dt.datetime.fromisoformat(ts.replace("Z", "+00:00")) if ts else None


def answered_by_someone_else(repo: str, number: int, author: str) -> bool:
    for c in gh(f"/repos/{repo}/issues/{number}/comments", {"per_page": 100}):
        if not is_bot(c.get("user")) and c.get("user", {}).get("login") != author:
            return True
    return False


def post(text: str, blocks_title: str) -> None:
    discord = os.environ.get("DISCORD_WEBHOOK_URL")
    slack = os.environ.get("SLACK_WEBHOOK_URL")
    if discord:
        payload, url = {"content": text}, discord
    elif slack:
        payload, url = {"text": f"*{blocks_title}*\n{text}"}, slack
    else:
        print("no DISCORD_WEBHOOK_URL or SLACK_WEBHOOK_URL set; printing instead:\n")
        print(text)
        return
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"posted ({r.status})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.environ.get("METRICS_REPO", DEFAULT_REPO))
    ap.add_argument("--hours", type=int, default=STALE_HOURS)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    now = dt.datetime.now(dt.timezone.utc)
    cutoff = now - dt.timedelta(hours=args.hours)

    stale = []
    for i in gh(f"/repos/{args.repo}/issues", {"state": "open", "per_page": 100}):
        if is_bot(i.get("user")):
            continue
        created = parse(i.get("created_at"))
        if created is None or created > cutoff:
            continue
        author = i.get("user", {}).get("login", "")
        if answered_by_someone_else(args.repo, i["number"], author):
            continue
        age_h = (now - created).total_seconds() / 3600
        stale.append((i, age_h))

    if not stale:
        print(f"nothing unanswered beyond {args.hours}h")
        return 0

    stale.sort(key=lambda t: -t[1])
    lines = [f"**{len(stale)} item(s) with no human response for over {args.hours}h**", ""]
    for i, age_h in stale[:10]:
        kind = "PR" if "pull_request" in i else "issue"
        lines.append(f"• [{kind} #{i['number']}]({i['html_url']}) — "
                     f"{int(age_h)}h — {i['title'][:80]}")
    if len(stale) > 10:
        lines.append(f"• …and {len(stale) - 10} more")
    lines += ["", "분류만 해도 됩니다. 상태 응답(\"확인 중, 다음 갱신 수요일\")도 "
                  "무응답보다 낫습니다."]
    text = "\n".join(lines)

    if args.dry_run:
        print(text)
        return 0
    post(text, "Unanswered questions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
