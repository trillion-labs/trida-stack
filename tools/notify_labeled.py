#!/usr/bin/env python3
"""Announce an item that just got a label we watch.

GitHub repository webhooks subscribe by event type and cannot filter on labels, so
"tell the channel when something is marked help wanted" is not something the webhook
can express. This runs in Actions, where the label is in the event payload.

Only labels in WATCHED produce a message. Everything else exits quietly -- the point
of a channel alert is that it means something, and a channel that announces every
label change is one people mute.

Usage (in Actions):
    GITHUB_EVENT_PATH=... DISCORD_WEBHOOK_URL=... python tools/notify_labeled.py
Locally:
    python tools/notify_labeled.py --event /tmp/event.json --dry-run
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from chat import post  # noqa: E402

# Keep this short. Each entry is a promise that the channel will care.
WATCHED = {
    "help wanted": "메인테이너가 당장 잡지 않는 작업입니다. 외부 기여를 환영합니다.",
    "good first issue": "첫 기여로 적당한 작업입니다. 배경과 검증 방법이 본문에 있습니다.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", type=pathlib.Path, default=None,
                    help="event payload json (default: $GITHUB_EVENT_PATH)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    import os
    raw = args.event or os.environ.get("GITHUB_EVENT_PATH") or ""
    # Path("") is the current directory, and a directory passes .exists(), so an unset
    # GITHUB_EVENT_PATH must be rejected before it becomes a path at all.
    if not str(raw):
        print("no event payload; nothing to do")
        return 0
    path = pathlib.Path(raw)
    if not path.is_file():
        print(f"event payload {path} is not a file; nothing to do")
        return 0

    event = json.loads(path.read_text())
    label = (event.get("label") or {}).get("name", "")
    if label not in WATCHED:
        print(f"label {label!r} is not watched; nothing to do")
        return 0

    item = event.get("issue") or event.get("pull_request") or {}
    number = item.get("number")
    title = item.get("title", "")
    url = item.get("html_url", "")
    kind = "PR" if "pull_request" in item or event.get("pull_request") else "issue"

    text = "\n".join([
        f"**`{label}`** — [{kind} #{number}]({url})",
        f"{title}",
        "",
        WATCHED[label],
    ])

    if args.dry_run:
        print(text)
        return 0
    post(text, f"{label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
