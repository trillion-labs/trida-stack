"""Post a message to the project's chat channel.

Shared by the notifiers so there is one place that decides where a message goes and
what shape it takes. Discord and Slack want different payload keys, and the channel
decision is expressed as "whichever webhook secret is set" rather than hard-coded, so
the platform can change without touching the notifiers.

With neither secret set this prints and returns normally. A notifier that fails the
build because no webhook is configured would make every scheduled run red on a repo
that has not set one up yet, and a permanently red job is one people stop reading.
"""

from __future__ import annotations

import json
import os
import urllib.request

DISCORD = "DISCORD_WEBHOOK_URL"
SLACK = "SLACK_WEBHOOK_URL"


def destination() -> tuple[str | None, str | None]:
    """(platform, url). (None, None) when no webhook is configured.

    Discord wins if both are set -- two copies of every alert is worse than one.
    """
    if url := os.environ.get(DISCORD):
        return "discord", url
    if url := os.environ.get(SLACK):
        return "slack", url
    return None, None


def payload_for(platform: str, text: str, title: str) -> dict:
    if platform == "discord":
        return {"content": text}
    # Slack has no inherent title, so it is folded into the message body.
    return {"text": f"*{title}*\n{text}"}


def post(text: str, title: str) -> str:
    """Send, or print when unconfigured. Returns what happened, for logs and tests."""
    platform, url = destination()
    if platform is None:
        print(f"no {DISCORD} or {SLACK} set; printing instead:\n")
        print(text)
        return "printed"

    req = urllib.request.Request(
        url,
        data=json.dumps(payload_for(platform, text, title)).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        print(f"posted to {platform} ({r.status})")
    return f"posted:{platform}"
