"""Example demo: Zoro-1 rates how believable your excuse is, then answers in one line.

  uv run my-demo --mock "my cat deleted my homework"
  uv run my-demo --model-url http://127.0.0.1:8080 "my cat deleted my homework"

Replace this with your demo. Keep a --mock flag so anyone can run it without a model.
"""
from __future__ import annotations

import argparse
import os

from .zoro import Zoro

LEVELS = ["obviously fake", "hard to believe", "could be true", "believable", "definitely true"]


def rate_excuse(zoro: Zoro, excuse: str) -> dict:
    answers = zoro.decide(
        {"situation": "A student is late with their homework and gives an excuse.", "excuse": excuse},
        {"believable": {"type": "score", "instructions": "How believable is this excuse?", "criteria": LEVELS},
         "funny": {"type": "noul", "instructions": "Is the excuse funny?"}})
    score = answers["believable"]["score"] / (len(LEVELS) - 1)     # E[level], scaled to 0-1
    reply = zoro.chat([{"role": "system", "content": "You are a strict but fair teacher. Answer in one sentence."},
                       {"role": "user", "content": f"My excuse: {excuse}"}])
    return {"believable": score, "funny": answers["funny"]["noul"], "reply": reply}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("excuse")
    ap.add_argument("--model-url", default=os.environ.get("ZORO_URL"))
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()
    if not a.mock and not a.model_url:
        ap.error("give --model-url (or ZORO_URL), or --mock")
    r = rate_excuse(Zoro(a.model_url, mock=a.mock), a.excuse)
    print(f"believable: {r['believable']:.0%}   funny: {r['funny']:.0%}\nteacher: {r['reply']}")


if __name__ == "__main__":
    main()
