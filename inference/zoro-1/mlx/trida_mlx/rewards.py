"""Per-turn rewards for online RL on Zoro-1 Chat (phase 1: the signal only, no training).

    trida-mlx-rewards --out ~/zoro-rl/rewards                      # judge with the running app's server
    trida-mlx-rewards --out DIR --judge-url http://127.0.0.1:8090  # or any server with a decision head
    trida-mlx-rewards --out DIR --no-judge                         # explicit signals only

Reads the app's saved chats (as datamine does), splits them into agent turns, and gives each turn one reward in
[-1, 1] from:

  * the session rating ("How's the agent doing so far?" 1 bad / 2 good / 3 very good, ``chat.ratings``), spread
    over the turns it covers by the judge's scores (credit assignment);
  * 👍 / 👎 on the reply (overrides everything else);
  * penalties of -0.5 each for a tool error, a failed reply check and an edit, capped at -1; a reply the user
    threw away with Regenerate (``retried`` on the user message) becomes its own turn at -0.5;
  * the hindsight progress judge: Zoro-1's decision head asked, after the user's next message, "did this turn make
    progress toward what the user wanted?" (Actor, Judge, or Teacher?, arXiv 2611.17167: hindsight + progress is
    the setting where decision models judge best). The reward is 2p - 1.

Writes ``turn_rewards.jsonl`` (one turn per line: prompt, reply, signals, judge p, reward, an SFT target when the
user edited the reply) and ``agreement.json``: does the judge agree with the user (mean judge score by rating, the
judge's AUC on 👍 vs 👎)? Those are the phase 1 exit numbers. Personal details are scrubbed as in datamine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import urllib.request
from itertools import pairwise
from pathlib import Path

from .datamine import DEFAULT_CHATS, _decide_kind, _status, collect_names, load_chats, openai_view, scrub

RATING_REWARD = {1: -1.0, 2: 0.5, 3: 1.0}   # 0 = the card was dismissed: no signal
PENALTY = 0.5
JUDGE_QUESTION = {
    "progress": {
        "type": "noul",
        "instructions": "Judge with hindsight: the user's next message shows how this turn landed. Did the "
                        "assistant's turn make progress toward what the user wanted?",
    }
}
MAX_FIELD = 4000   # characters of each state field sent to the judge


# ---- turns ---------------------------------------------------------------------------------------------------

def split_turns(chat: dict) -> list:
    """One record per agent turn: a user message, the assistant messages and tool runs that answered it, and the
    user's next message. /decide set-up messages, notices and compaction summaries are not turns."""
    msgs = chat.get("messages") or []
    turns = []
    for i, m in enumerate(msgs):
        if m.get("role") != "user" or _decide_kind(m) == "prompt":
            continue
        j = i + 1
        while j < len(msgs) and msgs[j].get("role") != "user":
            j += 1
        asst = [r for r in msgs[i + 1:j] if r.get("role") == "assistant" and _decide_kind(r) is None]
        if not asst:
            continue
        final = next((r for r in reversed(asst) if r.get("content") and not r.get("superseded")), asst[-1])
        if final.get("error") and not final.get("content"):
            continue
        runs = [r for a in asst for r in (a.get("toolRuns") or [])]
        calls = [{"name": r["call"]["name"], "arguments": r["call"]["arguments"], "status": _status(r),
                  "result": (r.get("result") or "")[:500]} for r in runs]
        nxt = next((msgs[k] for k in range(j, len(msgs)) if msgs[k].get("role") == "user"
                    and _decide_kind(msgs[k]) != "prompt"), None)
        check = final.get("check") or ""
        turns.append({
            "chat": chat.get("id"), "user_id": m.get("id"), "index": i, "end": j - 1,
            "prompt": openai_view(msgs[: i + 1], False),
            "user": m.get("content") or "",
            "reply": final.get("editedFrom") or final.get("content") or "",
            "edited_to": final.get("content") if final.get("editedFrom") else None,
            "calls": calls,
            "next_user": nxt.get("content") if nxt else None,
            "signals": {
                "feedback": next((a.get("feedback") for a in reversed(asst) if a.get("feedback")), None),
                "edited": bool(final.get("editedFrom")),
                "tool_errors": sum(c["status"] == "done" and c["result"].startswith("error") for c in calls),
                "check_failed": "✗" in check,
                "regenerated": 0,
            },
        })
        # replies the user threw away with Regenerate: negative samples, no judge (nothing came after them)
        for text in m.get("retried") or []:
            turns.append({**turns[-1], "kind": "regenerated", "reply": text, "edited_to": None, "calls": [],
                          "next_user": None,
                          "signals": {"feedback": None, "edited": False, "tool_errors": 0, "check_failed": False,
                                      "regenerated": 1}})
    return turns


def attach_ratings(chat: dict, turns: list) -> None:
    """Each non-dismissed rating covers the turns since the previous rating (dismissals are not boundaries)."""
    msgs = chat.get("messages") or []
    where = {m.get("id"): k for k, m in enumerate(msgs)}
    lo = -1
    for r in sorted(chat.get("ratings") or [], key=lambda r: where.get(r.get("afterMessageID"), -1)):
        value, at = r.get("value"), where.get(r.get("afterMessageID"))
        if value not in RATING_REWARD or at is None:
            continue
        group = [t for t in turns if lo < t["index"] <= at and t.get("kind") != "regenerated"]
        for t in group:
            t["rating"] = value
            t["rating_group"] = f'{chat.get("id")}:{r.get("afterMessageID")}'
        lo = at


# ---- the judge -----------------------------------------------------------------------------------------------

def judge_state(t: dict) -> dict:
    calls = "\n".join(f'{c["name"]}({c["arguments"]}) -> {c["status"]}: {c["result"][:300]}' for c in t["calls"])
    return {
        "user_request": t["user"][:MAX_FIELD],
        "assistant_tool_calls": calls[:MAX_FIELD] or "(none)",
        "assistant_reply": t["reply"][:MAX_FIELD],
        "user_next_message": (t["next_user"] or "(none: the conversation ended here)")[:MAX_FIELD],
    }


def judge_key(t: dict) -> str:
    return hashlib.sha256(json.dumps(judge_state(t), sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:24]


def http_judge(url: str, timeout: float = 120):
    """A judge function over a running server's /v1/decide (also served while /v1/systemone rolls out)."""
    base = url.rstrip("/")
    base = base if base.endswith("/v1") else base + "/v1"

    def ask(state: dict) -> float:
        body = {"request": {"state": state, "questions": JUDGE_QUESTION}}
        req = urllib.request.Request(base + "/decide", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return float(json.load(r)["answers"]["progress"]["noul"])
    return ask


def run_judge(turns: list, ask, cache: dict) -> None:
    for t in turns:
        if t.get("kind") == "regenerated":
            continue
        k = judge_key(t)
        if k not in cache:
            try:
                cache[k] = ask(judge_state(t))
            except Exception as e:  # noqa: BLE001 - one bad turn must not stop the run
                print(f"judge failed on {t['chat']}#{t['index']}: {e}", file=sys.stderr)
                continue
        t["judge_p"] = cache[k]


# ---- rewards -------------------------------------------------------------------------------------------------

def _softmax(xs: list) -> list:
    m = max(xs)
    e = [math.exp(x - m) for x in xs]
    s = sum(e)
    return [x / s for x in e]


def assign_rewards(turns: list) -> None:
    """reward = 👍/👎 if set; else the rating's share (or the judge score) plus the penalties, clipped to [-1, 1].
    A rating R over n turns gives turn i R * n * w_i, w = softmax(sign(R) * judge score): its mean stays R and the
    judge decides which turns earned it (uniform without judge scores)."""
    groups: dict = {}
    for t in turns:
        if "rating_group" in t:
            groups.setdefault(t["rating_group"], []).append(t)
    for group in groups.values():
        R = RATING_REWARD[group[0]["rating"]]
        scores = [2 * t["judge_p"] - 1 if "judge_p" in t else 0.0 for t in group]
        w = _softmax([math.copysign(1, R) * s for s in scores])
        for t, wi in zip(group, w):
            t["rating_share"] = R * len(group) * wi
    for t in turns:
        s = t["signals"]
        if s["feedback"] in ("up", "down"):
            t["reward"], t["reward_from"] = (1.0 if s["feedback"] == "up" else -1.0), "feedback"
            continue
        if "rating_share" in t:
            base, src = t["rating_share"], "rating"
        elif "judge_p" in t:
            base, src = 2 * t["judge_p"] - 1, "judge"
        else:
            base, src = 0.0, "none"
        n = s["tool_errors"] + s["check_failed"] + s["edited"] + s["regenerated"]
        penalty = max(-1.0, -PENALTY * n)
        t["reward"] = round(max(-1.0, min(1.0, base + penalty)), 4)
        t["reward_from"] = src + ("+penalties" if n else "")


# ---- agreement (the phase 1 exit numbers) ----------------------------------------------------------------------

def auc(pos: list, neg: list):
    """P(a 👍 turn's judge score > a 👎 turn's), ties count half (Mann-Whitney)."""
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def agreement(turns: list) -> dict:
    by_rating = {}
    for v in (1, 2, 3):
        ps = [t["judge_p"] for t in turns if t.get("rating") == v and "judge_p" in t]
        by_rating[str(v)] = {"turns": len(ps), "mean_judge_p": round(sum(ps) / len(ps), 4) if ps else None}
    up = [t["judge_p"] for t in turns if t["signals"]["feedback"] == "up" and "judge_p" in t]
    down = [t["judge_p"] for t in turns if t["signals"]["feedback"] == "down" and "judge_p" in t]
    means = [by_rating[k]["mean_judge_p"] for k in ("1", "2", "3")]
    present = [m for m in means if m is not None]
    a = auc(up, down)
    return {
        "turns": len(turns), "judged": sum("judge_p" in t for t in turns),
        "rated_turns": sum("rating" in t for t in turns),
        "judge_by_rating": by_rating,
        "judge_auc_thumbs": {"auc": a, "up": len(up), "down": len(down)},
        "exit_criteria": {
            "judge_rises_with_rating": all(x < y for x, y in pairwise(present)) if len(present) >= 2 else None,
            "auc_at_least_0.7": None if a is None else a >= 0.7,
        },
        "note": "Phase 1 -> 2 needs both exit criteria true on enough data (at least ~20 rated sessions and ~30 "
                "rated replies of each kind).",
    }


# ---- main ----------------------------------------------------------------------------------------------------

def build(chats: list, ask=None, cache=None) -> list:
    turns = []
    for chat in chats:
        ts = split_turns(chat)
        attach_ratings(chat, ts)
        turns += ts
    if ask is not None:
        run_judge(turns, ask, cache if cache is not None else {})
    assign_rewards(turns)
    return turns


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chats", default=str(DEFAULT_CHATS))
    ap.add_argument("--judge-url", default="http://127.0.0.1:8080",
                    help="a trida-mlx-server with a decision head (the app's own server by default)")
    ap.add_argument("--no-judge", action="store_true", help="explicit signals only")
    ap.add_argument("--no-scrub", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cache_file = out / "judge_cache.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    chats = load_chats(Path(a.chats).expanduser())
    turns = build(chats, None if a.no_judge else http_judge(a.judge_url), cache)
    cache_file.write_text(json.dumps(cache))
    on = not a.no_scrub
    if on:
        collect_names([turns])
    with open(out / "turn_rewards.jsonl", "w", encoding="utf-8") as fh:
        for t in turns:
            row = {k: v for k, v in t.items() if k not in ("index", "end", "rating_group")}
            if t["edited_to"]:
                row["sft_target"] = t["edited_to"]
            fh.write(json.dumps(scrub(row, on), ensure_ascii=False) + "\n")
    report = agreement(turns)
    (out / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
