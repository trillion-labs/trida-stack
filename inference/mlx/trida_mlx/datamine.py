"""Turn Zoro-1 Chat's local records into candidate training data (JSONL), with personal details scrubbed.

    trida-mlx-datamine --out ~/zoro-training-data        # defaults read the app's chats and request logs

Reads
  * chats:  ~/Library/Application Support/Zoro-1 Chat/chats/*.json (the app's saved conversations, with the
    harness's tool-routing line, reply checks, rewritten drafts, user feedback and user edits), and
  * logs:   ~/Library/Logs/Zoro-1 Chat/requests/*.json (every request the server got and its raw output).

Writes (one JSON object per line; see manifest.json for counts)
  preference_pairs.jsonl  prompt + rejected / chosen replies: a draft rewritten after a failed reply check
                          (source "review_rewrite") or a reply the user edited (source "user_edit")
  feedback.jsonl          replies the user rated 👍 / 👎, with the conversation before them
  tool_use.jsonl          every user turn in agent mode: what the router decided, which tools were called and
                          how they ended, refused / duplicate / gated calls, and a candidate label
  decisions.jsonl         every typed decision the server answered (tool routing, reply checks, /decide, tools):
                          state, questions, answers, reasoning; the model's own verdicts, not gold labels
  rollouts.jsonl          every logged generation as a trajectory: the exact request (messages, tools, template
                          switches, sampling) and the model's raw output (prompts / samples for on-policy
                          distillation or RL)
  failures.jsonl          generations that went wrong: loops, budget hit, empty answer, tool errors, gated tool
                          calls, and replies the user corrected in the next message

All labels here come from the model, the harness or one user: treat them as candidates and confirm them
(teacher model or a person) before training. Images are dropped unless --keep-images.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

HOME = Path.home()
DEFAULT_CHATS = HOME / "Library/Application Support/Zoro-1 Chat/chats"
DEFAULT_LOGS = HOME / "Library/Logs/Zoro-1 Chat/requests"

# ---- scrubbing ---------------------------------------------------------------------------------------------

_SCRUB = [
    (re.compile(r"(?<![\w])/?Users/[^/\s\"'\\]+"), "/Users/USER"),
    (re.compile(r"(?<![\w])/?home/[^/\s\"'\\]+"), "/home/USER"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"), "<EMAIL>"),
    (re.compile(r"\bhf_[A-Za-z0-9]{20,}"), "<SECRET>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), "<SECRET>"),
    (re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), "<SECRET>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<SECRET>"),
    (re.compile(r"data:image/[a-z]+;base64,[A-Za-z0-9+/=]+"), "[image]"),
]


_NAMES: list = []   # account names found in paths (/Users/<name>, /home/<name>, /mnt/.../<name>/): scrubbed everywhere


def collect_names(rows, extra=()) -> None:
    """Account names to scrub wherever they appear (file names, prompts), from the paths in the data."""
    import getpass
    found = set(n for n in extra if n)
    try:
        found.add(getpass.getuser())
    except Exception:  # noqa: BLE001
        pass
    text = json.dumps(rows, ensure_ascii=False)
    found |= set(re.findall(r"(?:Users|home)/([A-Za-z][A-Za-z0-9._-]{2,})", text))
    common = {"user", "users", "shared", "guest", "runner", "ubuntu", "root", "admin", "home", "test", "default"}
    found = {n for n in found if n.lower() not in common and len(n) >= 3}
    _NAMES[:] = sorted(found, key=len, reverse=True)


def scrub(x, on: bool = True):
    if not on:
        return x
    if isinstance(x, str):
        for pat, rep in _SCRUB:
            x = pat.sub(rep, x)
        for n in _NAMES:
            x = re.sub(r"(?<![A-Za-z0-9])" + re.escape(n) + r"(?![a-z0-9])", "USER", x)
        return x
    if isinstance(x, list):
        return [scrub(v, on) for v in x]
    if isinstance(x, dict):
        return {k: scrub(v, on) for k, v in x.items()}
    return x


# ---- chats (the app's saved conversations) --------------------------------------------------------------

def load_chats(d: Path) -> list:
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            s = json.loads(f.read_text())
        except Exception:  # noqa: BLE001 - a half-written file
            continue
        chat = s.get("chat") or {}
        chat["_cards"] = s.get("cards") or {}
        chat["_file"] = f.name
        out.append(chat)
    return out


def _decide_kind(m: dict):
    d = m.get("decide")
    if not d:
        return None
    return next(iter(d)) if isinstance(d, dict) else d


def openai_view(msgs: list, keep_images: bool) -> list:
    """The messages as the model saw them (the app's history rules): no /decide set-up messages or notices,
    no rewritten drafts, the latest compaction summary instead of what came before it."""
    start, summary = 0, None
    for i, m in enumerate(msgs):
        if m.get("compaction"):
            start, summary = i + 1, m["compaction"]
    out = []
    if summary:
        out.append({"role": "system", "content": "Summary of the earlier conversation:\n" + summary})
    for m in msgs[start:]:
        if _decide_kind(m) == "prompt" or m.get("superseded"):
            continue
        if m.get("error") and not m.get("content"):
            continue
        role = m.get("role")
        e = {"role": role, "content": m.get("content") or ""}
        imgs = m.get("images") or []
        if imgs:
            if keep_images:
                e["images"] = ["data:image/png;base64," + (i if isinstance(i, str) else base64.b64encode(bytes(i)).decode())
                               for i in imgs]
            else:
                e["content"] = ("[image] " * len(imgs)) + e["content"]
        if role == "assistant":
            if m.get("reasoning"):
                e["reasoning"] = m["reasoning"]
            runs = m.get("toolRuns") or []
            if runs:
                e["tool_calls"] = [{"name": r["call"]["name"], "arguments": r["call"]["arguments"]} for r in runs]
        if role == "tool":
            e["name"] = m.get("toolName")
        out.append(e)
    return out


def _status(run: dict) -> str:
    s = run.get("status")
    return next(iter(s)) if isinstance(s, dict) else str(s)


_CORRECTION = re.compile(
    r"(아니|다시|틀렸|잘못|왜 |멍청|말고|한글로|한국어로|영어로|그게 아니|이해 못|"
    r"\bno[,.! ]|\bwrong\b|\bagain\b|not what|i said|i asked|\bin english\b|\bin korean\b|you didn'?t|that'?s not)",
    re.IGNORECASE)


def mine_chats(chats: list, keep_images: bool):
    pairs, feedback, tool_use, corrections = [], [], [], []
    for chat in chats:
        msgs = chat.get("messages") or []
        cid = chat.get("id")
        for i, m in enumerate(msgs):
            role = m.get("role")
            # rewritten after review: the superseded draft vs the reply that replaced it
            if role == "assistant" and m.get("superseded"):
                j = next((k for k in range(i + 1, len(msgs)) if msgs[k].get("role") == "assistant"
                          and (msgs[k].get("route") or "").startswith("Rewritten after review")), None)
                if j is not None and msgs[j].get("content"):
                    pairs.append({"source": "review_rewrite", "chat": cid,
                                  "prompt": openai_view(msgs[:i], keep_images),
                                  "rejected": m.get("content"), "chosen": msgs[j]["content"],
                                  "rejected_check": m.get("check"), "chosen_check": msgs[j].get("check"),
                                  "reasons": (msgs[j].get("route") or "").split(": ", 1)[-1]})
            if role == "assistant" and m.get("editedFrom"):
                pairs.append({"source": "user_edit", "chat": cid, "prompt": openai_view(msgs[:i], keep_images),
                              "rejected": m["editedFrom"], "chosen": m.get("content")})
            if role == "assistant" and m.get("feedback"):
                feedback.append({"chat": cid, "rating": m["feedback"], "prompt": openai_view(msgs[:i], keep_images),
                                 "reply": m.get("editedFrom") or m.get("content"), "check": m.get("check"),
                                 "route": m.get("route")})
            if role != "user" or _decide_kind(m) == "prompt":
                continue
            # the replies to this user message, up to the next user message
            j = i + 1
            replies = []
            while j < len(msgs) and msgs[j].get("role") != "user":
                replies.append(msgs[j])
                j += 1
            asst = [r for r in replies if r.get("role") == "assistant" and _decide_kind(r) is None]
            if not asst:
                continue
            final = next((r for r in reversed(asst) if r.get("content") and not r.get("superseded")), asst[-1])
            # the next user message reads like a correction of this reply
            nxt = msgs[j] if j < len(msgs) and _decide_kind(msgs[j]) != "prompt" else None
            if nxt and _CORRECTION.search(nxt.get("content") or ""):
                corrections.append({"kind": "user_corrected", "chat": cid,
                                    "prompt": openai_view(msgs[: i + 1], keep_images),
                                    "reply": final.get("content"), "correction": nxt.get("content")})
            runs = [r for a in asst for r in (a.get("toolRuns") or [])]
            route = next((a.get("route") for a in asst if a.get("route")), None)
            if not runs and not route:
                continue  # not an agent-mode turn
            calls = [{"name": r["call"]["name"], "arguments": r["call"]["arguments"], "status": _status(r),
                      "result": (r.get("result") or "")[:500]} for r in runs]
            refused = [c for c in calls if "is not needed for this message" in c["result"]]
            failed = [c for c in calls if c["status"] == "done" and c["result"].startswith("error")]
            denied = [c for c in calls if c["status"] == "denied" and c not in refused]
            seen, dups = set(), 0
            for c in calls:
                k = (c["name"], c["arguments"])
                dups += k in seen
                seen.add(k)
            check = final.get("check") or ""
            good = check.startswith("Check:") and "✗" not in check
            ok_calls = [c for c in calls if c["status"] == "done" and not c["result"].startswith("error")]
            if good and not calls:
                label = "no_tool"
            elif good and ok_calls and not refused:
                label = "tool:" + ",".join(sorted({c["name"] for c in ok_calls}))
            else:
                label = "needs_review"
            tool_use.append({"chat": cid, "prompt": openai_view(msgs[: i + 1], keep_images), "route": route,
                             "calls": calls, "refused_by_router": len(refused), "tool_errors": len(failed),
                             "denied_by_user": len(denied), "duplicate_calls": dups, "final_reply": final.get("content"),
                             "final_check": check or None, "label_candidate": label})
    return pairs, feedback, tool_use, corrections


# ---- request logs (what the server got and produced) ------------------------------------------------------

def mine_logs(d: Path):
    decisions, failures, rollouts = [], [], []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        try:
            rec = json.loads(f.read_text())
        except Exception:  # noqa: BLE001
            continue
        path, body, resp = rec.get("path", ""), rec.get("body") or {}, rec.get("response")
        if path.endswith("/decide"):
            qs = body.get("questions") or {}
            keys = set(qs)
            kind = ("tool_routing" if keys in ({"need"}, {"needs_tool", "kind"}) else
                    "reply_check" if keys & {"language", "request", "complete"} else
                    "ui_click" if keys & {"click", "field"} else "other")
            decisions.append({"log": f.name, "kind": kind, "state": body.get("state"), "questions": qs,
                              "image": bool(body.get("image")), "answers": (resp or {}).get("answers"),
                              "direct": ((resp or {}).get("direct") or {}).get("answers"),
                              "escalated": (resp or {}).get("escalated"), "reasoning": (resp or {}).get("reasoning"),
                              "cut": (resp or {}).get("cut")})
            continue
        if not path.endswith("chat/completions") or not isinstance(resp, dict):
            continue
        st = resp.get("stats") or {}
        # the trajectory itself: the exact request (messages, tools, template switches) and the model's raw
        # output, i.e. an on-policy sample of the deployed model (for on-policy distillation / RL)
        rollouts.append({"log": f.name, "messages": body.get("messages"), "tools": body.get("tools"),
                         "tool_choice": body.get("tool_choice"),
                         "chat_template_kwargs": body.get("chat_template_kwargs"),
                         "sampling": {k: body.get(k) for k in ("temperature", "top_k", "top_p")},
                         "trida_mode": body.get("trida_mode"), "completion_raw": resp.get("raw"),
                         "finish_reason": st.get("finish_reason"), "new_tokens": st.get("new_tokens")})
        why = []
        if st.get("finish_reason") == "repetition":
            why.append("loop")
        if st.get("finish_reason") == "length":
            why.append("budget")
        if st.get("loop_stopped_thinking"):
            why.append("loop_in_thinking")
        if st.get("gate_blocked"):
            why.append("tool_call_gated")
        calls = resp.get("tool_calls") or []
        if len(calls) > len({(c.get("name"), json.dumps(c.get("arguments"), sort_keys=True)) for c in calls}):
            why.append("duplicate_tool_calls")
        if not (resp.get("content") or "").strip() and not calls:
            why.append("empty_answer")
        last = (body.get("messages") or [{}])[-1]
        if last.get("role") == "tool" and isinstance(last.get("content"), str) and last["content"].startswith("error"):
            why.append("after_tool_error")   # this generation reacts to a tool error
        if why:
            failures.append({"kind": "generation", "log": f.name, "why": why, "messages": body.get("messages"),
                             "tools": [t.get("function", {}).get("name") for t in body.get("tools") or []],
                             "tool_choice": body.get("tool_choice"), "raw": (resp.get("raw") or "")[:20000],
                             "stats": st})
    return decisions, failures, rollouts


# ---- main ----------------------------------------------------------------------------------------------

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--chats", default=str(DEFAULT_CHATS))
    ap.add_argument("--logs", default=str(DEFAULT_LOGS))
    ap.add_argument("--keep-images", action="store_true", help="keep attached images (as data URLs)")
    ap.add_argument("--no-scrub", action="store_true", help="keep user names in paths, emails and tokens")
    ap.add_argument("--scrub-name", action="append", default=[], help="another name to replace with USER (repeatable)")
    a = ap.parse_args(argv)
    on = not a.no_scrub
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    chats = load_chats(Path(a.chats).expanduser())
    pairs, feedback, tool_use, corrections = mine_chats(chats, a.keep_images)
    decisions, failures, rollouts = mine_logs(Path(a.logs).expanduser())
    files = {"preference_pairs": pairs, "feedback": feedback, "tool_use": tool_use, "decisions": decisions,
             "failures": failures + corrections, "rollouts": rollouts}
    if on:
        collect_names(list(files.values()), a.scrub_name)
    counts = {}
    for name, rows in files.items():
        with open(out / f"{name}.jsonl", "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(scrub(r, on), ensure_ascii=False) + "\n")
        counts[name] = len(rows)
    labels = {}
    for r in tool_use:
        labels[r["label_candidate"].split(":")[0]] = labels.get(r["label_candidate"].split(":")[0], 0) + 1
    manifest = {"chats_read": len(chats), "files": counts, "tool_use_labels": labels,
                "scrubbed": on, "images": "kept" if a.keep_images else "dropped",
                "note": "labels are candidates from the model, the harness or one user; confirm before training. "
                        "Tool results can include file names and file contents from the workspace: review before "
                        "sharing."}
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    print(json.dumps(manifest, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
