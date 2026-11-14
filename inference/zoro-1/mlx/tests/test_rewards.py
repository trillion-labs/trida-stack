"""rewards: the app's chats -> agent turns -> per-turn rewards and the judge-agreement report (CPU, no model)."""
import json
import tempfile
from pathlib import Path

from trida_mlx import rewards


def msg(role, content="", **kw):
    return {"id": kw.pop("id", f"{role}-{abs(hash(content)) % 10**8}"), "role": role, "content": content,
            "reasoning": "", "images": [], "toolRuns": kw.pop("toolRuns", []), **kw}


def run_(name, result, status="done"):
    return {"call": {"id": f"c-{name}", "name": name, "arguments": "{}"}, "status": {status: {}},
            "result": result, "images": []}


def chat():
    return {"id": "c1", "messages": [
        msg("user", "list my files", id="u1"),
        msg("assistant", "", toolRuns=[run_("list_dir", "error: no such folder")]),
        msg("assistant", "I couldn't find that folder.", id="a1", check="Check: ✗ complete 0.10"),
        msg("user", "try ~/Desktop", id="u2", retried=["Sorry, I can't do that."]),
        msg("assistant", "Your Desktop has 3 files.", id="a2", feedback="up"),
        msg("user", "/decide", decide={"prompt": {}}),
        msg("assistant", "Which question?", decide={"setup": {}}),
        msg("user", "thanks", id="u3"),
        msg("assistant", "You're welcome!", id="a3", editedFrom="No problem."),
    ], "ratings": [
        {"afterMessageID": "a2", "value": 0, "date": 0},     # dismissed: no signal, not a boundary
        {"afterMessageID": "a3", "value": 3, "date": 1},
    ]}


def test_split_turns_and_ratings():
    c = chat()
    ts = rewards.split_turns(c)
    rewards.attach_ratings(c, ts)
    main = [t for t in ts if t.get("kind") != "regenerated"]
    assert [t["user_id"] for t in main] == ["u1", "u2", "u3"]          # /decide set-up is not a turn
    t1, t2, t3 = main
    assert t1["signals"]["tool_errors"] == 1 and t1["signals"]["check_failed"]
    assert t1["next_user"] == "try ~/Desktop"
    assert t2["next_user"] == "thanks"                                  # skips the /decide prompt
    assert t3["reply"] == "No problem." and t3["edited_to"] == "You're welcome!" and t3["signals"]["edited"]
    assert t3["next_user"] is None
    regen = [t for t in ts if t.get("kind") == "regenerated"]
    assert [t["reply"] for t in regen] == ["Sorry, I can't do that."] and "rating" not in regen[0]
    assert all(t["rating"] == 3 for t in main)                          # the dismissal is not a boundary


def test_rewards_and_credit_assignment():
    p = {"u1": 0.1, "u2": 0.9, "u3": 0.6}
    seen = []

    def ask(state):
        seen.append(state)
        return next(v for k, v in p.items() if state["user_request"] == {"u1": "list my files", "u2": "try ~/Desktop",
                                                                          "u3": "thanks"}[k])
    cache = {}
    ts = rewards.build([chat()], ask, cache)
    by = {t["user_id"] + (t.get("kind") or ""): t for t in ts}
    assert len(seen) == 3 and len(cache) == 3                           # regenerated replies are not judged
    assert by["u2"]["reward"] == 1.0 and by["u2"]["reward_from"] == "feedback"
    assert by["u2regenerated"]["reward"] == -0.5
    # rating 3 (+1) over 3 turns: shares follow the judge, mean stays +1 before penalties and clipping
    shares = [by[k]["rating_share"] for k in ("u1", "u2", "u3")]
    assert abs(sum(shares) / 3 - 1.0) < 1e-9 and shares[0] < shares[2] < shares[1]
    assert by["u1"]["reward"] == round(max(-1.0, min(1.0, shares[0] - 1.0)), 4)  # tool error + failed check: -1
    assert by["u3"]["reward"] == round(min(1.0, shares[2] - 0.5), 4)    # edited: -0.5
    # a second run uses the cache
    rewards.build([chat()], ask, cache)
    assert len(seen) == 3
    assert set(seen[0]) == {"user_request", "assistant_tool_calls", "assistant_reply", "user_next_message"}
    assert "list_dir" in seen[0]["assistant_tool_calls"]


def test_negative_rating_blames_the_worst_turn():
    c = chat()
    c["ratings"] = [{"afterMessageID": "a3", "value": 1, "date": 0}]
    for m in c["messages"]:
        m.pop("feedback", None)
    ts = rewards.build([c], lambda s: 0.9 if s["user_request"] == "thanks" else 0.2)
    by = {t["user_id"]: t for t in ts if t.get("kind") != "regenerated"}
    assert by["u2"]["rating_share"] < by["u3"]["rating_share"] < 0     # low judge score -> more of the blame


def test_agreement_and_cli():
    ts = rewards.build([chat()], lambda s: 0.9 if "Desktop" in s["user_request"] else 0.3)
    rep = rewards.agreement(ts)
    assert rep["judge_by_rating"]["3"]["turns"] == 3 and rep["judge_auc_thumbs"]["up"] == 1
    assert rep["judge_auc_thumbs"]["auc"] is None                       # no 👎 yet
    assert rewards.auc([0.9, 0.8], [0.1, 0.8]) == 0.875
    tmp = Path(tempfile.mkdtemp())
    (tmp / "chats").mkdir()
    (tmp / "chats" / "c1.json").write_text(json.dumps({"chat": chat(), "cards": {}}))
    assert rewards.main(["--out", str(tmp / "out"), "--chats", str(tmp / "chats"), "--no-judge"]) == 0
    rows = [json.loads(line) for line in (tmp / "out" / "turn_rewards.jsonl").read_text().splitlines()]
    assert len(rows) == 4 and {r["reward_from"] for r in rows} >= {"feedback", "rating+penalties"}
    assert next(r for r in rows if r["user_id"] == "u3")["sft_target"] == "You're welcome!"
    assert json.loads((tmp / "out" / "agreement.json").read_text())["judged"] == 0
