"""datamine: the app's chats (as Swift's JSONEncoder writes them) and request logs -> candidate training data,
with personal details scrubbed."""
import json
import tempfile
from pathlib import Path

from trida_mlx import datamine


def msg(role, content="", **kw):
    return {"id": kw.pop("id", f"{role}-{abs(hash(content)) % 10**8}"), "role": role, "content": content,
            "reasoning": kw.pop("reasoning", ""), "images": kw.pop("images", []), "toolRuns": kw.pop("toolRuns", []),
            **kw}


def run_(name, args, result, status="done"):
    return {"call": {"id": f"c-{name}", "name": name, "arguments": json.dumps(args)}, "status": {status: {}},
            "result": result, "images": []}


def test_mine_chats_and_logs():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "chats").mkdir()
    (tmp / "logs").mkdir()
    chat = {"id": "chat-1", "title": "t", "updated": 0, "messages": [
        msg("user", "translate this into Korean: good morning"),
        msg("assistant", "Good morning!", route="Tools: none (answer 0.83)", superseded=True,
            check="Check: ✗ language 0.08"),
        msg("assistant", "좋은 아침이에요!", route="Rewritten after review: language", check="Check: ✓ language 0.97"),
        msg("user", "한글로 해달라고"),   # reads like a correction of the reply above
        msg("assistant", "네, 좋은 아침이에요.", feedback="up", check="Check: ✓ language 0.95"),
        msg("user", "write a poem about Jay to /Users/jdoe/Desktop/jay.md"),
        msg("assistant", "", route="Tools: files 0.69", toolRuns=[
            run_("run_shell", {"command": "cat > jay.md"},
                 "error: run_shell is not needed for this message (the tool router allowed ...)", "denied")]),
        msg("assistant", "I've saved jay.md", editedFrom="I've saved jay.md", check="Check: ✗ claims match results 0.05"),
        msg("assistant", "Earlier conversation summarized", decide={"prompt": {}}),
    ]}
    chat["messages"][7]["content"] = "I could not save jay.md: the terminal call was refused."
    (tmp / "chats" / "chat-1.json").write_text(json.dumps({"version": 1, "chat": chat, "cards": {}}))
    log = {"path": "/v1/chat/completions",
           "body": {"messages": [{"role": "user", "content": "hi, I'm jdoe (jdoe@example.com)"}],
                    "temperature": 0.6, "chat_template_kwargs": {"enable_thinking": True}},
           "response": {"raw": "calculator", "content": "", "tool_calls": [{"name": "calculator", "arguments": {}}] * 3,
                        "stats": {"finish_reason": "repetition", "new_tokens": 40}}}
    (tmp / "logs" / "20261029-000000-aaaaaa.json").write_text(json.dumps(log))
    dec = {"path": "/v1/decide", "body": {"state": {"conversation": ["user: hi"]}, "questions": {"need": {}}},
           "response": {"answers": {"need": {"choice": "answer"}}}}
    (tmp / "logs" / "20261029-000001-bbbbbb.json").write_text(json.dumps(dec))

    out = tmp / "out"
    assert datamine.main(["--out", str(out), "--chats", str(tmp / "chats"), "--logs", str(tmp / "logs"),
                          "--scrub-name", "jdoe"]) == 0
    rows = {p.stem: [json.loads(l) for l in p.read_text().splitlines()] for p in out.glob("*.jsonl")}
    pairs = rows["preference_pairs"]
    assert {p["source"] for p in pairs} == {"review_rewrite", "user_edit"}
    rr = next(p for p in pairs if p["source"] == "review_rewrite")
    assert rr["rejected"] == "Good morning!" and rr["chosen"] == "좋은 아침이에요!" and rr["reasons"] == "language"
    assert rr["prompt"][-1]["content"].startswith("translate this")
    assert rows["feedback"][0]["rating"] == "up"
    tu = rows["tool_use"]
    assert len(tu) == 2 and tu[1]["refused_by_router"] == 1 and tu[1]["label_candidate"] == "needs_review"
    fails = rows["failures"]
    gen = next(f for f in fails if f["kind"] == "generation")
    assert set(gen["why"]) >= {"loop", "duplicate_tool_calls"} and "empty_answer" not in gen["why"]  # it did call tools
    assert any(f["kind"] == "user_corrected" and f["correction"] == "한글로 해달라고" for f in fails)
    assert rows["decisions"][0]["kind"] == "tool_routing" and len(rows["rollouts"]) == 1
    text = "".join(p.read_text() for p in out.glob("*.jsonl"))
    assert "jdoe" not in text and "@example.com" not in text and "/Users/USER/Desktop" in text
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["preference_pairs"] == 2 and manifest["scrubbed"] is True
