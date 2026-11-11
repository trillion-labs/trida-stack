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
    # the draft's reply check, with the user's mark: the English draft is NOT in the language asked for
    req = {"state": {"user message": "translate this into Korean: good morning", "response": "Good morning!"},
           "questions": {"language": {"type": "noul", "instructions": "Is the response in the language asked for?"},
                         "request": {"type": "noul", "instructions": "Does it do what was asked?"}},
           "reasoning": "never"}
    chat["messages"][1]["decisions"] = [{"id": "d1", "kind": "Reply check", "request": json.dumps(req),
                                        "response": json.dumps({"answers": {"language": {"type": "noul", "noul": 0.32},
                                                                            "request": {"type": "noul", "noul": 0.9}}}),
                                        "labels": {"language": "false"}}]
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
    lab = rows["decision_labels"]          # kev's labelled record: only the marked question, yes/no as 0 / 1
    assert len(lab) == 1 and list(lab[0]["questions"]) == ["language"]
    assert lab[0]["questions"]["language"]["label"] == 0 and lab[0]["source"] == "Reply check"
    assert lab[0]["state"]["response"] == "Good morning!" and lab[0]["model_answers"]["language"]["noul"] == 0.32
    text = "".join(p.read_text() for p in out.glob("*.jsonl"))
    assert "jdoe" not in text and "@example.com" not in text and "/Users/USER/Desktop" in text
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["files"]["preference_pairs"] == 2 and manifest["scrubbed"] is True


def test_shared_decision_envelope_logs(tmp_path):
    request = {"state": "printer jam", "questions": {"need": {"type": "noul"}}, "image": "data:..."}
    record = {"path": "/v1/decide", "body": {"request": request, "reasoning": False},
              "response": {"answers": {"need": {"type": "noul", "noul": 0.9}}}}
    (tmp_path / "decision.json").write_text(json.dumps(record))
    decisions, _, _ = datamine.mine_logs(tmp_path)
    assert decisions[0]["kind"] == "tool_routing"
    assert decisions[0]["state"] == request["state"] and decisions[0]["questions"] == request["questions"]
    assert decisions[0]["image"]


# split so the repo's detect-private-key hook does not take the fake key below for a real one
PEM = "-----{} OPENSSH " + "PRIVATE KEY-----"


def test_scrub_catches_common_secrets():
    # a `cat .env` / credentials dump in tool output must not reach shared training data
    dump = ("WANDB_API_KEY=0123456789abcdef0123456789abcdef01234567\n"
            "SIMUSER_API_KEY='my-provider-key-123'\n"
            "aws_secret_access_key = wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY\n"
            '{"api_key": "abcdefgh12345678"}\n'
            "Authorization: Bearer abc.DEF-123_456789\n"
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U\n"
            "github_pat_11ABCDEFG0123456789_abcdefghijklmnopqrstuvwxyz\n"
            "xoxb-1234567890-abcdefghij\n"
            + PEM.format("BEGIN") + "\nb3BlbnNzaC1rZXktdjEAAAAA\n" + PEM.format("END") + "\n")
    out = datamine.scrub(dump)
    for secret in ("0123456789abcdef", "my-provider-key", "wJalrXUtnFEMI", "abcdefgh1234", "DEF-123_456789",
                   "eyJhbGci", "github_pat_11", "xoxb-12", "b3BlbnNzaC1rZXktdjEAAAAA"):
        assert secret not in out, secret
    assert "WANDB_API_KEY=" in out          # the name stays, so the example still reads as a .env file
    # ordinary words in token counts survive
    assert datamine.scrub('{"max_tokens": 600, "prompt_tokens": 12}') == '{"max_tokens": 600, "prompt_tokens": 12}'


def test_names_come_from_absolute_paths_only():
    rows = [{"text": "see https://x.io/home/index and /Users/jdoe/code"}]
    datamine.collect_names(rows)
    out = datamine.scrub("the index of /Users/jdoe/code, https://x.io/home/index")
    assert "jdoe" not in out and "the index of" in out and "x.io/home/index" in out


def test_keep_images_keeps_them_through_the_scrub():
    url = "data:image/png;base64,iVBORw0KGgo="
    assert datamine.scrub({"images": [url]}, keep_images=True) == {"images": [url]}
    assert datamine.scrub({"images": [url]}) == {"images": ["[image]"]}
