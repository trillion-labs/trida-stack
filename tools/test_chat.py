"""Tests for chat destination selection and the label notifier.

Neither can be checked live: no webhook is configured, so every real run takes the
"print instead" branch and the send path never executes. A notifier that silently
posts nowhere looks exactly like one that works, which is why these exist.
"""

import json
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import chat  # noqa: E402
import notify_labeled  # noqa: E402

DISCORD_URL = "https://discord.example/api/webhooks/1/abc"
SLACK_URL = "https://hooks.slack.example/services/T/B/x"


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    monkeypatch.delenv(chat.DISCORD, raising=False)
    monkeypatch.delenv(chat.SLACK, raising=False)


def test_no_webhook_configured_prints_and_does_not_raise(capsys):
    assert chat.post("hello", "Title") == "printed"
    assert "hello" in capsys.readouterr().out


def test_discord_selected_when_only_discord_is_set(monkeypatch):
    monkeypatch.setenv(chat.DISCORD, DISCORD_URL)
    assert chat.destination() == ("discord", DISCORD_URL)


def test_slack_selected_when_only_slack_is_set(monkeypatch):
    monkeypatch.setenv(chat.SLACK, SLACK_URL)
    assert chat.destination() == ("slack", SLACK_URL)


def test_discord_wins_when_both_are_set(monkeypatch):
    # Two copies of every alert is worse than one.
    monkeypatch.setenv(chat.DISCORD, DISCORD_URL)
    monkeypatch.setenv(chat.SLACK, SLACK_URL)
    assert chat.destination()[0] == "discord"


def test_payload_shape_differs_per_platform():
    # Discord reads "content"; Slack reads "text" and has no title field of its own.
    assert chat.payload_for("discord", "body", "T") == {"content": "body"}
    assert chat.payload_for("slack", "body", "T") == {"text": "*T*\nbody"}


def test_send_path_posts_json_to_the_configured_url(monkeypatch):
    monkeypatch.setenv(chat.DISCORD, DISCORD_URL)
    captured = {}

    class _Resp:
        status = 204
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["body"] = json.loads(req.data.decode())
        captured["ctype"] = req.headers.get("Content-type")
        captured["method"] = req.get_method()
        return _Resp()

    monkeypatch.setattr(chat.urllib.request, "urlopen", fake_urlopen)
    assert chat.post("the message", "Unanswered questions") == "posted:discord"
    assert captured["url"] == DISCORD_URL
    assert captured["body"] == {"content": "the message"}
    assert captured["ctype"] == "application/json"
    assert captured["method"] == "POST"


# --- notify_labeled -------------------------------------------------------------

def _event(tmp_path, label, number=42):
    p = tmp_path / "event.json"
    p.write_text(json.dumps({
        "action": "labeled",
        "label": {"name": label},
        "issue": {"number": number, "title": "제목",
                  "html_url": f"https://github.com/o/r/issues/{number}"},
    }))
    return p


def test_watched_label_produces_a_message(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["x", "--event", str(_event(tmp_path, "help wanted")), "--dry-run"])
    assert notify_labeled.main() == 0
    out = capsys.readouterr().out
    assert "help wanted" in out and "#42" in out


def test_unwatched_label_is_silent(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv",
                        ["x", "--event", str(_event(tmp_path, "documentation")), "--dry-run"])
    assert notify_labeled.main() == 0
    assert "not watched" in capsys.readouterr().out


def test_missing_event_path_exits_cleanly(capsys, monkeypatch):
    # Regression: Path("") is the current directory, and a directory passes .exists(),
    # so an unset GITHUB_EVENT_PATH used to try to read "." and raise IsADirectoryError.
    monkeypatch.delenv("GITHUB_EVENT_PATH", raising=False)
    monkeypatch.setattr(sys, "argv", ["x", "--dry-run"])
    assert notify_labeled.main() == 0
    assert "no event payload" in capsys.readouterr().out


def test_directory_as_event_path_exits_cleanly(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--event", str(tmp_path), "--dry-run"])
    assert notify_labeled.main() == 0
    assert "not a file" in capsys.readouterr().out
