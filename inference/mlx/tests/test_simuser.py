"""simuser against a fake user-LLM server and a fake app automation API (same endpoints as Automation.swift)."""
import json
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from trida_mlx import simuser


def serve(handler):
    srv = HTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{srv.server_port}"


class FakeLLM(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def _send(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        self._send({"data": [{"id": "fake-user"}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeLLM.calls.append(body)
        if "designing a realistic test user" in body["messages"][0]["content"]:
            text = 'sure: {"persona": "a student", "goal": "translate a note", "style": "short", "first_message": "translate: hi"}'
        else:
            text = "<think>hmm</think>thanks! [DONE]"
        self._send({"choices": [{"message": {"content": text}}]})


class FakeApp(BaseHTTPRequestHandler):
    sent = []

    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        assert self.headers["Authorization"] == "Bearer tok"
        self._send(200, {"ready": True, "model": "zoro-1-mlx-q8"})

    def do_POST(self):
        assert self.headers["Authorization"] == "Bearer tok"
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if self.path == "/chats":
            return self._send(200, {"id": "C1"})
        FakeApp.sent.append(body["text"])
        self._send(200, {"messages": [
            {"role": "user", "content": body["text"]},
            {"role": "assistant", "content": "", "tool_runs": [{"name": "now", "arguments": "{}", "status": "done", "result": "10:00"}]},
            {"role": "assistant", "content": "안녕!", "check": "Check: ✓ language 0.97"}]})


def test_episode_runs_and_logs():
    tmp = Path(tempfile.mkdtemp())
    (tmp / "token").write_text("tok")
    simuser.TOKEN_FILE = tmp / "token"
    llm, app = serve(FakeLLM), serve(FakeApp)
    assert simuser.main(["--user-base", llm + "/v1", "--app", app, "--episodes", "2", "--max-turns", "4",
                         "--scenario", "translate_paste", "--out", str(tmp / "out"), "--seed", "1"]) == 0
    assert FakeApp.sent == ["translate: hi", "thanks!"] * 2          # [DONE] ends the episode after sending
    # the user LLM saw the tool activity and the reply, not its own <think>
    turn2 = FakeLLM.calls[1]["messages"]
    assert turn2[-1]["role"] == "user" and "[the assistant ran: now({}) -> done: 10:00]" in turn2[-1]["content"]
    assert turn2[-1]["content"].endswith("안녕!") and turn2[-2] == {"role": "assistant", "content": "translate: hi"}
    log = next((tmp / "out").glob("episodes-*.jsonl")).read_text().splitlines()
    rec = json.loads(log[0])
    assert rec["scenario"] == "translate_paste" and len(rec["turns"]) == 2 and rec["spec"]["goal"] == "translate a note"


def test_dotenv_and_missing_url(monkeypatch):
    tmp = Path(tempfile.mkdtemp())
    (tmp / ".env").write_text('# comment\nexport SIMUSER_BASE_URL="http://example:1/v1"\nSIMUSER_MODEL=m\nEMPTY=\n')
    for k in ("SIMUSER_BASE_URL", "SIMUSER_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("SIMUSER_MODEL", "already-set")
    simuser.load_dotenv([tmp / "missing.env", tmp / ".env"])
    import os
    assert os.environ["SIMUSER_BASE_URL"] == "http://example:1/v1" and os.environ["SIMUSER_MODEL"] == "already-set"
    monkeypatch.delenv("SIMUSER_BASE_URL")
    monkeypatch.setattr(simuser, "dotenv_paths", lambda: [])
    try:
        simuser.main(["--episodes", "1"])
        raise AssertionError("ran without a user LLM URL")
    except SystemExit as e:
        assert e.code == 2
