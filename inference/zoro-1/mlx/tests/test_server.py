"""Server flags and request hygiene from the #69 review, over HTTP against a subprocess server."""
import json
import os
import subprocess
import sys
import time
import urllib.error

import pytest

from serve import post, server
from test_decision import REQ, checkpoint  # noqa: F401

CHAT = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 8, "temperature": 0}


def test_defaults_and_memory_report(checkpoint):
    with server(checkpoint, "--mlx-cache-limit-mb", 64) as (base, card):
        post(base + "/chat/completions", CHAT)
        with __import__("urllib.request").request.urlopen(base + "/models") as r:
            after = json.load(r)["data"][0]
    # 64K unless asked: every client (evals, Hermes) gets it, and 128K costs about 4 GB of KV on a small Mac
    assert card["context_length"] == 65536
    assert card["mlx_cache_limit_mb"] == 64
    mem = after["memory_mb"]
    assert set(mem) == {"active", "cache", "peak"} and mem["peak"] > 0


@pytest.mark.parametrize("gate", ["1.5", "-0.1", "nan"])
def test_bad_tool_call_gate_is_refused(checkpoint, gate):
    p = subprocess.run([sys.executable, "-m", "trida_mlx.server", "--model", str(checkpoint), "--port", "1",
                        "--tool-call-gate", gate], capture_output=True, text=True, timeout=60)
    assert p.returncode == 2 and "tool-call-gate" in p.stderr


def test_client_cannot_set_internal_fields(checkpoint):
    # _gate / _banned_ids / _in_think are set by the server itself; a client sending them got a 500 (bad _gate)
    # or a new 1 MB bias array cached for good (each distinct _banned_ids)
    with server(checkpoint) as (base, _):
        r = post(base + "/chat/completions", {**CHAT, "_gate": "x", "_banned_ids": list(range(50)), "_in_think": 1})
        assert r["choices"][0]["message"]["content"] is not None


def test_bad_tool_call_min_prob_is_a_400(checkpoint):
    tools = [{"type": "function", "function": {"name": "now", "parameters": {"type": "object", "properties": {}}}}]
    with server(checkpoint) as (base, _):
        with pytest.raises(urllib.error.HTTPError) as e:
            post(base + "/chat/completions", {**CHAT, "tools": tools, "tool_call_min_prob": 2})
        assert e.value.code == 400


def test_logging_failure_never_breaks_a_request(checkpoint, tmp_path):
    blocker = tmp_path / "logs"
    blocker.write_text("a file where the log directory should be")
    with server(checkpoint, "--log-requests", blocker) as (base, _):
        assert post(base + "/chat/completions", CHAT)["choices"]
        assert "answers" in post(base + "/decide", {"request": REQ})


def test_old_request_logs_are_pruned(checkpoint, tmp_path):
    old, new, other = tmp_path / "20200101-000000-aaaaaa.json", tmp_path / "20990101-000000-bbbbbb.json", tmp_path / "notes.txt"
    for f in (old, new, other):
        f.write_text("{}")
    os.utime(old, (time.time() - 30 * 86400,) * 2)
    with server(checkpoint, "--log-requests", tmp_path, "--log-requests-keep-days", 7):
        pass
    assert not old.exists() and new.exists() and other.exists()
