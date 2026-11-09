"""Self-routing (``trida_mode: "auto"``) on the tiny multimodal checkpoint with a random pointer head.

Pinned: the routing state is a stable prefix across turns (tools first, oldest turn first) so the Decider's
prefix cache hits; truncation drops the oldest turns at fixed cut points; malformed tool entries are skipped;
the threshold policy (P >= threshold -> causal); any failure to decide falls back to causal; bad server flags
fail at startup; over HTTP an auto request decodes in the mode its route names and reports the route,
per-request thresholds steer it, and bad values are a 400.
"""
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import pytest

from test_decision import REQ, checkpoint, decider  # noqa: F401
from trida_mlx import routing, server
from trida_mlx.decide_request import render
from trida_mlx.engine import Engine

CHAT = [
    {"role": "system", "content": "be brief"},
    {"role": "user", "content": "what is the weather in Seoul?"},
    {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": "{}"}}]},
    {"role": "tool", "content": "sunny, 21C"},
]
TOOLS = [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]


def test_route_state_is_a_stable_prefix():
    state = routing.route_state(CHAT, tools=TOOLS)
    assert list(state) == ["tools available", "conversation"] and state["tools available"] == "get_weather"
    assert state["conversation"] == ["system: be brief", "user: what is the weather in Seoul?",
                                     "assistant: [calls get_weather]", "tool: sunny, 21C"]
    # a later turn's state extends an earlier turn's state
    assert render(state).startswith(render(routing.route_state(CHAT[:2], tools=TOOLS)))
    parts = [{"type": "text", "text": "what is this"}, {"type": "image_url", "image_url": {"url": "x"}}]
    assert routing.route_state([{"role": "user", "content": parts}])["conversation"] == ["user: what is this [image]"]
    assert routing.route_state(prompt="2+2=") == {"request": "2+2="}


def test_route_state_skips_malformed_tools():
    msgs = [{"role": "assistant", "content": "", "tool_calls": [{"function": None}, "junk", {"function": {}}]}]
    state = routing.route_state(msgs, tools=[{"type": "function", "function": None}, None, {"function": "x"}])
    assert state == {"conversation": ["assistant: "]}


def test_fit_state_cuts_oldest_at_fixed_points():
    words = lambda s: len(s.split())  # noqa: E731 - a stand-in tokenizer
    budget, starts = 400, []
    turns = []
    for i in range(120):
        turns.append(f"user: turn {i} " + "w " * 18)
        fitted = routing.fit_state({"conversation": list(turns)}, words, budget)
        assert words(render(fitted)) <= budget and fitted["conversation"][-1] == turns[-1]
        starts.append(turns.index(fitted["conversation"][0]))
    # the first kept turn only moves at cut points: most turns keep the previous turn's prefix
    assert starts[0] == 0 and starts[-1] > 0
    assert sum(a != b for a, b in zip(starts, starts[1:])) <= len(starts) // 3


def test_prefix_cache_hits_on_the_next_turn(decider):  # noqa: F811
    routing.route(decider, routing.route_state(CHAT[:2], tools=TOOLS))
    routing.route(decider, routing.route_state(CHAT, tools=TOOLS))
    assert decider.cache_reused_tokens > 0


def test_threshold_policy(decider):  # noqa: F811
    state = routing.route_state(CHAT, tools=TOOLS)
    r = routing.route(decider, state, 0.0)
    p = r["p_careful"]
    assert r["mode"] == "causal" and 0.0 <= p <= 1.0 and r["ms"] >= 0
    assert routing.route(decider, state, p)["mode"] == "causal"          # P >= threshold -> causal
    if p < 1.0:
        assert routing.route(decider, state, min(1.0, p + 1e-3))["mode"] == "diffusion"


@pytest.mark.parametrize("error", [ValueError("decision row exceeds context length 8"), RuntimeError("metal")])
def test_any_failure_falls_back_to_causal(decider, monkeypatch, error):  # noqa: F811
    def fail(*_a, **_k):
        raise error

    monkeypatch.setattr(decider, "decide", fail)
    r = routing.route(decider, {"x": "y"}, 0.9)
    assert r["mode"] == "causal" and r["p_careful"] is None and str(error) in r["fallback"]


@pytest.mark.parametrize("flags", [["--route-threshold", "2"], ["--diffusion-block", "1"],
                                   ["--diffusion-threshold", "1.5"]])
def test_bad_server_flags_fail_at_startup(flags):
    with pytest.raises(SystemExit):
        server.main(["--model", "unused", *flags])


def test_engine_rejects_bad_diffusion_settings(checkpoint):  # noqa: F811
    with pytest.raises(ValueError):
        Engine(str(checkpoint), diffusion_block=1)


def test_http_auto(checkpoint):  # noqa: F811
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen([sys.executable, "-m", "trida_mlx.server", "--model", str(checkpoint), "--port",
                             str(port), "--prefill-chunk", "7", "--max-tokens", "8"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    base = f"http://127.0.0.1:{port}/v1"

    def chat(**extra):
        body = {"messages": CHAT, "tools": TOOLS, "temperature": 0, "trida_mode": "auto", **extra}
        req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))

    try:
        for _ in range(200):
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            try:
                urllib.request.urlopen(f"{base}/models", timeout=1)
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            pytest.fail("MLX server did not start")
        trida = chat()["usage"]["trida"]
        route, p = trida["route"], trida["route"]["p_careful"]
        assert route["threshold"] == routing.ROUTE_THRESHOLD and 0.0 <= p <= 1.0
        assert trida["mode"].startswith(route["mode"])  # decoded the way it was routed
        assert chat(route_threshold=0)["usage"]["trida"]["mode"] == "causal"
        if p < 1.0:
            assert chat(route_threshold=1)["usage"]["trida"]["mode"].startswith("diffusion")
        for bad in ({"route_threshold": 2}, {"route_threshold": "high"}, {"trida_mode": "smart"}):
            with pytest.raises(urllib.error.HTTPError) as err:
                chat(**bad)
            assert err.value.code == 400
    finally:
        proc.terminate()
        proc.communicate(timeout=10)
