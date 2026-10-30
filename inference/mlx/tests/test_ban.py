"""Banned tokens (SamplingParams.banned; server tool_choice "none" bans <tool_call>): never generated,
greedy self-spec still equals greedy AR, and sampled self-spec never emits them either."""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tiny_vl  # noqa: E402
from trida_mlx.decode import SamplingParams  # noqa: E402
from trida_mlx.engine import Engine  # noqa: E402

CKPT, _ = tiny_vl.build(Path(tempfile.mkdtemp(prefix="trida_ban_")) / "raw")


def run(e, prompt, sp, mode):
    return [i for ch in e.generate_ids(prompt, max_tokens=30, sampling=sp, mode=mode) for i in ch]


def test_banned_tokens_never_appear():
    e = Engine(str(CKPT), prompt_cache=False)
    prompt = e.encode(e.render([{"role": "user", "content": "count to ten please"}], enable_thinking=False))
    free = run(e, prompt, SamplingParams(temperature=0), "causal")
    ban = tuple(sorted(set(free[:3])))
    ar = run(e, prompt, SamplingParams(temperature=0, banned=ban), "causal")
    ss = run(e, prompt, SamplingParams(temperature=0, banned=ban), "self-spec")
    assert not set(ar) & set(ban) and ar == ss, (ban, ar, ss)
    for seed in range(3):
        s = run(e, prompt, SamplingParams(temperature=1.0, top_k=50, banned=ban, seed=seed), "self-spec")
        assert not set(s) & set(ban)


def test_loop_guard_detects_repeated_blocks():
    from trida_mlx.decode import looping
    call = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    assert looping([1, 2, 3] + call * 6) == len(call)
    assert looping([1, 2, 3] + call * 4) == 0                  # not repeated enough
    assert looping(list(range(300))) == 0                      # no repetition
    assert looping([5] * 100) == 1 and looping([5] * 50) == 0  # short runs are allowed
    varied = [t for i in range(20) for t in (7, 8, 100 + i)]   # similar lines that differ: not a loop
    assert looping(varied) == 0


def test_gate_allows_a_token_only_when_the_raw_model_is_sure():
    import mlx.core as mx
    import numpy as np
    from trida_mlx.decode import apply_ban
    x = mx.array(np.log(np.array([[0.04, 0.03, 0.03] + [0.9 / 97] * 97, [0.6, 0.2, 0.2] + [0.0] * 97]) + 1e-12))
    g = np.array(apply_ban(x, SamplingParams(gate=((0, 0.2),))))
    assert np.isneginf(g[0, 0]) and np.isfinite(g[1, 0])     # 0.04 < 0.2 blocked; 0.6 allowed
    assert np.allclose(g[0, 1:], np.array(x)[0, 1:])          # nothing else changes
    e = Engine(str(CKPT), prompt_cache=False)
    prompt = e.encode(e.render([{"role": "user", "content": "count to ten please"}], enable_thinking=False))
    free = run(e, prompt, SamplingParams(temperature=0), "causal")
    t = free[0]
    ar = run(e, prompt, SamplingParams(temperature=0, gate=((t, 1.01),)), "causal")      # never sure enough
    ss = run(e, prompt, SamplingParams(temperature=0, gate=((t, 1.01),)), "self-spec")
    assert t not in ar and ar == ss
    assert run(e, prompt, SamplingParams(temperature=0, gate=((t, 1e-9),)), "self-spec") == free  # always sure


def test_gate_log_records_near_misses_on_exact_rows():
    import mlx.core as mx
    import numpy as np
    from trida_mlx.decode import _gate_note
    sp = SamplingParams(gate=((0, 0.5),), gate_log=[])
    rows = mx.array(np.log(np.array([[0.3, 0.7] + [1e-9] * 8, [0.01, 0.99] + [1e-9] * 8, [0.9, 0.1] + [1e-9] * 8])))
    _gate_note(sp, rows)
    assert [round(p, 2) for p, _ in sp.gate_log] == [0.3]      # 0.01 is below the note floor, 0.9 passes
    e = Engine(str(CKPT), prompt_cache=False)
    prompt = e.encode(e.render([{"role": "user", "content": "count to ten please"}], enable_thinking=False))
    free = run(e, prompt, SamplingParams(temperature=0), "causal")
    for mode in ("causal", "self-spec"):
        spm = SamplingParams(temperature=0, gate=((free[0], 1.01),), gate_log=[])
        run(e, prompt, spm, mode)
        assert spm.gate_log and all(p >= 0.05 for p, _ in spm.gate_log)


def test_ngram_loop_detector():
    import random
    from trida_mlx.decode import LOOP_WINDOW, NgramLoop
    rng = random.Random(0)
    det, hist = NgramLoop(), []
    for _ in range(4000):                       # varied text: never a loop
        hist.append(rng.randrange(5000))
        assert not det.add(hist)
    block = [rng.randrange(5000) for _ in range(400)]   # a 400-token derivation redone over and over
    det, hist, fired = NgramLoop(), [], None            # (longer than the exact-repeat guard's 256-token period)
    for i, t in enumerate((block + [rng.randrange(5000) for _ in range(30)]) * 8):
        hist.append(t)
        if det.add(hist):
            fired = i
            break
    assert fired is not None and fired < 3 * LOOP_WINDOW


def test_loop_while_thinking_closes_the_thinking_and_answers():
    import json
    import socket
    import threading
    import time
    import urllib.request
    from trida_mlx import server
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=server.main, args=(["--model", str(CKPT), "--port", str(port), "--loop-guard"],),
                     daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"
    for _ in range(300):
        try:
            card = json.loads(urllib.request.urlopen(base + "/models", timeout=1).read())["data"][0]
            break
        except Exception:
            time.sleep(0.1)
    body = {"messages": [{"role": "user", "content": "hi"}], "max_tokens": 600, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": True}}
    r = json.loads(urllib.request.urlopen(urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})).read())
    assert card["loop_guard"] is True and card["tool_call_gate"] == 0.0
    t = r["usage"]["trida"]
    # the tiny random model emits one token forever: the guard stops the thinking, then the answer
    assert t["finish_reason"] == "repetition" and t.get("loop_stopped_thinking") is True
    assert "going in circles" in (r["choices"][0]["message"].get("reasoning_content") or "")
