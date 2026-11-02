"""Typed decisions (decision.py) on a tiny random multimodal checkpoint with a random pointer head.

Pinned: the vendored jev.py equals inference/jev.py; the shared-state branching gives the same
logits as running every question's full row from scratch (text and image states); answers are
well-formed probabilities and the temperature divides the logits; escalation runs the reasoning
only when the direct answer is unsure, never decides on cut-off reasoning, and decides again with
the reasoning in the state; jev_head.pt reads without torch; the server's /v1/decide round trip,
plain and streamed.
"""
import base64
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

import tiny_vl  # noqa: E402
from trida_mlx import decision, jev  # noqa: E402
from trida_mlx.engine import Engine  # noqa: E402

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp(prefix="trida_decide_"))
# the tiny tokenizer needs jev's delimiter tokens too (they fit in its 300-token vocab)
tiny_vl.SPECIALS = tiny_vl.SPECIALS + [t for t in jev.SPECIAL if t not in tiny_vl.SPECIALS]
CKPT, IDS = tiny_vl.build(TMP / "raw")
D = tiny_vl.TINY["hidden_size"]
rng = np.random.default_rng(0)
HEAD = {"q.weight": rng.normal(0, 0.5, (16, D)), "q.bias": rng.normal(0, 0.1, 16),
        "k.weight": rng.normal(0, 0.5, (16, D)), "k.bias": rng.normal(0, 0.1, 16)}
mx.save_safetensors(str(CKPT / "jev_head.safetensors"), {k: mx.array(v.astype(np.float32)) for k, v in HEAD.items()})

REQ = {"state": {"ticket": "the printer on floor 3 is jammed again", "customer": "acme"},
       "questions": {"team": {"type": "choice", "instructions": "Which team handles it?",
                              "criteria": {"it": "IT support", "billing": "Billing", "facilities": "Facilities"}},
                     "urgent": {"type": "noul", "instructions": "Is it urgent?"},
                     "severity": {"type": "score", "instructions": "How severe?", "criteria": ["low", "medium", "high"]}}}

ENGINE = Engine(str(CKPT), prompt_cache=False)
ENGINE.model.set_dtype(mx.float32)   # exact comparisons (bf16 matmuls differ with the call split)
ENGINE.vision.set_dtype(mx.float32)
DEC = decision.Decider.load(ENGINE)


def full_row_logits(dec, rows):
    """Reference: each question's whole row from an empty cache, no shared-state snapshot."""
    out = []
    for r in rows:
        cache = dec.engine.rt.make_cache()
        h = dec.engine.rt.prefill(cache, r["ids"], hidden_positions=r["opts"] + [r["decide"]])
        out.append(np.array(dec.head(h[-1], h[:-1])))
    return out


def test_vendored_jev_matches_inference():
    ref = HERE.parent.parent / "jev.py"
    if not ref.is_file():
        return  # installed without the rest of trida-stack
    ours = (HERE.parent / "trida_mlx" / "jev.py").read_text()
    body = ours.split("unchanged ----\n", 1)[1]
    assert body == ref.read_text(), "trida_mlx/jev.py drifted from inference/jev.py; re-vendor it"


def test_branching_equals_full_rows_text_and_image():
    for image in (None, tiny_vl.png(3)):
        qs, rows, state_len = DEC.encode(REQ, image)
        assert len(rows) == 3 and all(r["ids"][:state_len] == rows[0]["ids"][:state_len] for r in rows)
        got, want = DEC.logits(rows, state_len), full_row_logits(DEC, rows)
        for g, w in zip(got, want):
            assert np.allclose(g, w, atol=1e-4), (g, w)
        if image is not None:
            assert any(t < 0 for t in rows[0]["ids"][:state_len])  # image tokens in the state


def test_answers_and_temperature():
    out = DEC.decide(REQ)
    a = out["answers"]
    assert set(a) == {"team", "urgent", "severity"}
    assert a["team"]["choice"] in ("it", "billing", "facilities")
    assert abs(sum(a["team"]["probabilities"].values()) - 1) < 1e-3
    assert 0 <= a["urgent"]["noul"] <= 1 and 0 <= a["severity"]["score"] <= 2
    assert out["temperature"] == decision.DEFAULT_TEXT_TEMPERATURE
    hot, cold = DEC.decide(REQ, temperature=50.0), DEC.decide(REQ, temperature=0.05)
    spread = lambda o: max(o["answers"]["team"]["probabilities"].values())  # noqa: E731
    assert spread(hot) < spread(out) < spread(cold)


def test_escalation_paths():
    never = DEC.decide_or_reason(REQ, reasoning="never")
    assert not never["escalated"] and never["answers"] == never["direct"]["answers"]
    sure = DEC.decide_or_reason(REQ, threshold=0.0)          # nothing is below 0: no reasoning
    assert not sure["escalated"]
    events = []
    cut = DEC.decide_or_reason(REQ, threshold=1.01, max_reasoning_tokens=6, on_event=lambda k, p: events.append(k))
    assert cut["escalated"] and events[0] == "direct" and events[-1] == "result"
    if "cut" in cut:   # the usual case: 6 tokens of a random model never close </think>
        assert "reasoned" not in cut and cut["answers"] == cut["direct"]["answers"]
    else:              # it happened to stop: then it decided again
        assert cut["answers"] == cut["reasoned"]["answers"]
    # a finished reasoning: decide again with it in the state
    real = DEC.think
    DEC.think = lambda *a, **k: {"text": "the jam is a facilities matter", "ended": "complete", "stats": {}}
    try:
        done = DEC.decide_or_reason(REQ, threshold=1.01)
    finally:
        DEC.think = real
    assert done["escalated"] and done["answers"] == done["reasoned"]["answers"]
    rq = decision.with_reasoning(REQ, "x")
    assert rq["state"]["reasoning"] == "x" and rq["state"]["ticket"] == REQ["state"]["ticket"]


def test_bad_requests():
    for bad in ({"questions": {}}, {"state": "s", "questions": {}},
                {"state": "s", "questions": {"q": {"type": "pick", "criteria": {}}}},
                {"state": "s", "questions": {"q": {"type": "choice", "criteria": {}}}}):
        try:
            DEC.decide(bad)
        except decision.DecisionError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_torch_file_reader():
    pt = HERE / "fixtures" / "jev_head_tiny.pt"
    expect = HERE / "fixtures" / "jev_head_tiny.json"
    if not pt.is_file():
        return
    saved = decision.load_torch_file(pt)
    want = json.loads(expect.read_text())
    assert saved["head_dim"] == want["head_dim"]
    for k, v in want["head"].items():
        assert np.allclose(saved["head"][k], np.array(v, np.float32), atol=1e-6), k
    # bf16 tensors are widened to float32 exactly
    assert np.allclose(saved["bf16"], np.array(want["bf16"], np.float32))


def test_server_decide_round_trip():
    from trida_mlx import server
    s = socket.socket(); s.bind(("127.0.0.1", 0)); port = s.getsockname()[1]; s.close()
    threading.Thread(target=server.main, args=(["--model", str(CKPT), "--port", str(port)],), daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"
    for _ in range(300):
        try:
            card = json.loads(urllib.request.urlopen(f"{base}/models", timeout=1).read())["data"][0]
            break
        except Exception:
            time.sleep(0.1)
    assert card["decision"] is True

    def post(body):
        return urllib.request.urlopen(urllib.request.Request(f"{base}/decide", data=json.dumps(body).encode(),
                                                             headers={"Content-Type": "application/json"}))
    img = "data:image/png;base64," + base64.b64encode(tiny_vl.png(5)).decode()
    r = json.loads(post({**REQ, "image": img, "reasoning": "never"}).read())
    assert set(r["answers"]) == {"team", "urgent", "severity"} and r["direct"]["temperature"] == 1.0
    lines = [ln.decode().strip() for ln in post({**REQ, "threshold": 1.01, "max_reasoning_tokens": 5, "stream": True})]
    events = [json.loads(ln[6:]) for ln in lines if ln.startswith("data: {")]
    kinds = [e["type"] for e in events]
    assert kinds[0] == "direct" and kinds[-1] == "result" and events[-1]["escalated"], kinds
    assert lines[-2] == "data: [DONE]" or "data: [DONE]" in lines
    try:
        post({"state": "s", "questions": {"q": {"type": "nope"}}})
        raise AssertionError("bad request accepted")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_long_reasoning_is_trimmed_to_fit_the_state():
    long = " ".join(f"step{i} checks the jam" for i in range(3000)) + " so: facilities"
    fitted = DEC.fit_reasoning(REQ, long)
    assert fitted != long and fitted.startswith("…") and fitted.endswith("so: facilities")
    used, room = DEC._state_room(decision.with_reasoning(REQ, fitted), None)
    assert used <= room
    assert DEC.fit_reasoning(REQ, "short reasoning") == "short reasoning"
    real = DEC.think
    DEC.think = lambda *a, **k: {"text": long, "ended": "complete", "stats": {}}
    try:
        out = DEC.decide_or_reason(REQ, threshold=1.01)
    finally:
        DEC.think = real
    assert out["escalated"] and "reasoning_trimmed" in out and out["answers"] == out["reasoned"]["answers"]
