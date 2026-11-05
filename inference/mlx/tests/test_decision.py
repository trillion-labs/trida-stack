"""Offline serving-contract, hidden-state, head, image and HTTP checks on random tiny weights."""

import base64
import importlib.util
import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

import tiny_vl
from trida_mlx import convert, decision, decide_request
from trida_mlx.engine import Engine

REQ = {
    "state": {"ticket": "printer jam"},
    "questions": {
        "team": {"type": "choice", "criteria": {"it": "support", "billing": "billing"}},
        "urgent": {"type": "noul"},
        "severity": {"type": "score", "criteria": ["low", "medium", "high"]},
    },
}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    path = tmp_path_factory.mktemp("decision")
    old = tiny_vl.SPECIALS
    tiny_vl.SPECIALS = old + decide_request.SPECIAL
    try:
        ckpt, _ = tiny_vl.build(path / "raw")
    finally:
        tiny_vl.SPECIALS = old
    rng = np.random.default_rng(5)
    head = {
        f"{k}.{part}": mx.array(rng.normal(0, 0.1, shape).astype(np.float32))
        for k in ("q", "k")
        for part, shape in (("weight", (16, 64)), ("bias", (16,)))
    }
    mx.save_safetensors(str(ckpt / "jev_head.safetensors"), head)
    (ckpt / "temperature.json").write_text(json.dumps({"text": 1.5, "image": 2.0}))
    return ckpt


@pytest.fixture
def decider(checkpoint):
    eng = Engine(str(checkpoint), prompt_cache=False, prefill_chunk=7)
    eng.model.set_dtype(mx.float32)
    eng.vision.set_dtype(mx.float32)
    return decision.Decider(eng)


def test_shared_contract_is_pinned():
    canonical = Path(__file__).resolve().parents[2] / "decide_request.py"
    assert Path(decide_request.__file__).read_bytes() == canonical.read_bytes()


def test_torch_head_parity_and_pt_import(tmp_path, decider):
    torch = pytest.importorskip("torch")
    spec = importlib.util.spec_from_file_location(
        "reference_head", Path(__file__).resolve().parents[2] / "jev_head.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    head = module.PointerHead(64, 16)
    head.load_state_dict({k: torch.from_numpy(np.array(v)) for k, v in decider.head.w.items()})
    torch.save({"head": head.state_dict(), "head_dim": 16}, tmp_path / "jev_head.pt")
    loaded = decision.load_head(tmp_path, 64)
    rng = np.random.default_rng(6)
    hd, ho = (
        rng.normal(size=(1, 64)).astype(np.float32),
        rng.normal(size=(3, 64)).astype(np.float32),
    )
    z = (
        head(torch.from_numpy(hd), torch.from_numpy(ho), torch.zeros(3, dtype=torch.long))
        .detach()
        .numpy()
    )
    np.testing.assert_allclose(loaded(mx.array(hd[0]), mx.array(ho)), z, atol=1e-6)
    for t in (0.7, 2.0):
        np.testing.assert_allclose(
            mx.softmax(loaded(mx.array(hd[0]), mx.array(ho)) / t),
            torch.softmax(torch.from_numpy(z) / t, -1),
            atol=1e-6,
        )


@pytest.mark.parametrize("image", [False, True])
def test_hidden_states_match_prefix_forwards(decider, image):
    req = decide_request.parse(
        {
            **REQ,
            **(
                {"image": "data:image/png;base64," + base64.b64encode(tiny_vl.png(1)).decode()}
                if image
                else {}
            ),
        }
    )
    img = decider._image(req)
    rows = decider.encode(req, img)
    rt = decider.engine.rt
    for row in rows:
        positions = row["opts"] + [row["decide"]]
        h = rt.prefill(rt.make_cache(), row["ids"], hidden_positions=positions)
        # Mirrors SGLang's one causal-prefix query at each delimiter.
        expected = mx.concatenate(
            [
                rt.prefill(rt.make_cache(), row["ids"][: i + 1], hidden_positions=[i])
                for i in positions
            ]
        )
        np.testing.assert_allclose(h, expected, atol=2e-5, rtol=1e-5)
        logits = rt.prefill(rt.make_cache(), row["ids"])
        np.testing.assert_allclose(rt._lm_head(h[-1:])[0], logits, atol=2e-5, rtol=1e-5)
    out = decider.decide(req.raw)
    assert set(out["answers"]) == {"team", "urgent", "severity"}
    assert out["answers"]["urgent"]["type"] == "noul"
    assert 0 <= out["answers"]["urgent"]["noul"] <= 1
    assert abs(sum(out["answers"]["team"]["probabilities"].values()) - 1) < 0.001
    assert 0 <= out["answers"]["severity"]["score"] <= 2


def test_reasoning_and_temperature(decider, monkeypatch):
    seen = []

    def think(request, image, max_tokens, **kwargs):
        seen.append(max_tokens)
        return {"text": "IT handles printers", "ended": "complete", "stats": {}}

    monkeypatch.setattr(decider, "think", think)
    result = decider.decide(REQ, reasoning=True, max_reasoning_tokens=32)
    assert (
        seen == [32] and result["reasoning"] == "IT handles printers" and "reasoning_ms" in result
    )
    expected = decider.decide(decide_request.with_reasoning(REQ, result["reasoning"]))
    assert result["answers"] == expected["answers"]
    req = decide_request.parse(REQ)
    logits = decider.logits(decider.encode(req, None))
    expected = {
        q.qid: decide_request.to_answer(q, mx.softmax(z / 1.5).tolist())
        for q, z in zip(req.questions, logits)
    }
    assert decider.decide(REQ)["answers"] == expected
    assert decider.decide(REQ, temperature=50)["answers"] != expected


@pytest.mark.parametrize(
    "kwargs",
    [
        {"temperature": 0},
        {"temperature": float("nan")},
        {"reasoning": "always"},
        {"max_reasoning_tokens": -1},
    ],
)
def test_bad_options(decider, kwargs):
    with pytest.raises(decide_request.RequestError):
        decider.decide(REQ, **kwargs)


def test_convert_head_and_q8_drift(checkpoint, tmp_path):
    convert.main(
        ["--model", str(checkpoint), "--out", str(tmp_path), "--bits", "8", "--group-size", "32"]
    )
    raw, q8 = decision.Decider(Engine(str(checkpoint))), decision.Decider(Engine(str(tmp_path)))
    assert q8.temperatures == raw.temperatures
    for k in raw.head.w:
        np.testing.assert_array_equal(raw.head.w[k], q8.head.w[k])
    req = decide_request.parse(REQ)
    a, b = raw.logits(raw.encode(req, None)), q8.logits(q8.encode(req, None))
    # A tiny random checkpoint regression bound, not a full-checkpoint quality claim.
    for x, y in zip(a, b):
        assert float(mx.max(mx.abs(mx.softmax(x / 1.5) - mx.softmax(y / 1.5)))) < 0.02


def test_http_decisions(checkpoint):
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "trida_mlx.server",
            "--model",
            str(checkpoint),
            "--port",
            str(port),
            "--prefill-chunk",
            "7",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    base = f"http://127.0.0.1:{port}/v1"

    def post(body):
        return urllib.request.urlopen(
            urllib.request.Request(
                f"{base}/decide",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=30,
        )

    try:
        for _ in range(200):
            if proc.poll() is not None:
                pytest.fail(proc.stderr.read().decode())
            try:
                with urllib.request.urlopen(f"{base}/models", timeout=1) as response:
                    assert json.load(response)["data"][0]["decision"] is True
                break
            except (OSError, urllib.error.URLError):
                time.sleep(0.1)
        else:
            pytest.fail("MLX server did not start")
        expected = decision.Decider(Engine(str(checkpoint), prefill_chunk=7)).decide(REQ)["answers"]
        assert json.load(post({"request": REQ}))["answers"] == expected
        image = "data:image/png;base64," + base64.b64encode(tiny_vl.png(2)).decode()
        assert "answers" in json.load(post({"request": {**REQ, "image": image}}))
        assert "reasoning" in json.load(
            post({"request": REQ, "reasoning": True, "max_reasoning_tokens": 3})
        )
        with post({"request": REQ, "reasoning": "auto", "threshold": 0, "stream": True}) as response:
            lines = [line.decode().strip() for line in response]
        events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
        assert [event["type"] for event in events] == ["direct", "result"]
        assert not events[-1]["escalated"] and "data: [DONE]" in lines
        with post({"request": REQ, "reasoning": True, "max_reasoning_tokens": 3,
                   "stream": True, "trida_mode": "self-spec"}) as response:
            lines = [line.decode().strip() for line in response]
        events = [json.loads(line[6:]) for line in lines if line.startswith("data: {")]
        kinds = [event["type"] for event in events]
        assert kinds[0] == "direct" and kinds[-1] == "result"
        assert "reasoning" in kinds and "data: [DONE]" in lines
        if "cut" in events[-1]:
            assert events[-1]["answers"] == events[0]["answers"]
        for body in (
            {},
            [],
            {"request": {}},
            {"request": REQ, "reasoning": "always"},
            {"request": {**REQ, "image": "/etc/hosts"}},
            {"request": {**REQ, "image": "https://example.com/a.png"}},
        ):
            with pytest.raises(urllib.error.HTTPError) as err:
                post(body)
            assert err.value.code == 400
    finally:
        proc.terminate()
        proc.communicate(timeout=10)


def test_missing_head_and_context(decider, monkeypatch):
    monkeypatch.setattr(decider, "context_length", 2)
    with pytest.raises(decide_request.RequestError, match="context length"):
        decider.decide(REQ)
    monkeypatch.setattr(decider, "head", None)
    with pytest.raises(decide_request.RequestError, match="no decision head"):
        decider.decide(REQ)


def test_reasoning_preserves_chat_cache(decider):
    eng = decider.engine
    eng.use_prompt_cache = True
    ids = eng.tokenizer.encode("existing chat", add_special_tokens=False)
    eng.rt.prefill(eng.cache, ids)
    current, slots = eng.cur, eng.slots
    decider.decide(REQ, reasoning=True, max_reasoning_tokens=3)
    assert eng.cur is current and eng.slots is slots and eng.use_prompt_cache
    assert eng.cache.tokens == ids


def test_comparison_detects_contract_and_probability_drift():
    from compare_decisions import differences

    assert not differences({"p": 0.5}, {"p": 0.501}, 0.002)
    assert differences({"p": 0.5}, {"p": 0.51}, 0.002)
    assert differences({"p": 0.5}, {"p": float("nan")}, 0.002)
    assert differences({"choice": "a"}, {"choice": "b"}, 1.0)
    assert differences({"p": 0.5}, {"other": 0.5}, 1.0)
