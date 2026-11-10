"""Snapshot, prefix reuse, auto reasoning, and torch-free head regression tests from #69."""
import json
import pickle
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

from test_decision import REQ, checkpoint, decider  # noqa: F401
from trida_mlx import decision, decide_request


@pytest.mark.parametrize("image", [False, True])
def test_branching_and_cached_states_match_full_rows(decider, image):
    from PIL import Image
    request = {**REQ, "image": Image.new("RGB", (32, 32), "red")} if image else REQ
    req = decide_request.parse(request)
    img = decider._image(req)
    rows = decider.encode(req, img)
    expected = []
    for row in rows:
        h = decider.engine.rt.prefill(decider.engine.rt.make_cache(), row["ids"],
                                    hidden_positions=row["opts"] + [row["decide"]])
        expected.append(decider.head(h[-1], h[:-1]))
    for reuse in (False, True):
        actual = decider.logits(rows)
        for got, want in zip(actual, expected):
            np.testing.assert_allclose(got, want, atol=2e-5, rtol=1e-5)
        assert bool(decider.cache_reused_tokens) == reuse
    assert len(decider.state_cache) == 1
    # Changing only the questions shares the state; another state/image must be scored cold.
    decider.decide({**request, "questions": {"new": {"type": "noul"}}})
    assert decider.cache_reused_tokens > 0
    changed = {**request, "state": "another ticket"}
    cached = decider.decide(changed)["answers"]
    assert decider.cache_reused_tokens == 0
    decider.clear_cache()
    assert decider.decide(changed)["answers"] == cached


def test_state_extension_and_cache_budget(decider):
    decider.decide({**REQ, "state": "printer"})
    decider.decide({**REQ, "state": "printer jam"})
    assert decider.cache_reused_tokens > 0
    decider.clear_cache()
    decider.cache_bytes = 1
    decider.decide(REQ)
    assert not decider.state_cache
    decider.cache_bytes = 0
    decider.decide(REQ)
    decider.decide(REQ)
    assert decider.cache_reused_tokens == 0 and not decider.state_cache


def test_a_cycle_of_more_states_than_fit_still_hits(decider):
    # Plain LRU evicts exactly the state needed next in a cycle (a file list searched again batch by batch), so
    # every request is cold. Entries that were never reused go first, newest first, so most of the cycle stays.
    states = [f"ticket {c}" for c in "ABCD"]
    decider.clear_cache()
    decider.decide({**REQ, "state": states[0]})
    one = sum(c.nbytes() for c in decider.state_cache.values())
    decider.clear_cache()
    decider.cache_bytes = int(3.5 * one)                       # room for 3 of the 4
    hits = []
    for _ in range(3):
        for s in states:
            decider.decide({**REQ, "state": s})
            hits.append(decider.cache_reused_tokens > 0)
    assert sum(hits[4:]) >= 4, hits                            # passes 2 and 3: at least 2 of 4 hit each
    assert sum(c.nbytes() for c in decider.state_cache.values()) <= decider.cache_bytes


def test_auto_reasoning_threshold_and_cutoff(decider, monkeypatch):
    events, calls = [], []
    monkeypatch.setattr(decider, "think", lambda *a, **k: calls.append(k) or {
        "text": "unfinished", "ended": "budget", "stats": {"finish_reason": "length"}})
    sure = decider.decide(REQ, reasoning="auto", threshold=0,
                          on_event=lambda kind, payload: events.append(kind))
    assert not sure["escalated"] and not calls and events == ["direct", "result"]
    uncertain = decider.decide(REQ, reasoning="auto", threshold=1)
    assert calls and uncertain["escalated"] and "cut" in uncertain
    assert uncertain["answers"] == uncertain["direct"]["answers"]
    assert "reasoned" not in uncertain
    monkeypatch.setattr(decider, "think", lambda *a, **k: {
        "text": "IT handles printers", "ended": "complete", "stats": {}})
    done = decider.decide(REQ, reasoning="auto", threshold=1)
    assert done["escalated"] and "cut" not in done
    assert done["answers"] == done["reasoned"]["answers"]


def test_long_reasoning_keeps_conclusion(decider):
    text = " ".join(f"step{i} checks the jam" for i in range(3000)) + " so: facilities"
    fitted = decider.fit_reasoning(REQ, text)
    assert fitted.startswith("…") and fitted.endswith("so: facilities")
    assert len(fitted) < len(text)
    req = decide_request.parse(decide_request.with_reasoning(REQ, fitted))
    state = req.encode_rows(decider.engine.tokenizer)[0]["ids"]
    q_id = decider.engine.tokenizer.convert_tokens_to_ids(decide_request.SPECIAL[1])
    assert state.index(q_id) <= decide_request.MAX_STATE
    assert decider.fit_reasoning(REQ, "short reasoning") == "short reasoning"


def test_torch_free_reader_fixture_and_layout_validation():
    fixtures = Path(__file__).parent / "fixtures"
    got = decision.load_torch_file(fixtures / "jev_head_tiny.pt")
    expected = json.loads((fixtures / "jev_head_tiny.json").read_text())
    assert got["head_dim"] == expected["head_dim"]
    for name, weight in expected["head"].items():
        np.testing.assert_allclose(got["head"][name], weight, atol=1e-6)
    np.testing.assert_array_equal(got["bf16"], expected["bf16"])
    with pytest.raises(ValueError, match="exceeds"):
        decision._rebuild_tensor(np.zeros(2), 0, (3,), (1,))
    with pytest.raises(ValueError, match="invalid"):
        decision._rebuild_tensor(np.zeros(2), 0, (2,), (-1,))


def test_torch_free_reader_rejects_objects(tmp_path):
    import zipfile
    with zipfile.ZipFile(tmp_path / "bad.pt", "w") as archive:
        archive.writestr("data.pkl", pickle.dumps(Path("bad")))
    with pytest.raises(pickle.UnpicklingError, match="unsupported object"):
        decision.load_torch_file(tmp_path / "bad.pt")


@pytest.mark.parametrize("kwargs", [
    {"threshold": -0.1}, {"threshold": 1.01}, {"threshold": True},
    {"threshold": float("nan")}, {"reasoning": 1}, {"mode": "bogus"},
])
def test_extension_options_are_validated(decider, kwargs):
    with pytest.raises(decide_request.RequestError):
        decider.decide(REQ, **kwargs)


def test_model_card_temperature_is_explicit(checkpoint):
    from trida_mlx.engine import Engine
    decider = decision.Decider(Engine(str(checkpoint)), text_temperature=1.52)
    assert decider.temperatures["text"] == 1.52
    assert decider.temperatures["image"] == 2.0


def test_cancelled_reasoning_restores_chat_cache(decider):
    from trida_mlx.server import _Cancelled
    eng = decider.engine
    ids = eng.tokenizer.encode("existing chat", add_special_tokens=False)
    eng.rt.prefill(eng.cache, ids)
    current, slots = eng.cur, eng.slots

    def cancel(kind, payload):
        if kind == "piece":
            raise _Cancelled()

    with pytest.raises(_Cancelled):
        decider.decide(REQ, reasoning=True, max_reasoning_tokens=3, on_event=cancel)
    assert eng.cur is current and eng.slots is slots and eng.cache.tokens == ids
    # Cancellation does not leave the next request on a question's branch.
    assert "answers" in decider.decide(REQ)


@pytest.mark.parametrize("calibration,override,expected", [
    (None, None, {"text": 1.52, "image": 1.0}),
    ({"image": 2.0}, None, {"text": 1.52, "image": 2.0}),
    ({"text": 1.25, "image": 2.0}, None, {"text": 1.25, "image": 2.0}),
    ({"text": 1.25, "image": 2.0}, 0.9, {"text": 0.9, "image": 2.0}),
])
def test_server_temperature_fallback_and_overrides(checkpoint, tmp_path, calibration, override, expected):
    import shutil
    from trida_mlx.engine import Engine
    path = tmp_path / "Zoro-1"
    shutil.copytree(checkpoint, path)
    (path / "temperature.json").unlink()
    if calibration is not None:
        (path / "temperature.json").write_text(json.dumps(calibration))
    loaded = decision.Decider.load(Engine(str(path)), text_temperature=override)
    assert loaded.temperatures == expected
    # The per-request override still takes precedence over checkpoint/server settings.
    assert loaded.decide(REQ, temperature=50)["answers"] != loaded.decide(REQ)["answers"]


@pytest.mark.parametrize("reasoning", [True, "auto"])
@pytest.mark.parametrize("mode", [None, "causal", "self-spec"])
def test_reasoning_mode_and_progress_reason(decider, monkeypatch, reasoning, mode):
    calls, events = [], []

    def think(*args, **kwargs):
        calls.append(kwargs["mode"])
        assert events[0][0] == "direct" and events[0][1]["reason"]
        return {"text": "IT handles printers", "ended": "complete", "stats": {}}

    monkeypatch.setattr(decider, "think", think)
    result = decider.decide(REQ, reasoning=reasoning, mode=mode, threshold=1,
                            on_event=lambda kind, payload: events.append((kind, payload)))
    assert calls == [mode or "self-spec"]
    assert events[0][1]["reason"] == result["reason"]
    assert events[-1][0] == "result" and result["escalated"]
    if reasoning == "auto":
        assert "below 1" in result["reason"] and result["lowest"]["question"] in result["reason"]
    else:
        assert result["reason"] == "reasoning requested"


def test_loader_refuses_oversized_views_and_odd_zips(tmp_path):
    import zipfile
    # a zero stride passes the bounds check but would .copy() into a huge array (a hostile jev_head.pt)
    with pytest.raises(ValueError, match="larger than its storage"):
        decision._rebuild_tensor(np.zeros(2), 0, (1 << 20, 1 << 20), (0, 0))
    with zipfile.ZipFile(tmp_path / "empty.pt", "w") as z:
        z.writestr("archive/version", "3")
    with pytest.raises(ValueError, match="data.pkl"):
        decision.load_torch_file(tmp_path / "empty.pt")
