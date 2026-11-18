"""Failure isolation, slow clients and shared completion: no GPU or fork import."""
import ast
import asyncio
from contextlib import aclosing
import logging
import os
from pathlib import Path
import sys
import textwrap
from types import ModuleType, SimpleNamespace

import pytest
import torch

from decide_request import parse, reasoning_complete
from self_spec_decide_engine import SelfSpecDecideEngine
from test_self_spec_decide_cpu import engine, REQUEST


@pytest.mark.parametrize("text,finish,stop,complete", [
    ("done</think>", "stop", None, True), ("done", "stop", 50, True),
    ("done", "stop", "</think>", True), ("unclosed", "stop", 51, False),
    ("done</think>", "abort", None, False), ("done</think>", "length", None, False),
    ("done</think>", "unknown", None, False), ("done", "stop", None, False),
])
def test_completion_rule_accepts_native_stripped_stop_tokens(text, finish, stop, complete):
    assert reasoning_complete(text, finish, end_think_id=50, stop_reason=stop) is complete
    assert reasoning_complete(text, {"type": finish, "matched": stop}, end_think_id=50) is complete


def test_unread_stream_does_not_hold_native_admission():
    async def run():
        e = engine()
        e.loop = asyncio.get_running_loop()
        release, finished = asyncio.Event(), asyncio.Event()
        async def generate(**kw):
            if not kw.get("stream"):
                return {"text": "next", "meta_info": {}}
            async def chunks():
                yield {"text": "first", "meta_info": {}}
                await release.wait()
                yield {"text": "first done", "meta_info": {}}
                finished.set()
            return chunks()
        e.engine = SimpleNamespace(async_generate=generate,
            tokenizer_manager=SimpleNamespace(abort_request=lambda rid: release.set()))
        async with aclosing(e._astream([100], {"max_new_tokens": 8})) as stream:
            assert (await anext(stream))[0] == "first"
            release.set()
            await finished.wait()
            # Never read the remaining response; a new native request must still enter.
            out = await asyncio.wait_for(e._generate([100], {"max_new_tokens": 1}), 1)
            assert out["text"] == "next"
            assert not e._admission.locked()
    asyncio.run(run())


def test_abandoned_nonstreaming_chat_is_cancelled(monkeypatch):
    from fastapi.testclient import TestClient
    import serve_decide
    e = engine()
    e.tok.ids = {**e.tok.ids, "<|im_end|>": 51}
    entered, cleaned = [], []
    async def chat(*args, **kwargs):
        entered.append(True)
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.append(True)
    e.achat = chat
    original = serve_decide._while_connected
    async def disconnected(work, _request):
        class Disconnected:
            async def is_disconnected(self):
                return bool(entered)
        return await original(work, Disconnected())
    monkeypatch.setattr(serve_decide, "_while_connected", disconnected)
    response = TestClient(serve_decide.create_app(e)).post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 499 and entered == cleaned == [True]


@pytest.mark.parametrize("stuck", [False, True])
def test_readout_deadline_drains_or_marks_engine_unavailable(stuck):
    async def run():
        e = engine()
        e.readout_timeout = 0.005
        clock, release = [0.0], asyncio.Event()
        e.loop = SimpleNamespace(time=lambda: clock[0]) if stuck else asyncio.get_running_loop()
        async def generate(**kw):
            await release.wait()
            return {"meta_info": {"hidden_states": [[[1., 2.]]]}}
        def abort(rid):
            if stuck:
                clock[0] = 31.0
            else:
                release.set()
        e.engine = SimpleNamespace(async_generate=generate, tokenizer_manager=SimpleNamespace(abort_request=abort))
        with pytest.raises(RuntimeError if stuck else asyncio.TimeoutError):
            await e.alogits(parse(REQUEST).encode(e.tok))
        assert not e._admission.locked() and e._failed is stuck
        if not stuck:
            assert await e.alogits(parse(REQUEST).encode(e.tok))
    asyncio.run(run())


@pytest.mark.parametrize("version", [None, 1, 2])
def test_backend_checks_readout_overlay_version(monkeypatch, version):
    import serve
    calls = []
    for name in ("sglang", "sglang.srt", "sglang.srt.layers", "sglang.srt.layers.logits_processor"):
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    sys.modules["sglang"].Engine = lambda **kw: calls.append(kw) or "engine"
    sys.modules["sglang.srt.layers.logits_processor"].LogitsProcessor = SimpleNamespace(
        trida_decision_readout_version=version)
    monkeypatch.setattr(serve, "for_sglang", lambda ckpt: ckpt)
    e = engine()
    e.config = Path("settings.yaml")
    if version != 2:
        with pytest.raises(RuntimeError, match=sys.executable):
            e._build_engine("checkpoint", 0.5, 8192, {})
        assert not calls
    else:
        assert e._build_engine("checkpoint", 0.5, 8192, {}) == "engine"
        assert calls[0]["max_running_requests"] == 1


def test_patch_mismatch_aborts_only_bad_readout():
    patch = (Path(__file__).parent / "sglang/decision_readout.patch").read_text()
    lines = patch.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("+    def _capture_trida_decision_prefill"))
    added = []
    for line in lines[start:]:
        if not line.startswith("+"):
            break
        added.append(line[1:])
    namespace = {"logger": logging.getLogger(__name__), "os": os}
    exec(compile(ast.parse(textwrap.dedent("\n".join(added))), "readout_patch", "exec"), namespace)
    def request(rid, last):
        req = SimpleNamespace(rid=rid, return_hidden_states=True, hidden_states=[], origin_input_ids=[1, last], errors=[])
        req.set_finish_with_abort = req.errors.append
        return req
    good, bad = request("good", 2), request("bad", 99)
    output = SimpleNamespace(hidden_states=torch.tensor([[1., 2.], [3., 4.]]),
        trida_decision_prefill=True, trida_decision_end_ids=[2, 2], trida_decision_seq_lens=[2, 2])
    namespace["_capture_trida_decision_prefill"](None, SimpleNamespace(reqs=[good, bad]), SimpleNamespace(logits_output=output))
    assert good.hidden_states == [[[1., 2.]]] and good.errors == []
    assert bad.hidden_states == [] and "wrong prompt endpoint" in bad.errors[0]
    assert "SPDX-License-Identifier: Apache-2.0" in patch


def test_streaming_reasoning_hides_split_closing_tag():
    from decide_engine import DecideEngine
    async def run():
        e = engine(DecideEngine)
        async def events(*args, **kwargs):
            yield "answer</th"
            yield "ink>"
            yield {"finish_reason": "stop"}
        e._athink_events = events
        assert "".join([part async for part in e.athink_stream(REQUEST)]) == "answer"
    asyncio.run(run())


def test_native_stop_reason_is_preserved_when_reasoning_is_incomplete():
    async def run():
        e = engine()
        async def generate(*args, **kwargs):
            return {"text": "unclosed", "meta_info": {"finish_reason": {"type": "stop", "matched": 51}}}
        async def logits(enc, image=None):
            return [torch.tensor([0., 1.])]
        e._generate, e.alogits = generate, logits
        out = await e.adecide(REQUEST, reasoning=True)
        assert out["cut"] == "length" and out["answers_source"] == "direct"
        assert out["reasoning_stats"]["finish_reason"] == "stop"
        assert out["reasoning_stats"]["reasoning_complete"] is False
        assert out["reasoning_stats"]["sampling"] == e.sampling
    asyncio.run(run())


def test_capabilities_describe_selfspec_constraints():
    from fastapi.testclient import TestClient
    from serve_decide import create_app
    e = engine()
    client = TestClient(create_app(e))
    expected = {"backend": "self-spec", "images": False, "auto_reasoning": True,
                "serial": True, "reasoning_sampling": e.sampling}
    assert client.get("/health").json()["capabilities"] == expected
    assert client.get("/v1/models").json()["data"][0]["capabilities"] == expected
