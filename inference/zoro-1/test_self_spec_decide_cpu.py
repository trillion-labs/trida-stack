"""Decision policy, serial admission, and cancellation without SGLang or a GPU."""
import asyncio
from contextlib import aclosing
import re
import threading
from types import SimpleNamespace

import pytest
import torch

from decide_engine import PointerHead
from decide_request import MAX_STATE, RequestError, parse, with_reasoning
from self_spec_decide_engine import SelfSpecDecideEngine
from test_decide_cpu import CharTok

REQUEST = {"state": "A record", "questions": {"route": {"type": "choice", "criteria": {"a": "A", "b": "B"}}}}


class Tok(CharTok):
    def apply_chat_template(self, messages, tokenize=False, **kw):
        return [100] * 8 if tokenize else "a prompt"


def engine():
    e = SelfSpecDecideEngine.__new__(SelfSpecDecideEngine)
    e.tok, e.proc, e.ckpt = Tok(), None, "/ckpt/model"
    e.text_T = e.image_T = 1.0
    e.context_length, e.image_pad = 16384, 7
    e._stops, e.reasoning_temperature = [50, 51], 1.0
    e._admission, e._lock, e._failed = asyncio.Lock(), threading.Lock(), False
    e.sampling = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
    e.head = PointerHead(2, 2)
    return e


def test_auto_skips_generation_and_reasoned_answers_replace_direct():
    async def run():
        e = engine()
        calls = []
        async def logits(enc, image=None):
            calls.append("score")
            return [torch.log(torch.tensor([0.1, 0.9]))]
        async def generate(ids, params, **kw):
            calls.append("think")
            assert params["max_new_tokens"] == 32
            return {"text": "Checked the record.</think>", "meta_info": {"finish_reason": {"type": "stop", "matched": 50}}}
        e.alogits, e._generate = logits, generate
        out = await e.adecide(REQUEST, reasoning="auto", threshold=0.7)
        assert not out["escalated"] and calls == ["score"]
        calls.clear()
        out = await e.adecide(REQUEST, reasoning="auto", threshold=1.0, max_reasoning_tokens=32)
        assert out["escalated"] and out["answers"] == out["reasoned"]["answers"]
        assert out["lowest"]["top_probability"] == 0.9
        assert calls == ["score", "think", "score"]
    asyncio.run(run())


@pytest.mark.parametrize("text,finish", [("Still thinking", {"type": "length"}), ("", {"type": "stop", "matched": 50}),
                                         ("Unclosed", {"type": "stop", "matched": 51})])
def test_unfinished_reasoning_uses_direct(text, finish):
    async def run():
        e = engine()
        scores = []
        async def logits(enc, image=None):
            scores.append(1)
            return [torch.tensor([0.0, 1.0])]
        async def generate(ids, params, **kw):
            return {"text": text, "meta_info": {"finish_reason": finish}}
        e.alogits, e._generate = logits, generate
        result = await e.adecide(REQUEST, reasoning=True)
        assert result["answers"] == result["direct"]["answers"] and "cut" in result
        assert len(scores) == 1 and "reasoned" not in result
    asyncio.run(run())


@pytest.mark.parametrize("reasoning", [True, "auto"])
@pytest.mark.parametrize("budget", [None, 32])
@pytest.mark.parametrize("overflow", [0, 1])
def test_full_reasoning_context_returns_direct_over_http(reasoning, budget, overflow):
    from fastapi.testclient import TestClient
    from serve_decide import create_app
    from decide_request import prompt

    class PromptTok(Tok):
        def apply_chat_template(self, messages, **kw):
            return messages[0]["content"]

    e = engine()
    e.tok = PromptTok()
    request = {"state": "A record", "questions": {
        f"q{i}": {"type": "noul", "instructions": "x" * 7000} for i in range(3)
    }}
    # Each decision row fits, but their combined reasoning prompt does not.
    e.context_length = len(e.tok(prompt(request)).input_ids) - overflow
    assert all(len(row["ids"]) < 8192 for row in parse(request).encode(e.tok).rows)
    calls = []
    async def logits(enc, image=None):
        calls.append("direct")
        return [torch.tensor([0.0, 1.0]) for _ in enc.rows]
    async def generate(*args, **kw):
        pytest.fail("reasoning must not enter native generation without context space")
    e.alogits, e._generate = logits, generate
    body = {"request": request, "reasoning": reasoning, "threshold": 1.0}
    if budget is not None:
        body["max_reasoning_tokens"] = budget
    with TestClient(create_app(e)) as client:
        response = client.post("/v1/decide", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["answers"] == result["direct"]["answers"] and "cut" in result
    assert "reasoned" not in result and "reasoning" not in result and calls == ["direct"]


def test_explicit_reasoning_budget_that_does_not_fit_preserves_direct():
    async def run():
        e = engine()
        # Tok renders an eight-token reasoning prompt, leaving sixteen tokens.
        e.context_length = 24
        async def logits(enc, image=None):
            return [torch.tensor([0.0, 1.0])]
        async def generate(*args, **kw):
            pytest.fail("explicit oversized budget must not start generation")
        e.alogits, e._generate = logits, generate
        result = await e.adecide(REQUEST, reasoning=True, max_reasoning_tokens=32)
        assert result["answers"] == result["direct"]["answers"] and "cut" in result
    asyncio.run(run())


@pytest.mark.parametrize("excess", [0, 1])
def test_reasoned_state_includes_the_state_delimiter_budget(excess):
    async def run():
        e = engine()
        thought = "checked"
        overhead = len(parse(with_reasoning({**REQUEST, "state": ""}, thought)).state)
        request = {**REQUEST, "state": "x" * (MAX_STATE - 1 - overhead + excess)}
        calls = []
        async def logits(enc, image=None):
            calls.append(enc)
            return [torch.tensor([0.0, 1.0])]
        async def generate(ids, params, **kw):
            return {"text": thought + "</think>", "meta_info": {"finish_reason": {"type": "stop", "matched": 50}}}
        e.alogits, e._generate = logits, generate
        result = await e.adecide(request, reasoning=True)
        if excess:
            assert result["answers"] == result["direct"]["answers"] and "cut" in result
            assert "reasoned" not in result and len(calls) == 1
        else:
            assert "reasoned" in result and "cut" not in result and len(calls) == 2
    asyncio.run(run())


def test_special_token_escaping_cannot_silently_drop_reasoning():
    class SpecialAwareTok(Tok):
        def __call__(self, text, add_special_tokens=False):
            pieces = re.findall(r"<\|[A-Za-z0-9_]+\|>|.", text, flags=re.DOTALL)
            return SimpleNamespace(input_ids=[999 if part.startswith("<|") else ord(part) + 100 for part in pieces])

    async def run():
        e = engine()
        e.tok = SpecialAwareTok()
        request = {**REQUEST, "state": "<|fim_suffix|>" * 1000}
        thought = "checked"
        augmented = parse(with_reasoning(request, thought))
        assert len(e.tok(augmented.state).input_ids) < MAX_STATE
        # The actual encoder expands control-looking strings before tokenizing.
        from decide_request import _user
        assert len(_user(e.tok, augmented.state)) > MAX_STATE
        calls = []
        async def logits(enc, image=None):
            calls.append(enc)
            return [torch.tensor([0.0, 1.0])]
        async def generate(ids, params, **kw):
            return {"text": thought + "</think>", "meta_info": {"finish_reason": {"type": "stop", "matched": 50}}}
        e.alogits, e._generate = logits, generate
        result = await e.adecide(request, reasoning=True)
        assert result["answers"] == result["direct"]["answers"] and "cut" in result
        assert "reasoned" not in result and len(calls) == 1
    asyncio.run(run())


@pytest.mark.parametrize("options", [{"reasoning": "yes"}, {"threshold": float("nan")}, {"threshold": True},
                                     {"max_reasoning_tokens": 0}, {"max_reasoning_tokens": True}, {"temperature": 0}])
def test_invalid_decision_options_rejected_before_inference(options):
    async def run():
        with pytest.raises(RequestError):
            await engine().adecide(REQUEST, **options)
    asyncio.run(run())


def test_capture_and_generation_are_serial_and_cancel_waits_for_abort():
    async def run():
        e = engine()
        e.loop = asyncio.get_running_loop()
        entered, release = asyncio.Event(), asyncio.Event()
        active, max_active, calls, aborts = 0, 0, [], []
        async def generate(**kw):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            calls.append(kw)
            if len(calls) == 1:
                entered.set()
                await release.wait()
            active -= 1
            return {"text": "ok", "meta_info": {"hidden_states": [[[1.0, 2.0]]], "completion_tokens": 1}}
        def abort(rid):
            aborts.append(rid)
            release.set()
        e.engine = SimpleNamespace(async_generate=generate, tokenizer_manager=SimpleNamespace(abort_request=abort))
        from decide_request import parse
        first = asyncio.create_task(e.alogits(parse(REQUEST).encode(e.tok)))
        await entered.wait()
        second = asyncio.create_task(e._generate([100], {"max_new_tokens": 1}))
        await asyncio.sleep(0)
        assert len(calls) == 1
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        await second
        assert len(aborts) == 1 and max_active == 1 and not e._admission.locked()
    asyncio.run(run())


def test_closing_stream_aborts_before_releasing_admission():
    async def run():
        e = engine()
        e.loop = asyncio.get_running_loop()
        release, aborts = asyncio.Event(), []
        async def chunks():
            yield {"text": "first", "meta_info": {}}
            await release.wait()
            yield {"text": "first done", "meta_info": {}}
        async def generate(**kw):
            return chunks()
        def abort(rid):
            aborts.append(rid)
            release.set()
        e.engine = SimpleNamespace(async_generate=generate, tokenizer_manager=SimpleNamespace(abort_request=abort))
        async with aclosing(e._astream([100], {"max_new_tokens": 8})) as stream:
            assert (await anext(stream))[0] == "first"
        assert aborts and not e._admission.locked()
    asyncio.run(run())


def test_cancellation_before_native_admission_is_retried():
    async def run():
        e = engine()
        e.loop = asyncio.get_running_loop()
        started, released = asyncio.Event(), asyncio.Event()
        admitted, attempts = False, []
        async def generate(**kw):
            nonlocal admitted
            started.set()
            await asyncio.sleep(.02)
            admitted = True
            await released.wait()
            return {"text": "", "meta_info": {}}
        def abort(rid):
            attempts.append(rid)
            if admitted:
                released.set()
        e.engine = SimpleNamespace(async_generate=generate, tokenizer_manager=SimpleNamespace(abort_request=abort))
        task = asyncio.create_task(e._generate([100], {"max_new_tokens": 128}))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert len(attempts) >= 2 and not e._admission.locked()
    asyncio.run(run())


def test_stream_start_failure_is_propagated_without_hanging():
    async def run():
        e = engine()
        e.loop = asyncio.get_running_loop()
        async def fail(**kw):
            raise ValueError("native error")
        e.engine = SimpleNamespace(async_generate=fail)
        with pytest.raises(ValueError, match="native error"):
            async for _ in e._astream([100], {"max_new_tokens": 1}):
                pass
        assert not e._admission.locked()
    asyncio.run(run())


def test_http_forwards_auto_and_budget_and_rejects_sampling_override():
    from fastapi.testclient import TestClient
    from serve_decide import create_app
    e = engine()
    received = []
    async def decide(*args, **kw):
        received.append((args, kw))
        return {"answers": {}}
    e.adecide = decide
    with TestClient(create_app(e)) as client:
        response = client.post("/v1/decide", json={"request": REQUEST, "reasoning": "auto", "threshold": 0.8,
                                                   "max_reasoning_tokens": 12})
        assert response.status_code == 200
        assert received[0][0][-1] == "auto" and received[0][1] == {"threshold": 0.8, "max_reasoning_tokens": 12}
        response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}], "temperature": 0})
        assert response.status_code == 400 and "fixed" in response.text
