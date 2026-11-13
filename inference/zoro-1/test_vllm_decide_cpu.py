"""Decision runner boundary tests, without importing vLLM or allocating a GPU."""
import asyncio
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from decide_request import parse, RequestError
from test_decide_cpu import CharTok

spec = importlib.util.spec_from_file_location(
    "vllm_decide_engine", Path(__file__).parent / "vllm" / "vllm_decide_engine.py")
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
worker_spec = importlib.util.spec_from_file_location(
    "trida_worker_extension", Path(__file__).parent / "vllm" / "trida_worker_extension.py")
worker_module = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker_module)

REQUEST = {"state": "invoice", "questions": {"q": {"type": "noul"}}}


def capture_worker():
    class Model:
        def compute_logits(self, hidden):
            return hidden * 2

    worker = worker_module.DecisionWorkerExtension()
    worker.model_runner = SimpleNamespace(model=Model())
    return worker


def test_worker_capture_is_idempotent_and_does_not_modify_logits():
    worker = capture_worker()
    identity = worker.trida_install_hidden_capture()
    assert worker.trida_install_hidden_capture() == identity
    worker.trida_arm_hidden_capture("first")
    h = torch.tensor([[1.0, 2.0, 3.0]])
    assert worker.model_runner.model.compute_logits(h).tolist() == [[2.0, 4.0, 6.0]]
    h.zero_()  # A reused model buffer must not mutate the saved vector.
    captured = worker.trida_take_hidden_capture("first")
    assert captured["hidden"] == [1.0, 2.0, 3.0]
    assert captured["calls"] == 1 and captured["model_id"] == identity["model_id"]
    assert worker.trida_clear_hidden_capture("first")
    assert worker.model_runner.model.compute_logits(torch.ones(1, 3)).tolist() == [[2.0, 2.0, 2.0]]


def test_worker_rejects_stale_tickets_and_overlapping_capture():
    worker = capture_worker()
    worker.trida_install_hidden_capture()
    worker.trida_arm_hidden_capture("new")
    with pytest.raises(RuntimeError, match="still armed"):
        worker.trida_arm_hidden_capture("other")
    with pytest.raises(RuntimeError, match="ticket"):
        worker.trida_take_hidden_capture("old")
    assert worker.trida_clear_hidden_capture("old") is False
    worker.model_runner.model.compute_logits(torch.ones(1, 3))
    assert worker.trida_take_hidden_capture("new")["ticket"] == "new"


@pytest.mark.parametrize("kind", ["missing", "multiple", "batched", "nan"])
def test_worker_fails_closed_on_ambiguous_capture(kind):
    worker = capture_worker()
    worker.trida_install_hidden_capture()
    worker.trida_arm_hidden_capture("test")
    if kind != "missing":
        h = torch.ones(2 if kind == "batched" else 1, 3)
        if kind == "nan":
            h[0, 0] = float("nan")
        worker.model_runner.model.compute_logits(h)
        if kind == "multiple":
            worker.model_runner.model.compute_logits(h)
    with pytest.raises(RuntimeError):
        worker.trida_take_hidden_capture("test")
    assert worker.trida_clear_hidden_capture("test")


def concurrent_worker():
    worker = worker_module.ConcurrentDecisionWorkerExtension()
    worker.model_runner = SimpleNamespace(model=object(), batch_sharder=None,
        add_requests=lambda output: None, sample=lambda h, b: h.sum().item())
    worker.trida_install_hidden_capture(4)
    return worker


def new_capture(native_id, ticket, length):
    return SimpleNamespace(req_id=native_id, prompt_len=length,
        sampling_params=SimpleNamespace(extra_args={worker_module.CAPTURE_TICKET_KEY: ticket}))


def mixed_batch():
    return SimpleNamespace(req_ids=["native-B-random", "ordinary", "native-A-random"], num_reqs=3,
        is_prefilling_np=[True, False, True], cu_num_logits_np=[0, 1, 2, 3],
        num_computed_prefill_tokens_np=[0, 6, 0], num_scheduled_tokens=[5, 1, 3],
        prefill_len_np=[5, 7, 3], num_draft_tokens=0, logits_indices=torch.tensor([4, 2, 0]))


def test_concurrent_worker_routes_reordered_rows_and_ignores_decode():
    worker = concurrent_worker()
    worker.trida_arm_hidden_capture("A", 3)
    worker.trida_arm_hidden_capture("B", 5)
    worker.model_runner.add_requests(SimpleNamespace(scheduled_new_reqs=[
        new_capture("native-A-random", "A", 3), new_capture("native-B-random", "B", 5)]))
    hidden = torch.arange(18.).reshape(6, 3)
    assert worker.model_runner.sample(hidden, mixed_batch()) == hidden.sum().item()
    a, b = worker.trida_take_hidden_capture("A"), worker.trida_take_hidden_capture("B")
    assert a["native_id"] == "native-A-random" and a["hidden"] == hidden[0].tolist()
    assert b["native_id"] == "native-B-random" and b["hidden"] == hidden[4].tolist()
    status = worker.trida_capture_status()
    assert status["active"] == status["bindings"] == status["failed"] == 0
    assert status["multi_capture_batch_sizes"] == {3: 1}


def test_concurrent_clear_cannot_remove_other_capture_or_revive_late_request():
    worker = concurrent_worker()
    worker.trida_arm_hidden_capture("A", 3)
    worker.trida_arm_hidden_capture("B", 5)
    assert worker.trida_clear_hidden_capture("A")
    worker.model_runner.add_requests(SimpleNamespace(scheduled_new_reqs=[
        new_capture("late-A", "A", 3), new_capture("native-B-random", "B", 5)]))
    assert worker.trida_clear_hidden_capture("A")
    assert worker.trida_capture_status()["active"] == 1
    assert worker.trida_capture_status()["bindings"] == 1
    assert worker.trida_capture_status()["late_ignored"] == 1
    worker.trida_clear_hidden_capture("B")
    assert worker.trida_capture_status()["bindings"] == 0
    assert worker.trida_capture_status()["cancelled_bound"] == 1
    assert worker.trida_capture_status()["cancelled_after_model"] == 0


@pytest.mark.parametrize("bad", ["decode", "multiple_logits", "speculation", "partial", "repeat"])
def test_concurrent_worker_rejects_ambiguous_rows(bad):
    worker = concurrent_worker()
    worker.trida_arm_hidden_capture("A", 3)
    worker.model_runner.add_requests(SimpleNamespace(scheduled_new_reqs=[new_capture("native-A-random", "A", 3)]))
    batch = mixed_batch()
    if bad == "decode":
        batch.is_prefilling_np[2] = False
    elif bad == "multiple_logits":
        batch.cu_num_logits_np[-1] = 4
    elif bad == "speculation":
        batch.num_draft_tokens = 1
    elif bad == "partial":
        batch.num_scheduled_tokens[2] = 1
    h = torch.arange(18.).reshape(6, 3)
    worker.model_runner.sample(h, batch)
    if bad == "repeat":
        worker.model_runner.sample(h, batch)
    with pytest.raises(RuntimeError):
        worker.trida_take_hidden_capture("A")
    assert worker.trida_clear_hidden_capture("A")


def test_concurrent_worker_capacity_is_bounded():
    worker = concurrent_worker()
    for i in range(4):
        worker.trida_arm_hidden_capture(str(i), 3)
    with pytest.raises(RuntimeError, match="capacity"):
        worker.trida_arm_hidden_capture("overflow", 3)
    worker.trida_clear_hidden_capture("0")
    worker.trida_arm_hidden_capture("replacement", 3)


def test_http_disconnect_waits_for_decision_cleanup():
    pytest.importorskip("fastapi")
    from serve_decide import _decision_while_connected
    from fastapi import HTTPException

    async def scenario():
        started, disconnect, cleaning, finish = (asyncio.Event() for _ in range(4))

        class Request:
            async def is_disconnected(self):
                await disconnect.wait()
                return True

        async def work():
            try:
                started.set()
                await asyncio.Event().wait()
            finally:
                cleaning.set()
                await finish.wait()

        task = asyncio.create_task(_decision_while_connected(work(), Request()))
        await started.wait()
        disconnect.set()
        await cleaning.wait()
        assert not task.done()
        finish.set()
        with pytest.raises(HTTPException) as error:
            await task
        assert error.value.status_code == 499

    asyncio.run(scenario())


class Tok(CharTok):
    def apply_chat_template(self, *args, **kwargs):
        return "reasoning prompt"


def bare_engine():
    engine = backend.VllmDecideEngine.__new__(backend.VllmDecideEngine)
    engine.tok, engine.proc = Tok(), None
    engine.text_T = engine.image_T = 1
    engine.image_pad = engine.tok.convert_tokens_to_ids("<|image_pad|>")
    engine.context_length, engine.reasoning_temperature, engine._stops = 8192, 0.6, [99]
    engine._execution_lock = asyncio.Lock()
    engine._worker_identity = {"pid": 12, "model_id": 34}
    engine.max_concurrency = 1
    return engine


def test_image_prompt_collapses_only_the_image_span():
    row = {"ids": [1, 6, 7, 7, 7, 8, 2, 3, 4, 5]}
    image = SimpleNamespace(width=56, height=84)
    out = backend.row_prompt(row, 7, image)
    assert out["prompt_token_ids"] == [1, 6, 7, 8, 2, 3, 4, 5]
    assert row["ids"] == [1, 6, 7, 7, 7, 8, 2, 3, 4, 5]
    assert out["multi_modal_data"]["image"] is image
    assert out["mm_processor_kwargs"] == {"min_pixels": 4704, "max_pixels": 4704}


def test_context_smaller_than_decision_rows_rejected_before_loading():
    with pytest.raises(ValueError, match="context_length"):
        backend.VllmDecideEngine("unused", context_length=4096)


@pytest.mark.parametrize("concurrency", [1, 2, 4])
def test_only_one_generation_runner_is_created(monkeypatch, concurrency):
    made = []

    class AsyncLLM:
        @staticmethod
        def from_engine_args(args):
            made.append(args)
            return AsyncLLM()

        async def collective_rpc(self, method, **kw):
            assert method == "trida_install_hidden_capture"
            return [{"pid": 12, "model_id": 34}]

    for module, contents in {
        "vllm.engine.arg_utils": {"AsyncEngineArgs": lambda **kw: kw},
        "vllm.v1.engine.async_llm": {"AsyncLLM": AsyncLLM},
    }.items():
        monkeypatch.setitem(sys.modules, module, SimpleNamespace(**contents))
    engine = bare_engine()
    engine.max_concurrency = concurrency
    asyncio.run(engine._start("unused", 0.5, 8192))
    assert len(made) == 1
    generation = made[0]
    assert generation["max_num_seqs"] == concurrency
    assert generation["enable_chunked_prefill"] is False
    assert generation["max_num_batched_tokens"] >= 8192
    assert generation["enable_prefix_caching"] is False
    assert generation["runner"] == "generate"
    name = "ConcurrentDecisionWorkerExtension" if concurrency > 1 else "DecisionWorkerExtension"
    assert generation["worker_extension_cls"] == f"trida_worker_extension.{name}"


@pytest.mark.parametrize("bad", ["short", "nan", "ticket", "model", "calls", "unfinished"])
def test_capture_identity_and_finiteness_are_required(monkeypatch, bad):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    engine = bare_engine()
    engine.head = SimpleNamespace(q=SimpleNamespace(in_features=4))
    cleared = []

    class Generation:
        async def generate(self, *args):
            yield SimpleNamespace(finished=bad != "unfinished")

    async def rpc(method, ticket):
        if method == "trida_arm_hidden_capture":
            return ticket
        if method == "trida_clear_hidden_capture":
            cleared.append(ticket)
            return True
        return {"ticket": "wrong" if bad == "ticket" else ticket,
                "pid": 12, "model_id": 99 if bad == "model" else 34,
                "calls": 2 if bad == "calls" else 1,
                "hidden": [float("nan") if bad == "nan" else 0.] * (3 if bad == "short" else 4)}

    engine.generating, engine._worker_rpc = Generation(), rpc
    with pytest.raises(RuntimeError):
        asyncio.run(engine._capture_prefix({"prompt_token_ids": [1]}))
    assert len(cleared) == 1 and not engine._execution_lock.locked()


def test_selected_hidden_states_follow_option_order():
    engine = bare_engine()
    enc = parse(REQUEST).encode(engine.tok)
    received = []

    async def capture(request, expected_len):
        assert expected_len == len(request["prompt_token_ids"])
        return torch.tensor([len(request["prompt_token_ids"]) - 1.])

    def head(decide, options, owner):
        received.append((decide.tolist(), options.tolist(), owner.tolist()))
        return torch.tensor([0.0, 1.0])

    engine._capture_prefix, engine.head = capture, head
    asyncio.run(engine.alogits(enc))
    row = enc.rows[0]
    assert received == [([[row["decide"]]], [[p] for p in row["opts"]], [0, 0])]


@pytest.mark.parametrize("text,finish,cut", [
    ("unfinished", "length", "length"), ("", "stop", "empty"),
    ("x" * 8000, "stop", "state_limit"), ("yes, the invoice is valid", "stop", None),
])
def test_reasoning_fallback_keeps_original_state(text, finish, cut):
    engine = bare_engine()
    seen = []

    async def stream(*args, **kw):
        yield text, {"finish_reason": finish, "completion_tokens": 10}

    async def logits(enc, image):
        seen.append(enc.rows[0]["ids"])
        return [torch.tensor([0.0, 1.0])]

    engine._astream, engine.alogits = stream, logits
    answer = asyncio.run(engine.adecide(REQUEST, reasoning=True))
    assert answer.get("cut") == cut
    assert answer["answers_source"] == ("direct" if cut else "reasoned")
    assert answer["reasoning"] == text
    original = parse(REQUEST).encode(engine.tok).rows[0]["ids"]
    assert (seen[0] == original) == bool(cut)


@pytest.mark.parametrize("budget", [0, True, 4097])
def test_reasoning_budget_rejected_before_generation(budget):
    with pytest.raises(RequestError, match="max_reasoning_tokens"):
        asyncio.run(bare_engine().adecide(REQUEST, reasoning=True, max_reasoning_tokens=budget))


def test_generation_preserves_sampling_and_stream_finish_metadata(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))
    captured = []

    class Generation:
        async def generate(self, request, params, request_id):
            captured.append((request, params))
            for text, ids, finish in [("a", [1], None), ("ab", [1, 2], "length")]:
                yield SimpleNamespace(outputs=[SimpleNamespace(text=text, token_ids=ids, finish_reason=finish)])

    engine = bare_engine()
    engine.engine = backend._Generation(Generation())

    async def consume():
        return [event async for event in engine._astream([10], {
            "max_new_tokens": 2, "temperature": 0.6, "top_k": 20, "top_p": 0.95})]

    events = asyncio.run(consume())
    assert [piece for piece, _ in events] == ["a", "b"]
    assert events[-1][1] == {"completion_tokens": 2, "finish_reason": "length"}
    assert captured == [({"prompt_token_ids": [10]}, {
        "max_tokens": 2, "temperature": 0.6, "top_k": 20, "top_p": 0.95})]


def test_cancelled_capture_finishes_cleanup_before_unlocking(monkeypatch):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))

    async def scenario():
        engine = bare_engine()
        engine.head = SimpleNamespace(q=SimpleNamespace(in_features=4))
        started, cleanup_started, allow_cleanup, allow_generate = (asyncio.Event() for _ in range(4))
        closed = []

        class Generation:
            async def generate(self, *args):
                try:
                    started.set()
                    await allow_generate.wait()
                    yield SimpleNamespace(finished=True)
                finally:
                    closed.append(True)

        async def rpc(method, ticket):
            if method == "trida_arm_hidden_capture":
                return ticket
            if method == "trida_clear_hidden_capture":
                cleanup_started.set()
                await allow_cleanup.wait()
                return True
            return {"ticket": ticket, "calls": 1, "pid": 12, "model_id": 34, "hidden": [1.] * 4}

        engine.generating, engine._worker_rpc = Generation(), rpc
        task = asyncio.create_task(engine._capture_prefix({"prompt_token_ids": [1]}))
        await started.wait()
        task.cancel()
        await cleanup_started.wait()
        assert closed and engine._execution_lock.locked()
        task.cancel()  # A second cancellation must not abandon worker cleanup.
        await asyncio.sleep(0)
        assert engine._execution_lock.locked()
        allow_cleanup.set()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not engine._execution_lock.locked()
        allow_generate.set()
        assert (await engine._capture_prefix({"prompt_token_ids": [2]})).tolist() == [1.] * 4

    asyncio.run(scenario())


@pytest.mark.parametrize("finish", ["close", "cancel"])
def test_capture_waits_for_generation_iterator_cleanup(monkeypatch, finish):
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(SamplingParams=lambda **kw: kw))

    async def scenario():
        engine = bare_engine()
        engine.head = SimpleNamespace(q=SimpleNamespace(in_features=4))
        blocked, pending = asyncio.Event(), asyncio.Event()
        closed, arms = [], []

        class Generation:
            async def generate(self, request, params, request_id):
                if request["prompt_token_ids"] == [10]:
                    try:
                        yield SimpleNamespace(outputs=[SimpleNamespace(text="a", token_ids=[1], finish_reason=None)])
                        pending.set()
                        await blocked.wait()
                    finally:
                        closed.append(True)
                else:
                    yield SimpleNamespace(finished=True)

        async def rpc(method, ticket):
            if method == "trida_arm_hidden_capture":
                assert closed
                arms.append(ticket)
                return ticket
            if method == "trida_clear_hidden_capture":
                return True
            return {"ticket": ticket, "calls": 1, "pid": 12, "model_id": 34, "hidden": [1.] * 4}

        engine.generating, engine._worker_rpc = Generation(), rpc
        adapter = backend._Generation(engine.generating, engine._execution_lock)
        stream = await adapter.async_generate([10], {"max_new_tokens": 8}, stream=True)
        assert (await stream.__anext__())["text"] == "a"
        capture = asyncio.create_task(engine._capture_prefix({"prompt_token_ids": [2]}))
        await asyncio.sleep(0)
        assert not arms
        if finish == "close":
            await stream.aclose()
        else:
            next_chunk = asyncio.create_task(stream.__anext__())
            await pending.wait()
            next_chunk.cancel()
            with pytest.raises(asyncio.CancelledError):
                await next_chunk
        assert (await capture).tolist() == [1.] * 4
        assert len(arms) == 1 and not engine._execution_lock.locked()

    asyncio.run(scenario())


def test_http_forwards_requested_reasoning_budget():
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from serve_decide import create_app
    captured = []

    class Engine:
        ckpt = "test"

        async def adecide(self, request, image, temperature, reasoning, *, max_reasoning_tokens):
            captured.append((request, reasoning, max_reasoning_tokens))
            return {"answers": {}}

    response = TestClient(create_app(Engine())).post("/v1/decide", json={
        "request": REQUEST, "reasoning": True, "max_reasoning_tokens": 1})
    assert response.status_code == 200
    assert captured == [(REQUEST, True, 1)]
