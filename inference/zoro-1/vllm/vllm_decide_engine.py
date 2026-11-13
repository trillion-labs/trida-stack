"""Typed decisions, reasoning, and chat from one stock vLLM generation runner.

A worker extension returns request-indexed prompt hidden states before the
unchanged vocabulary projection. Concurrency is bounded to four requests;
concurrency one retains the original exclusive capture path.
"""
import asyncio
from contextlib import aclosing
import os
from pathlib import Path
import threading
import time
import uuid

import torch

from decide_engine import (DecideEngine, PointerHead, END_THINK, load_image_processor,
                           serving_temperatures)
from decide_request import (MAX_STATE, MAX_BRANCH, VISION, RequestError, _user, parse, prompt,
                            render, with_reasoning, to_answer)


def row_prompt(row, image_pad, image=None):
    """vLLM expands the single image placeholder back into the encoded image span."""
    ids = list(row["ids"])
    if image is None:
        return {"prompt_token_ids": ids}
    first = ids.index(image_pad)
    count = ids.count(image_pad)
    return {"prompt_token_ids": ids[:first + 1] + ids[first + count:],
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {"min_pixels": image.width * image.height,
                                    "max_pixels": image.width * image.height}}


class _Generation:
    """The small SGLang-shaped generation interface used by the shared chat code."""

    def __init__(self, engine, execution_lock=None):
        self.engine = engine
        self.execution_lock = execution_lock if execution_lock is not None else asyncio.Lock()

    async def async_generate(self, input_ids, sampling_params, stream=False, image_data=None):
        if self.engine is None:
            raise RuntimeError("generation engine is not initialized")
        from vllm import SamplingParams
        params = dict(sampling_params)
        params["max_tokens"] = params.pop("max_new_tokens")
        request = {"prompt_token_ids": input_ids}
        if image_data is not None:
            request["multi_modal_data"] = {"image": image_data}
            request["mm_processor_kwargs"] = {"min_pixels": image_data.width * image_data.height,
                                               "max_pixels": image_data.width * image_data.height}

        async def events():
            async with self.execution_lock:
                # Closing the native iterator aborts cancelled streaming requests
                # before a subsequent hidden-state capture acquires the lock.
                async with aclosing(self.engine.generate(request, SamplingParams(**params), uuid.uuid4().hex)) as outputs:
                    async for out in outputs:
                        completion = out.outputs[0]
                        yield {"text": completion.text, "meta_info": {
                            "completion_tokens": len(completion.token_ids),
                            "finish_reason": completion.finish_reason}}

        if stream:
            return events()
        last = None
        async for last in events():
            pass
        if last is None:
            raise RuntimeError("vLLM returned no generation result")
        return last


class VllmDecideEngine(DecideEngine):
    """Reuse the decision contract and chat formatting; replace only engine execution."""

    def __init__(self, ckpt, mem_fraction=0.5, context_length=16384, reasoning_temperature=0.6,
                 max_concurrency=4):
        if not 0 < mem_fraction < 1:
            raise ValueError("mem_fraction must be > 0 and < 1")
        if type(context_length) is not int or context_length < MAX_BRANCH:
            raise ValueError(f"context_length must be at least {MAX_BRANCH} to fit decision rows")
        if type(max_concurrency) is not int or max_concurrency not in (1, 2, 4):
            raise ValueError("max_concurrency must be 1, 2, or 4")
        from transformers import AutoTokenizer
        self.ckpt, self.reasoning_temperature = ckpt, reasoning_temperature
        self.context_length = context_length
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.proc = load_image_processor(ckpt)
        saved = torch.load(Path(ckpt) / "jev_head.pt", map_location="cpu", weights_only=True)
        self.head = PointerHead(saved["head"]["q.weight"].shape[1], saved["head_dim"]).eval()
        self.head.load_state_dict(saved["head"])
        self.text_T, self.image_T = serving_temperatures(ckpt)
        self.image_pad = self.tok.convert_tokens_to_ids(VISION[1])
        self._stops = [self.tok.convert_tokens_to_ids(t) for t in (END_THINK, "<|im_end|>")]
        self._lock = threading.Lock()
        self.loop = asyncio.new_event_loop()
        self.max_concurrency = max_concurrency
        self._execution_lock = asyncio.BoundedSemaphore(max_concurrency)
        # Disable ambient diffusion plugins before spawning the engine.
        os.environ["VLLM_PLUGINS"] = ""
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        # Avoid a runtime NVCC build for the optional FlashInfer sampler.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        self.generating = None
        try:
            self.loop.run_until_complete(self._start(ckpt, mem_fraction, context_length))
        except BaseException:
            self.shutdown()
            raise
        self.engine = _Generation(self.generating, self._execution_lock)

    async def _start(self, ckpt, memory, context):
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        concurrency = self.max_concurrency
        extension = ("ConcurrentDecisionWorkerExtension" if concurrency > 1 else "DecisionWorkerExtension")
        self.generating = AsyncLLM.from_engine_args(AsyncEngineArgs(
            model=ckpt, runner="generate", dtype="bfloat16", max_model_len=context,
            gpu_memory_utilization=memory, max_num_seqs=concurrency, enforce_eager=True,
            enable_prefix_caching=False, trust_remote_code=True,
            enable_chunked_prefill=False, max_num_batched_tokens=context,
            worker_extension_cls=f"trida_worker_extension.{extension}"))
        args = (concurrency,) if concurrency > 1 else ()
        self._worker_identity = await self._worker_rpc("trida_install_hidden_capture", *args)

    async def _worker_rpc(self, method, *args):
        replies = await self.generating.collective_rpc(method, args=args, timeout=30)
        if len(replies) != 1:
            raise RuntimeError("decision capture supports exactly one worker")
        return replies[0]

    async def _clear_capture(self, ticket):
        # Keep the lock until cleanup finishes, even if the caller is cancelled
        # again while its native generation iterator is being closed.
        cleanup = asyncio.create_task(self._worker_rpc("trida_clear_hidden_capture", ticket))
        cancelled = False
        while not cleanup.done():
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                cancelled = True
        if cleanup.result() is not True:
            raise RuntimeError("worker capture cleanup did not complete")
        if cancelled:
            raise asyncio.CancelledError

    async def _capture_prefix(self, request, expected_len=None):
        from vllm import SamplingParams
        ticket = uuid.uuid4().hex
        concurrent = self.max_concurrency > 1
        expected_len = expected_len if expected_len is not None else len(request["prompt_token_ids"])
        async with self._execution_lock:
            try:
                args = (ticket, expected_len) if concurrent else (ticket,)
                armed = await self._worker_rpc("trida_arm_hidden_capture", *args)
                if armed != ticket:
                    raise RuntimeError("worker armed a different capture ticket")
                finished = False
                extra = {"extra_args": {"trida_capture_ticket": ticket}} if concurrent else {}
                async with aclosing(self.generating.generate(
                    request, SamplingParams(max_tokens=1, temperature=0, **extra), ticket)) as outputs:
                    async for out in outputs:
                        finished = out.finished
                if not finished:
                    raise RuntimeError("capture generation did not finish")
                result = await self._worker_rpc("trida_take_hidden_capture", ticket)
                if (result["ticket"] != ticket or result["calls"] != 1
                        or any(result[k] != self._worker_identity[k] for k in ("pid", "model_id"))):
                    raise RuntimeError("hidden state came from an unexpected capture or model")
                if concurrent and (result.get("expected_len") != expected_len or not result.get("native_id")):
                    raise RuntimeError("capture is not bound to the expected native request")
                h = torch.tensor(result["hidden"], dtype=torch.float32)
                if h.shape != (self.head.q.in_features,) or not torch.isfinite(h).all():
                    raise RuntimeError("captured hidden state has invalid dimensions or values")
                return h
            finally:
                await self._clear_capture(ticket)

    async def alogits(self, enc, image=None):
        if enc.has_image and image is None:
            raise ValueError("an image request needs its sized image")

        logits = []
        for row in enc.rows:
            states = []
            for pos in row["opts"] + [row["decide"]]:
                states.append(await self._capture_prefix(row_prompt(
                    {"ids": row["ids"][:pos + 1]}, self.image_pad, image), expected_len=pos + 1))
            h = torch.stack(states)
            with torch.no_grad():
                logits.append(self.head(h[-1:], h[:-1], torch.zeros(len(h) - 1, dtype=torch.long)))
        return logits

    async def adecide(self, request, image=None, temperature=None, reasoning=False, max_reasoning_tokens=768):
        t0 = time.perf_counter()
        req = parse(request, image)
        enc = req.encode(self.tok, self.proc)
        extra = {}
        if reasoning:
            if type(max_reasoning_tokens) is not int or not 1 <= max_reasoning_tokens <= 4096:
                raise RequestError("max_reasoning_tokens must be between 1 and 4096")
            messages = [{"role": "user", "content": prompt(request)}]
            kw = {}
            if req.image is not None:
                messages[0]["content"] = [{"type": "image"}, {"type": "text", "text": prompt(request)}]
                kw["image_data"] = self._sized(req.image, enc.grid)
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            ids = self.tok(text).input_ids
            image_tokens = int(enc.grid.prod(-1)[0]) // self.proc.merge_size ** 2 if enc.has_image else 0
            if len(ids) + max(0, image_tokens - 1) + max_reasoning_tokens > self.context_length:
                raise RequestError("reasoning prompt plus budget exceeds server context length")
            params = {"max_new_tokens": max_reasoning_tokens, "temperature": self.reasoning_temperature,
                      "top_p": 0.95, "top_k": 20, "stop_token_ids": self._stops}
            pieces, meta = [], {}
            async for piece, meta in self._astream(ids, params, **kw):
                pieces.append(piece.replace(END_THINK, ""))
            text = "".join(pieces).strip()
            extra = {"reasoning": text, "reasoning_ms": round((time.perf_counter() - t0) * 1000, 1),
                     "reasoning_stats": {"finish_reason": self._finish(meta),
                                         "completion_tokens": meta.get("completion_tokens", 0)},
                     "reasoning_trimmed": False}
            combined = with_reasoning(request, text)
            if self._finish(meta) == "length":
                extra["cut"] = "length"
            elif not text:
                extra["cut"] = "empty"
            elif len(_user(self.tok, render(combined["state"]))) + 1 + (image_tokens + 2 if image_tokens else 0) > MAX_STATE:
                # Preserve the original input and never silently score shortened reasoning.
                extra.update(cut="state_limit", reasoning_trimmed=True)
            else:
                req = parse(combined, req.image)
                enc = req.encode(self.tok, self.proc)
            extra["answers_source"] = "direct" if "cut" in extra else "reasoned"
        T = temperature or (self.image_T if enc.has_image else self.text_T)
        z = await self.alogits(enc, self._sized(req.image, enc.grid) if enc.has_image else None)
        answers = {q.qid: to_answer(q, torch.softmax(zq / T, -1).tolist()) for q, zq in zip(req.questions, z)}
        return {"answers": answers, "timing_ms": round((time.perf_counter() - t0) * 1000, 1), **extra}

    def shutdown(self):
        if self.generating is not None:
            self.generating.shutdown()
