"""One text-only diffusion engine for self-spec chat, reasoning, and decisions.

Requires the pinned noncommercial SGLang fork and decision_readout.patch.
Admission is serial, including streamed generation and cancellation cleanup.
"""
import asyncio
from contextlib import aclosing
import math
from pathlib import Path
import time
import uuid

import torch

from decide_engine import DecideEngine, END_THINK, _strip_end, prefix_prompts
from decide_request import MAX_STATE, RequestError, parse, prompt, with_reasoning


def top_probability(answer):
    return max(answer["noul"], 1 - answer["noul"]) if answer["type"] == "noul" else max(answer["probabilities"].values())


class SelfSpecDecideEngine(DecideEngine):
    supports_auto_reasoning = True

    def __init__(self, ckpt, mem_fraction_static=0.5, context_length=16384, config=None):
        self.context_length = context_length
        self.config = Path(config) if config else Path(__file__).parent / "configs/zoro1_self_spec_b15_g8.yaml"
        import yaml
        settings = yaml.safe_load(self.config.read_text())
        if settings.get("causal_prefill") is not True:
            raise ValueError("decisions require causal_prefill: true")
        self.sampling = {"temperature": settings.get("temperature", 1.0),
                         "top_p": settings.get("top_p", 1.0), "top_k": settings.get("top_k", -1)}
        self._admission = asyncio.Lock()
        self._failed = False
        super().__init__(ckpt, mem_fraction_static, context_length,
                         reasoning_temperature=self.sampling["temperature"])

    def decide(self, request, image=None, temperature=None, reasoning=False,
               max_reasoning_tokens=None, threshold=0.7):
        return self._run(self.adecide(request, image, temperature, reasoning,
                                      max_reasoning_tokens, threshold))

    def _build_engine(self, ckpt, mem_fraction_static, context_length, engine_kw):
        import sglang as sgl
        from sglang.srt.layers.logits_processor import LogitsProcessor
        from serve import for_sglang
        if getattr(LogitsProcessor, "trida_decision_readout_version", None) != 1:
            raise RuntimeError("Apply sglang/decision_readout.patch after two_stream_diffusion.patch; rebuild the backend")
        return sgl.Engine(model_path=for_sglang(ckpt), mem_fraction_static=mem_fraction_static,
                          context_length=context_length, dllm_algorithm="HybridDiffusionSelfSpec",
                          dllm_algorithm_config=str(self.config), enable_return_hidden_states=True,
                          disable_cuda_graph=True, disable_overlap_schedule=True,
                          max_running_requests=1, attention_backend="flashinfer", log_level="warning")

    def _check_generation(self, ids, params):
        if self._failed:
            raise RuntimeError("self-spec engine cleanup failed; restart the server")
        budget = params.get("max_new_tokens", 1)
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise RequestError("max tokens must be a positive integer")
        if len(ids) + budget > self.context_length:
            raise RequestError("prompt plus generation budget exceeds the server context length")

    async def _cleanup(self, task, rid):
        import anyio
        with anyio.CancelScope(shield=True):
            await self._drain(task, rid)

    async def _drain(self, task, rid):
        # Keep admission until the native engine acknowledges completion/abort.
        deadline = self.loop.time() + 30
        while not task.done():
            # A cancellation can beat tokenization/admission. Retry the
            # idempotent abort until the native request reaches a terminal state.
            self.engine.tokenizer_manager.abort_request(rid)
            try:
                await asyncio.wait_for(asyncio.shield(task), min(0.1, max(0.001, deadline - self.loop.time())))
            except asyncio.CancelledError:
                continue
            except asyncio.TimeoutError:
                if self.loop.time() < deadline:
                    continue
                self._failed = True
                task.cancel()
                raise RuntimeError("self-spec abort did not finish; restart the server") from None
            except Exception:
                break
        if not task.cancelled():
            task.exception()  # consume a native abort/error after cancellation

    async def _generate(self, ids, params, **kw):
        if kw.get("image_data") is not None:
            raise RequestError("self-spec decision serving currently supports text only")
        self._check_generation(ids, params)
        async with self._admission:
            self._check_generation(ids, params)
            rid = uuid.uuid4().hex
            task = asyncio.create_task(self.engine.async_generate(input_ids=ids, sampling_params=params, rid=rid, **kw))
            try:
                return await asyncio.shield(task)
            finally:
                await self._cleanup(task, rid)

    async def _astream(self, ids, params, **kw):
        if kw.get("image_data") is not None:
            raise RequestError("self-spec decision serving currently supports text only")
        self._check_generation(ids, params)
        async with self._admission:
            self._check_generation(ids, params)
            rid, queue = uuid.uuid4().hex, asyncio.Queue()

            async def produce():
                try:
                    generator = await self.engine.async_generate(input_ids=ids, sampling_params=params,
                                                                  stream=True, rid=rid, **kw)
                    async with aclosing(generator):
                        async for chunk in generator:
                            queue.put_nowait(chunk)
                finally:
                    queue.put_nowait(None)

            task = asyncio.create_task(produce())
            seen = ""
            try:
                while (chunk := await queue.get()) is not None:
                    text = chunk["text"]
                    piece = text[len(seen):] if text.startswith(seen) else text
                    seen = text if text.startswith(seen) else seen + text
                    yield piece, chunk.get("meta_info", {})
                task.result()
            finally:
                await self._cleanup(task, rid)

    async def alogits(self, enc, image=None):
        if image is not None or enc.has_image:
            raise RequestError("self-spec decision serving currently supports text only")
        prompts, spans = prefix_prompts(enc, self.image_pad)
        states = []
        # A whole decision's prefixes run under one admission lock. _generate
        # itself also takes that lock, so call the native API here directly.
        async with self._admission:
            for ids in prompts:
                params = {"max_new_tokens": 1, "temperature": 0}
                self._check_generation(ids, params)
                rid = uuid.uuid4().hex
                task = asyncio.create_task(self.engine.async_generate(input_ids=ids, sampling_params=params,
                                                                      return_hidden_states=True, rid=rid))
                try:
                    out = await asyncio.shield(task)
                finally:
                    await self._cleanup(task, rid)
                h = torch.tensor(out["meta_info"].get("hidden_states", []), dtype=torch.float32)
                if h.numel() != self.head.q.in_features or not torch.isfinite(h).all():
                    raise RuntimeError("invalid decision hidden states; check the readout patch")
                states.append(h.reshape(-1))
        states = torch.stack(states)
        with torch.no_grad():
            return [self.head(states[start + n - 1:start + n], states[start:start + n - 1],
                              torch.zeros(n - 1, dtype=torch.long)) for start, n in spans]

    async def adecide(self, request, image=None, temperature=None, reasoning=False,
                      max_reasoning_tokens=None, threshold=0.7):
        if not isinstance(reasoning, bool) and reasoning != "auto":
            raise RequestError('reasoning must be a boolean or "auto"')
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not math.isfinite(threshold) or not 0 <= threshold <= 1:
            raise RequestError("threshold must be between 0 and 1")
        if temperature is not None and (isinstance(temperature, bool) or not isinstance(temperature, (int, float))
                                        or not math.isfinite(temperature) or temperature <= 0):
            raise RequestError("temperature must be finite and positive")
        budget = max_reasoning_tokens if max_reasoning_tokens is not None else (8192 if reasoning == "auto" else 768)
        if not isinstance(budget, int) or isinstance(budget, bool) or budget < 1:
            raise RequestError("max_reasoning_tokens must be a positive integer")
        req = parse(request, image)
        if req.image is not None:
            raise RequestError("self-spec decision serving currently supports text only")
        start = time.perf_counter()
        direct = await super().adecide(request, temperature=temperature)
        if reasoning is False:
            return direct
        qid, top = min(((qid, top_probability(a)) for qid, a in direct["answers"].items()), key=lambda x: x[1])
        result = {**direct, "direct": direct, "escalated": reasoning is True or top < threshold,
                  "threshold": threshold}
        if reasoning == "auto":
            result["lowest"] = {"question": qid, "top_probability": top}
        if result["escalated"]:
            result["reason"] = (f"reasoned because top probability was {top:.4g} (question {qid}), below {threshold}"
                                if reasoning == "auto" else "reasoning requested")
            rendered = self.tok.apply_chat_template([{"role": "user", "content": prompt(request)}],
                                                    tokenize=False, add_generation_prompt=True)
            ids = self.tok(rendered).input_ids
            available = self.context_length - len(ids)
            if available < 1 or (max_reasoning_tokens is not None and budget > available):
                result["cut"] = "reasoning prompt and budget exceed the server context; using the direct answer"
                result["timing_ms"] = round((time.perf_counter() - start) * 1000, 1)
                return result
            if max_reasoning_tokens is None:
                budget = min(budget, available)
            self._check_generation(ids, {"max_new_tokens": budget})
            thought_start = time.perf_counter()
            out = await self._generate(ids, {**self.sampling, "max_new_tokens": budget,
                                             "skip_special_tokens": False, "stop_token_ids": self._stops})
            raw = out["text"]
            finish = out["meta_info"].get("finish_reason") or {}
            complete = END_THINK in raw or (isinstance(finish, dict) and finish.get("matched") == self._stops[0])
            text = raw.split(END_THINK, 1)[0].replace("<think>", "").strip()
            result.update(reasoning=text, reasoning_ms=round((time.perf_counter() - thought_start) * 1000, 1))
            if not complete or not text:
                result["cut"] = "reasoning did not finish; using the direct answer"
            else:
                augmented = with_reasoning(request, text)
                # Do not silently truncate appended reasoning out of the state.
                if parse(augmented).state_token_count(self.tok) > MAX_STATE:
                    result["cut"] = "reasoning exceeds the decision state limit; using the direct answer"
                else:
                    try:
                        result["reasoned"] = await super().adecide(augmented, temperature=temperature)
                        result["answers"] = result["reasoned"]["answers"]
                    except RequestError:
                        result["cut"] = "reasoning exceeds the decision row limit; using the direct answer"
        result["timing_ms"] = round((time.perf_counter() - start) * 1000, 1)
        return result

    def _chat_request(self, messages, tools, max_new_tokens, temperature, top_p, top_k,
                      presence_penalty, enable_thinking):
        for message in messages:
            content = message.get("content")
            if isinstance(content, list) and any(part.get("type") != "text" for part in content):
                raise RequestError("self-spec decision serving currently supports text only")
        requested = {"temperature": temperature, "top_p": top_p, "top_k": top_k}
        if requested != self.sampling or presence_penalty != 0:
            raise RequestError(f"self-spec sampling is fixed by the server config: {self.sampling}; presence_penalty must be 0")
        return super()._chat_request(messages, tools, max_new_tokens, temperature, top_p, top_k,
                                     presence_penalty, enable_thinking)

    def validate_chat(self, messages, **kw):
        ids, params = self._chat_request(messages, **kw)
        self._check_generation(ids, params)

    async def achat(self, messages, tools=None, max_new_tokens=2048, temperature=1.0, top_p=0.95,
                    top_k=20, presence_penalty=0.0, enable_thinking=None):
        ids, params = self._chat_request(messages, tools, max_new_tokens, temperature, top_p, top_k,
                                         presence_penalty, enable_thinking)
        out = await self._generate(ids, params)
        meta = out["meta_info"]
        return {"text": _strip_end(out["text"]), "prompt_tokens": len(ids),
                "completion_tokens": meta.get("completion_tokens", 0), "finish_reason": self._finish(meta)}

    async def achat_stream(self, messages, tools=None, max_new_tokens=2048, temperature=1.0, top_p=0.95,
                           top_k=20, presence_penalty=0.0, enable_thinking=None):
        ids, params = self._chat_request(messages, tools, max_new_tokens, temperature, top_p, top_k,
                                         presence_penalty, enable_thinking)
        meta = {}
        async with aclosing(self._astream(ids, params)) as stream:
            async for piece, meta in stream:
                if piece := _strip_end(piece, strip=False):
                    yield piece
        yield {"prompt_tokens": len(ids), "completion_tokens": meta.get("completion_tokens", 0),
               "finish_reason": self._finish(meta)}
