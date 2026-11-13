#!/usr/bin/env python3
"""Serve a Trida decision checkpoint: typed decisions and chat from one engine, one copy of the weights.

  POST /v1/decide            {"request": {"state": ..., "questions": {...}}, "reasoning": false, "temperature": null}
                             -> {"answers": {qid: answer}, "timing_ms": ...} (+ "reasoning" when reasoning=true)
  POST /v1/chat/completions  OpenAI-compatible, streaming or not; `tools` supported (tool calls parsed from the
                             model's <tool_call> blocks); the thinking is returned as reasoning_content; honours
                             max_tokens, temperature, top_p, top_k, presence_penalty and
                             chat_template_kwargs.enable_thinking, and reports usage
  GET  /v1/models            the served model id (chat.py / eval.py read it)
  GET  /health

Usage:
  python serve_decide.py /path/to/decision-checkpoint --port 30001     # or $TRIDA_DECIDE_MODEL
  python serve.py /path/to/decision-checkpoint --mode decide           # the same, through serve.py

Defaults to stock SGLang >= 0.5.22 (Apache-2.0). --backend self-spec selects the experimental
noncommercial diffusion backend with decision_readout.patch: text-only, serial admission,
fixed YAML sampling. See README "Decisions". Both need fastapi and uvicorn.
A decision checkpoint is a Trida checkpoint with jev_head.pt (MODEL_CONTRACT.md); an optional temperature.json sets
its serving temperatures. The default uses causal decoding; the optional backend uses self-spec generation.
"""
import argparse
import asyncio
from contextlib import aclosing, suppress
import json
import os
import re
import time
import uuid

END_THINK, CALL_OPEN = "</think>", "<tool_call>"
_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def _string_params(tools):
    """{function name: names of its parameters typed string} from the request's OpenAI `tools`."""
    out = {}
    for t in tools or []:
        f = t.get("function", t) if isinstance(t, dict) else {}
        props = (f.get("parameters") or {}).get("properties") or {}
        types = {k: p.get("type") for k, p in props.items() if isinstance(p, dict)}
        out[f.get("name")] = {k for k, ty in types.items() if ty == "string" or (isinstance(ty, list) and "string" in ty)}
    return out


def _tool_calls(text, tools=None):
    """<parameter> values are plain text: a parameter the tool schema types as string stays text, any other value is
    read as JSON (3, true, null, [..], {..}) and stays text when it is not JSON."""
    strings = _string_params(tools)
    calls = []
    for m in _CALL.finditer(text):
        args = {}
        for k, v in _PARAM.findall(m.group(2)):
            v = v.strip()
            if k in strings.get(m.group(1), ()):
                args[k] = v
                continue
            try:
                args[k] = json.loads(v)
            except ValueError:
                args[k] = v
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    return calls


def split_turn(raw, thinking=True, truncated=False, tools=None):
    """A raw assistant turn -> (reasoning, visible text, [tool calls in OpenAI format]). A turn that never closed its
    thinking is all reasoning when max_tokens cut it (thinking on), else all text: some turns skip the close and
    answer anyway, and clients read the answer from the text."""
    if END_THINK in raw:
        reasoning, _, rest = raw.rpartition(END_THINK)
    elif thinking and truncated:
        reasoning, rest = raw, ""
    else:
        reasoning, rest = "", raw
    calls = _tool_calls(rest, tools)
    return reasoning.replace("<think>", "").strip(), _CALL.sub("", rest).strip(), calls


class TurnSplitter:
    """Streams a raw turn as (field, text) deltas: "reasoning_content" until </think>, then "content" until a
    <tool_call> starts. Holds back a tail that could be the start of a tag, so a tag split across chunks is never
    emitted as text. finish() settles the end the way split_turn reads the whole turn: the same tool calls, the text
    of a malformed <tool_call> kept as content, and a turn that never closed its thinking (and was not cut off) also
    sent as content, since split_turn reads that turn as the answer."""

    def __init__(self, thinking=True, tools=None):
        self.thinking, self.tools = thinking, tools
        self.field = "reasoning_content" if thinking else "content"
        self.buf, self.raw, self.calls_text, self.in_calls, self.lead = "", "", "", False, True

    def _emit(self, out, text):
        text = text.replace("<think>", "")
        if self.lead:                    # like split_turn's strip(): no whitespace at the start of a field
            text = text.lstrip()
            self.lead = not text
        if text:
            out.append((self.field, text))

    def feed(self, piece):
        self.raw += piece
        if self.in_calls:
            self.calls_text += piece
            return []
        self.buf += piece
        out = []
        while True:
            tag = END_THINK if self.field == "reasoning_content" else CALL_OPEN
            i = self.buf.find(tag)
            if i < 0:
                keep = max((k for k in range(1, len(tag)) if self.buf.endswith(tag[:k])), default=0)
                text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                self._emit(out, text)
                return out
            self._emit(out, self.buf[:i])
            self.buf = self.buf[i + len(tag):]
            if self.field == "reasoning_content":
                self.field, self.lead = "content", True
            else:
                self.in_calls, self.calls_text, self.buf = True, CALL_OPEN + self.buf, ""
                return out

    def finish(self, truncated=False):
        """-> (remaining deltas, tool calls). `truncated`: max_tokens cut the turn."""
        reasoning, text, calls = split_turn(self.raw, self.thinking, truncated, self.tools)
        out = []
        if self.in_calls:                # what is left of the <tool_call> text once the parsed calls are removed
            rest = _CALL.sub("", self.calls_text).rstrip()
            if rest.strip():
                out.append(("content", rest))
        elif self.buf:
            self._emit(out, self.buf.rstrip() if self.field == "content" else self.buf)
        if self.field == "reasoning_content" and text:   # thinking never closed: split_turn reads it as the answer
            out.append(("content", text))
        return out, calls


def _num(body, key, default, cast):
    """An optional numeric field: missing or null -> default (OpenAI clients send null for unset), else cast; a value
    that is not a number is the caller's error (400), not the server's."""
    v = body.get(key)
    if v is None:
        return default
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (cast is int and v != int(v)):
        from fastapi import HTTPException
        raise HTTPException(400, f"{key} must be {'an integer' if cast is int else 'a number'}, got {v!r}")
    return cast(v)


def create_app(engine):
    from fastapi import Body, FastAPI, HTTPException, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    from decide_request import RequestError

    name = os.path.basename(os.path.normpath(engine.ckpt))
    app = FastAPI(title="Trida decisions")

    @app.exception_handler(RequestError)
    async def request_error(_request, exc):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        if getattr(engine, "_failed", False):
            return JSONResponse(status_code=503, content={"ok": False, "model": name,
                                                         "error": "engine cleanup failed; restart required"})
        return {"ok": True, "model": name}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "trida"}]}

    @app.post("/v1/decide")
    async def decide(http_request: Request, body: dict = Body(...)):
        if "request" not in body:
            raise HTTPException(400, "body must be {\"request\": {...}, \"reasoning\"?: bool, \"temperature\"?: float}")
        try:
            # the image, if any, is in the request (base64 or a data URL): requests never name server files
            reasoning = body.get("reasoning", False)
            if not isinstance(reasoning, bool) and reasoning != "auto":
                raise RequestError('reasoning must be a boolean or "auto"')
            auto = getattr(engine, "supports_auto_reasoning", False)
            if reasoning == "auto" and not auto:
                raise RequestError('this backend does not support reasoning="auto"')
            options = {}
            if body.get("max_reasoning_tokens") is not None:
                budget = _num(body, "max_reasoning_tokens", 768, int)
                if budget < 1:
                    raise RequestError("max_reasoning_tokens must be positive")
                options["max_reasoning_tokens"] = budget
            if "threshold" in body:
                if not auto:
                    raise RequestError("this backend does not support threshold")
                options["threshold"] = _num(body, "threshold", 0.7, float)
            work = engine.adecide(body["request"], None, body.get("temperature"), reasoning, **options)
            if not getattr(engine, "cancel_disconnected_decisions", False):
                return await work
            task = asyncio.create_task(work)
            try:
                while not task.done():
                    await asyncio.wait({task}, timeout=0.05)
                    if not task.done() and await http_request.is_disconnected():
                        raise HTTPException(499, "client disconnected")
                return task.result()
            finally:
                if not task.done():
                    task.cancel()
                    with suppress(asyncio.CancelledError):
                        await task
        except RequestError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat(body: dict = Body(...)):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
        sampling = getattr(engine, "sampling", {"temperature": 0.6, "top_p": 0.95, "top_k": 20})
        kw = dict(tools=body.get("tools"), max_new_tokens=_num(body, "max_tokens", 2048, int),
                  temperature=_num(body, "temperature", sampling["temperature"], float), top_p=_num(body, "top_p", sampling["top_p"], float),
                  top_k=_num(body, "top_k", sampling["top_k"], int), presence_penalty=_num(body, "presence_penalty", 0.0, float),
                  enable_thinking=thinking)
        if hasattr(engine, "validate_chat"):
            engine.validate_chat(messages, **kw)
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())
        thinks = thinking is not False

        if body.get("stream"):
            async def events():
                def chunk(delta, finish=None, usage=None):
                    c = {"id": cid, "object": "chat.completion.chunk", "created": created, "model": name,
                         "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                    if usage:
                        c["usage"] = usage
                    return f"data: {json.dumps(c)}\n\n"
                yield chunk({"role": "assistant"})
                split, stats = TurnSplitter(thinks, kw["tools"]), {}
                async with aclosing(engine.achat_stream(messages, **kw)) as stream:
                    async for piece in stream:
                        if isinstance(piece, dict):
                            stats = piece
                            continue
                        for field, text in split.feed(piece):
                            yield chunk({field: text})
                rest, calls = split.finish(stats.get("finish_reason") == "length")
                for field, text in rest:
                    yield chunk({field: text})
                if calls:
                    yield chunk({"tool_calls": [{**c, "index": i} for i, c in enumerate(calls)]})
                finish = "tool_calls" if calls else stats.get("finish_reason", "stop")
                usage = {"prompt_tokens": stats.get("prompt_tokens", 0),
                         "completion_tokens": stats.get("completion_tokens", 0)}
                usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
                yield chunk({}, finish, usage)
                yield "data: [DONE]\n\n"
            return StreamingResponse(events(), media_type="text/event-stream")

        out = await engine.achat(messages, **kw)
        reasoning, text, calls = split_turn(out["text"], thinks, out["finish_reason"] == "length", kw["tools"])
        # content is a string (possibly empty) unless the turn is a tool call, as OpenAI's API returns it
        msg = {"role": "assistant", "content": (text or None) if calls else text, "reasoning_content": reasoning or None}
        if calls:
            msg["tool_calls"] = calls
        usage = {"prompt_tokens": out["prompt_tokens"], "completion_tokens": out["completion_tokens"],
                 "total_tokens": out["prompt_tokens"] + out["completion_tokens"]}
        return {"id": cid, "object": "chat.completion", "created": created, "model": name,
                "choices": [{"index": 0, "message": msg,
                             "finish_reason": "tool_calls" if calls else out["finish_reason"]}],
                "usage": usage}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("model", nargs="?", default=os.environ.get("TRIDA_DECIDE_MODEL"),
                    help="decision checkpoint directory (with jev_head.pt); default $TRIDA_DECIDE_MODEL")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=30001)
    ap.add_argument("--mem-fraction", type=float, default=None,
                    help="static memory fraction (default 0.8 for causal, 0.5 for self-spec)")
    ap.add_argument("--context-length", type=int, default=16384)
    ap.add_argument("--backend", choices=["causal", "self-spec"], default="causal",
                    help="self-spec: experimental text-only server; parity is not yet accepted")
    ap.add_argument("--self-spec-config", help="self-spec YAML; fixes sampling for every generation request")
    ap.add_argument("--decision-max-state", type=int, default=7552)
    ap.add_argument("--decision-max-row", type=int, default=8192)
    ap.add_argument("--reasoning-token-limit", type=int, default=4096,
                    help="shared reasoning budget cap (1..8192; default 4096)")
    a = ap.parse_args()
    if not a.model:
        ap.error("give a decision checkpoint (positional) or set TRIDA_DECIDE_MODEL")
    if a.mem_fraction is None:
        a.mem_fraction = 0.5 if a.backend == "self-spec" else 0.8
    import uvicorn

    if a.backend == "self-spec":
        from self_spec_decide_engine import SelfSpecDecideEngine
        engine = SelfSpecDecideEngine(a.model, mem_fraction_static=a.mem_fraction,
                                      context_length=a.context_length, config=a.self_spec_config,
                                      decision_max_state=a.decision_max_state, decision_max_row=a.decision_max_row,
                                      reasoning_token_limit=a.reasoning_token_limit)
    else:
        if a.self_spec_config:
            ap.error("--self-spec-config requires --backend self-spec")
        from decide_engine import DecideEngine
        engine = DecideEngine(a.model, mem_fraction_static=a.mem_fraction, context_length=a.context_length,
                              decision_max_state=a.decision_max_state, decision_max_row=a.decision_max_row,
                              reasoning_token_limit=a.reasoning_token_limit)
    # The HTTP server runs on the engine's own event loop, so concurrent requests batch in the engine.
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host=a.host, port=a.port, loop="none"))
    try:
        engine.loop.run_until_complete(server.serve())
    finally:
        engine.shutdown()


if __name__ == "__main__":   # sglang.Engine spawns its workers, which re-import this file
    main()
