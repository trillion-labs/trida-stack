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

Needs stock SGLang >= 0.5.22 (Apache-2.0) plus fastapi and uvicorn in the same environment; see README "Decisions".
A decision checkpoint is a Trida checkpoint with jev_head.pt (MODEL_CONTRACT.md); an optional temperature.json sets
its serving temperatures. Generation here is causal decoding; the two-stream decode modes are serve.py's others.
"""
import argparse
import json
import os
import re
import time
import uuid

END_THINK, CALL_OPEN = "</think>", "<tool_call>"
_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def _tool_calls(text):
    calls = []
    for m in _CALL.finditer(text):
        args = {}
        for k, v in _PARAM.findall(m.group(2)):
            v = v.strip()
            try:
                args[k] = json.loads(v) if v[:1] in "[{" else v
            except ValueError:
                args[k] = v
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    return calls


def split_turn(raw, thinking=True, truncated=False):
    """A raw assistant turn -> (reasoning, visible text, [tool calls in OpenAI format]). A turn that never closed its
    thinking is all reasoning when max_tokens cut it (thinking on), else all text: some turns skip the close and
    answer anyway, and clients read the answer from the text."""
    if END_THINK in raw:
        reasoning, _, rest = raw.rpartition(END_THINK)
    elif thinking and truncated:
        reasoning, rest = raw, ""
    else:
        reasoning, rest = "", raw
    calls = _tool_calls(rest)
    return reasoning.replace("<think>", "").strip(), _CALL.sub("", rest).strip(), calls


class TurnSplitter:
    """Streams a raw turn as (field, text) deltas: "reasoning_content" until </think>, then "content" until a
    <tool_call> starts (tool calls are parsed from the whole turn at the end). Holds back a tail that could be the
    start of a tag, so a tag split across chunks is never emitted as text."""

    def __init__(self, thinking=True):
        self.field = "reasoning_content" if thinking else "content"
        self.buf, self.raw, self.in_calls = "", "", False

    def feed(self, piece):
        self.raw += piece
        self.buf += piece
        out = []
        while True:
            tag = END_THINK if self.field == "reasoning_content" else CALL_OPEN
            if self.in_calls:
                self.buf = ""
                return out
            i = self.buf.find(tag)
            if i < 0:
                keep = max((k for k in range(1, len(tag)) if self.buf.endswith(tag[:k])), default=0)
                text, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                text = text.replace("<think>", "")
                if text:
                    out.append((self.field, text))
                return out
            text = self.buf[:i].replace("<think>", "")
            if text:
                out.append((self.field, text))
            self.buf = self.buf[i + len(tag):]
            if self.field == "reasoning_content":
                self.field = "content"
                self.buf = self.buf.lstrip("\n")
            else:
                self.in_calls = True

    def finish(self):
        """-> (remaining deltas, tool calls)."""
        out = [] if self.in_calls or not self.buf else [(self.field, self.buf)]
        rest = self.raw.rpartition(END_THINK)[2] if END_THINK in self.raw else self.raw
        return out, _tool_calls(rest)


def create_app(engine):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    from decide_request import RequestError

    name = os.path.basename(os.path.normpath(engine.ckpt))
    app = FastAPI(title="Trida decisions")

    @app.get("/health")
    def health():
        return {"ok": True, "model": name}

    @app.get("/v1/models")
    def models():
        return {"object": "list", "data": [{"id": name, "object": "model", "created": 0, "owned_by": "trida"}]}

    @app.post("/v1/decide")
    async def decide(body: dict = Body(...)):
        if "request" not in body:
            raise HTTPException(400, "body must be {\"request\": {...}, \"reasoning\"?: bool, \"temperature\"?: float}")
        try:
            return await engine.adecide(body["request"], None, body.get("temperature"), bool(body.get("reasoning")))
        except RequestError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat(body: dict = Body(...)):
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        thinking = (body.get("chat_template_kwargs") or {}).get("enable_thinking")
        kw = dict(tools=body.get("tools"), max_new_tokens=int(body.get("max_tokens") or 2048),
                  temperature=float(body.get("temperature", 0.6)), top_p=float(body.get("top_p", 0.95)),
                  top_k=int(body.get("top_k", 20)), presence_penalty=float(body.get("presence_penalty", 0.0)),
                  enable_thinking=thinking)
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
                split, stats = TurnSplitter(thinks), {}
                async for piece in engine.achat_stream(messages, **kw):
                    if isinstance(piece, dict):
                        stats = piece
                        continue
                    for field, text in split.feed(piece):
                        yield chunk({field: text})
                rest, calls = split.finish()
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
        reasoning, text, calls = split_turn(out["text"], thinks, out["finish_reason"] == "length")
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
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--context-length", type=int, default=16384)
    a = ap.parse_args()
    if not a.model:
        ap.error("give a decision checkpoint (positional) or set TRIDA_DECIDE_MODEL")
    import uvicorn

    from decide_engine import DecideEngine
    engine = DecideEngine(a.model, mem_fraction_static=a.mem_fraction, context_length=a.context_length)
    # The HTTP server runs on the engine's own event loop, so concurrent requests batch in the engine.
    server = uvicorn.Server(uvicorn.Config(create_app(engine), host=a.host, port=a.port, loop="none"))
    engine.loop.run_until_complete(server.serve())


if __name__ == "__main__":   # sglang.Engine spawns its workers, which re-import this file
    main()
