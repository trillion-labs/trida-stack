#!/usr/bin/env python3
"""Serve a Trida decision checkpoint: typed decisions and chat from one engine, one copy of the weights.

  POST /v1/decide            {"request": {"state": ..., "questions": {...}}, "reasoning": false, "temperature": null}
                             -> {"answers": {qid: answer}, "timing_ms": ...} (+ "reasoning" when reasoning=true)
  POST /v1/chat/completions  OpenAI-compatible, non-streaming; `tools` supported (tool calls parsed from the
                             model's <tool_call> blocks); the thinking is returned as message.reasoning_content
  GET  /health

Usage:
  python serve_decide.py /path/to/decision-checkpoint --port 30001     # or $TRIDA_DECIDE_MODEL

Needs stock SGLang >= 0.5.22 (Apache-2.0) plus fastapi and uvicorn in the same environment; see README "Decisions".
A decision checkpoint is a Trida checkpoint with jev_head.pt (MODEL_CONTRACT.md); an optional temperature.json sets
its serving temperatures. Generation here is causal decoding; the two-stream decode modes are serve.py's.
"""
import argparse
import json
import os
import re
import time
import uuid

_CALL = re.compile(r"<tool_call>\s*<function=([\w.-]+)>(.*?)</function>\s*</tool_call>", re.S)
_PARAM = re.compile(r"<parameter=([\w.-]+)>\n?(.*?)\n?</parameter>", re.S)


def split_turn(raw):
    """A raw assistant turn -> (reasoning, visible text, [tool calls in OpenAI format])."""
    reasoning, _, rest = raw.rpartition("</think>")
    calls = []
    for m in _CALL.finditer(rest):
        args = {}
        for k, v in _PARAM.findall(m.group(2)):
            v = v.strip()
            try:
                args[k] = json.loads(v) if v[:1] in "[{" else v
            except ValueError:
                args[k] = v
        calls.append({"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function",
                      "function": {"name": m.group(1), "arguments": json.dumps(args)}})
    text = _CALL.sub("", rest).strip()
    return reasoning.replace("<think>", "").strip(), text, calls


def create_app(engine):
    from fastapi import Body, FastAPI, HTTPException
    from fastapi.concurrency import run_in_threadpool

    from decide_request import RequestError

    app = FastAPI(title="Trida decisions")

    @app.get("/health")
    def health():
        return {"ok": True, "model": os.path.basename(os.path.normpath(engine.ckpt))}

    @app.post("/v1/decide")
    async def decide(body: dict = Body(...)):
        if "request" not in body:
            raise HTTPException(400, "body must be {\"request\": {...}, \"reasoning\"?: bool, \"temperature\"?: float}")
        try:
            return await run_in_threadpool(engine.decide, body["request"], None, body.get("temperature"),
                                           bool(body.get("reasoning")))
        except RequestError as e:
            raise HTTPException(400, str(e)) from e

    @app.post("/v1/chat/completions")
    async def chat(body: dict = Body(...)):
        if body.get("stream"):
            raise HTTPException(400, "streaming is not supported here; send stream=false")
        messages = body.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(400, "messages must be a non-empty list")
        raw = await run_in_threadpool(engine.chat, messages, body.get("tools"), int(body.get("max_tokens") or 2048),
                                      float(body.get("temperature", 0.6)))
        reasoning, text, calls = split_turn(raw)
        msg = {"role": "assistant", "content": text or None, "reasoning_content": reasoning or None}
        if calls:
            msg["tool_calls"] = calls
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(time.time()),
                "model": os.path.basename(os.path.normpath(engine.ckpt)),
                "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls" if calls else "stop"}]}

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
    uvicorn.run(create_app(engine), host=a.host, port=a.port)


if __name__ == "__main__":   # sglang.Engine spawns its workers, which re-import this file
    main()
