#!/usr/bin/env python3
"""OpenAI-compatible server for the ORIGINAL Qwen3-4B with STANDARD autoregressive decoding.

This is the AR baseline for comparison against the block-diffusion checkpoint. It reuses the exact
same ChatML prompt formatting + <tool_call> parsing as serve_qwen3_bd_openai.py / serve_trida_openai.py
(so the benchmark harness + tool extraction are identical), and differs ONLY in decoding: plain
`model.generate(...)` instead of block-diffusion unmasking.

Usage:
    python serve_qwen3_ar_openai.py --model $SCRATCH/models/Qwen3-4B --device cuda:0 --port 8001
"""
import argparse
import json
import os
import sys
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference.bd_generate import load_tokenizer_compat  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_trida_openai import (  # noqa: E402
    _TOOL_CALL_RE, split_think, extract_tool_calls, format_chat_prompt,
    CompletionRequest, ChatCompletionRequest,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="original Qwen3-4B dir/id (AR baseline)")
    p.add_argument("--served-name", default=None)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--max-new-tokens-cap", type=int, default=1024)
    p.add_argument("--dtype", default="bfloat16")
    # accepted-but-ignored (pool.sh passes these; they're block-diffusion-only)
    p.add_argument("--block-size", type=int, default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--allow-eos", action="store_true")
    return p.parse_args()


def build_app(args):
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    served_name = args.served_name or args.model

    print(f"[serve-ar] loading {args.model} on {device} ({args.dtype})", flush=True)
    tokenizer = load_tokenizer_compat(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    gen_lock = threading.Lock()
    print(f"[serve-ar] ready: {served_name} on {device} | eos={eos_id} pad={pad_id}", flush=True)

    app = FastAPI()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "qwen"}]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @torch.no_grad()
    def _generate(prompt, max_tokens, temperature, top_p, stop):
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        prompt_len = input_ids.shape[1]
        max_new = max(1, min(max_tokens, args.max_new_tokens_cap))
        with gen_lock:
            out = model.generate(
                input_ids=input_ids, max_new_tokens=max_new,
                do_sample=temperature > 0, temperature=temperature if temperature > 0 else None,
                top_p=(top_p if top_p is not None else args.top_p) if temperature > 0 else None,
                eos_token_id=eos_id, pad_token_id=pad_id,
            )
        gen = out[0][prompt_len:]
        eos_pos = (gen == eos_id).nonzero()
        finish = "length"
        if eos_pos.numel():
            gen = gen[: eos_pos[0].item()]; finish = "stop"
        c_tok = int(gen.shape[0])
        text = tokenizer.decode(gen, skip_special_tokens=False)
        stops = [stop] if isinstance(stop, str) else (stop or [])
        cut = len(text)
        for s in stops:
            if s and (i := text.find(s)) != -1:
                cut = min(cut, i)
        if cut < len(text):
            text = text[:cut]; finish = "stop"
        return text, prompt_len, c_tok, finish

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        t0 = time.time()
        text, p_tok, c_tok, finish = _generate(req.prompt, req.max_tokens, req.temperature, req.top_p, req.stop)
        return {"id": "cmpl-qwen3ar", "object": "text_completion", "created": int(t0), "model": served_name,
                "choices": [{"text": text, "index": 0, "finish_reason": finish, "logprobs": None}],
                "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok}}

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        t0 = time.time()
        prompt = format_chat_prompt(req.messages, req.tools)
        max_tokens = req.max_tokens or req.max_completion_tokens or 512
        text, p_tok, c_tok, finish = _generate(prompt, max_tokens, req.temperature, req.top_p, req.stop)
        reasoning, content = split_think(text)
        calls = extract_tool_calls(text)
        message = {"role": "assistant"}
        if calls:
            nl = _TOOL_CALL_RE.sub("", content).replace("<tool_call>", "").replace("</tool_call>", "").strip()
            message["content"] = nl or None
            message["tool_calls"] = [
                {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                 "function": {"name": c["name"], "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in calls]
            finish = "tool_calls"
        else:
            message["content"] = content
        if reasoning:
            message["reasoning_content"] = reasoning
        return {"id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(t0),
                "model": served_name,
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok}}

    return app


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")
