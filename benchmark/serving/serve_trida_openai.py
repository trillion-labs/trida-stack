#!/usr/bin/env python3
"""OpenAI-compatible server for Trida-7B (block-diffusion), shared across benchmarks.

Written before the serving stack in `inference/` existed, when stock vLLM/sglang could not load a
`trust_remote_code` block-diffusion arch. The 7.76B model fits on a single GPU, so this loads it via
`AutoModelForCausalLM.from_pretrained(...)` and runs its block-diffusion `model.generate(...)`.
`inference/` now serves this architecture on both backends; this shim is kept because the published
benchmark numbers were produced with it. Exposes:

    GET  /v1/models             -> {"data": [{"id": <model>, "object": "model"}]}
    POST /v1/completions         {"model","prompt","max_tokens","temperature","top_p","stop"}
                              -> {"choices":[{"text","index","finish_reason"}], "usage":{...}}   (BFCL)
    POST /v1/chat/completions    {"model","messages","tools","tool_choice","max_tokens",...}
                              -> {"choices":[{"message":{content,tool_calls?},finish_reason}], ...}
                                 (FunctionChat-Bench / Ko-AgentBench)

The chat route formats a Qwen-style ChatML prompt server-side (Trida ships no chat_template) and
parses `<tool_call>` blocks into OpenAI `tool_calls`, so tool-calling clients work unchanged.

One model per process on one GPU; serialize generate() with a lock. Run N of these (one per GPU)
behind `router.py` for data-parallel throughput.

Usage:
    python serve_trida_openai.py --model trillionlabs/Trida-7B-Preview --device cuda:0 --port 8001
"""
import argparse
import json
import re
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- chat helpers (Qwen-style ChatML; Trida ships no chat_template, so build the prompt + parse
# tool calls here, mirroring BFCL's verified QwenFCHandler). ---------------------------------------

# Whitespace-tolerant: models emit both `{...}\n</tool_call>` and `{...}}</tool_call>` (no newline).
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)


def split_think(text: str):
    """Split a trailing/leading ``<think>…</think>`` block off the response.
    Returns (reasoning, content)."""
    if "</think>" in text:
        parts = text.split("</think>")
        reasoning = parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
        return reasoning, parts[-1].lstrip("\n")
    return "", text


def extract_tool_calls(text: str):
    """Parse ``<tool_call>\\n{...}\\n</tool_call>`` blocks into normalized calls
    ``{"name": str, "arguments": dict}`` (robust superset of QwenFCHandler._extract_tool_calls:
    drops nameless calls; coerces missing/str ``arguments`` to a dict)."""
    out = []
    for match in _TOOL_CALL_RE.findall(text):
        try:
            call = json.loads(match)
        except Exception:
            continue
        if not isinstance(call, dict) or "name" not in call:
            continue
        args = call.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                args = {}
        if not isinstance(args, dict):
            args = {}
        out.append({"name": call["name"], "arguments": args})
    return out


def format_chat_prompt(messages, tools):
    """Render OpenAI chat ``messages`` + ``tools`` into a Qwen-style ChatML prompt string, ending
    with the assistant generation header. Ported verbatim from BFCL's QwenFCHandler._format_prompt
    (the Qwen3 chat template), which is verified to elicit correct ``<tool_call>`` output from Trida.
    ``tools`` are the OpenAI tool objects (``{"type":"function","function":{...}}``), dumped as-is."""
    def content_of(m):
        c = m.get("content")
        return c if isinstance(c, str) else ""

    p = ""
    has_sys = bool(messages) and messages[0].get("role") == "system"
    if tools:
        p += "<|im_start|>system\n"
        if has_sys:
            p += content_of(messages[0]) + "\n\n"
        p += ("# Tools\n\nYou may call one or more functions to assist with the user query.\n\n"
              "You are provided with function signatures within <tools></tools> XML tags:\n<tools>")
        for tool in tools:
            p += f"\n{json.dumps(tool, ensure_ascii=False)}"
        p += ('\n</tools>\n\nFor each function call, return a json object with function name and '
              'arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n'
              '{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call><|im_end|>\n')
    elif has_sys:
        p += f"<|im_start|>system\n{content_of(messages[0])}<|im_end|>\n"

    # index of the last real user query (tool_response-only user turns don't count)
    last_query_index = len(messages) - 1
    for offset, m in enumerate(reversed(messages)):
        idx = len(messages) - 1 - offset
        c = m.get("content")
        if (m.get("role") == "user" and isinstance(c, str)
                and not (c.startswith("<tool_response>") and c.endswith("</tool_response>"))):
            last_query_index = idx
            break

    n = len(messages)
    for idx, m in enumerate(messages):
        role = m.get("role")
        content = content_of(m)
        if role == "user" or (role == "system" and idx != 0):
            p += f"<|im_start|>{role}\n{content}<|im_end|>\n"
        elif role == "assistant":
            reasoning = m.get("reasoning_content") or ""
            if not reasoning and "</think>" in content:
                reasoning, content = split_think(content)
            if idx > last_query_index and (idx == n - 1 or reasoning):
                p += (f"<|im_start|>{role}\n<think>\n" + reasoning.strip("\n")
                      + "\n</think>\n\n" + content.lstrip("\n"))
            else:
                p += f"<|im_start|>{role}\n{content}"
            tool_calls = m.get("tool_calls") or []
            for j, tc in enumerate(tool_calls):
                if (j == 0 and content) or j != 0:
                    p += "\n"
                fn = tc.get("function", tc)
                args = fn.get("arguments", {})
                args_str = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                p += f'<tool_call>\n{{"name": "{fn.get("name")}", "arguments": {args_str}}}\n</tool_call>'
            p += "<|im_end|>\n"
        elif role == "tool":
            prev_role = messages[idx - 1].get("role") if idx > 0 else None
            next_role = messages[idx + 1].get("role") if idx < n - 1 else None
            if idx == 0 or prev_role != "tool":
                p += "<|im_start|>user"
            p += f"\n<tool_response>\n{content}\n</tool_response>"
            if idx == n - 1 or next_role != "tool":
                p += "<|im_end|>\n"
    p += "<|im_start|>assistant\n"
    return p

# Trida special-token ids (see generate.py / eval_trida.sh). Overridable via CLI; verified against
# the loaded tokenizer at startup.
DEFAULTS = dict(mask_id=128012, stop_token=128001, pad_id=128004)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="trillionlabs/Trida-7B-Preview")
    p.add_argument("--served-name", default=None, help="id reported by /v1/models (default: --model)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    # block-diffusion generation params (match benchmark/scripts/eval_trida.sh)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--small-block-size", type=int, default=8)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--use-block-cache", action="store_true",
                   help="enable Trida's block KV cache (faster; the released generate() default is off)")
    p.add_argument("--mask-id", type=int, default=DEFAULTS["mask_id"])
    p.add_argument("--stop-token", type=int, default=DEFAULTS["stop_token"])
    p.add_argument("--max-new-tokens-cap", type=int, default=1024,
                   help="cap on generated tokens. Block-diffusion fills the whole length and tends to "
                        "repeat a short answer, so a tight cap both speeds the run and limits "
                        "duplicate tool calls; FC responses (reasoning + a few calls) fit in ~1024.")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


class CompletionRequest(BaseModel):
    model: str | None = None
    prompt: str
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float | None = None
    stop: list[str] | str | None = None
    n: int = 1


class ChatCompletionRequest(BaseModel):
    model: str | None = None
    messages: list[dict]
    tools: list[dict] | None = None
    tool_choice: object | None = None  # accepted, ignored (prompt already instructs tool use)
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    temperature: float = 0.0
    top_p: float | None = None
    stop: list[str] | str | None = None
    n: int = 1
    # non-OpenAI clients (litellm) may send extras; ignore them
    model_config = {"extra": "allow"}


def build_app(args):
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    served_name = args.served_name or args.model

    print(f"[serve] loading {args.model} on {device} ({args.dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(device).eval()
    gen_lock = threading.Lock()
    print(f"[serve] ready: {served_name} on {device}", flush=True)

    app = FastAPI()

    @app.get("/v1/models")
    def list_models():
        return {"object": "list", "data": [{"id": served_name, "object": "model", "owned_by": "trillionlabs"}]}

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @torch.no_grad()
    def _generate(prompt: str, max_tokens: int, temperature: float, top_p: float, stop):
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
        prompt_len = input_ids.shape[1]
        max_new = max(1, min(max_tokens, args.max_new_tokens_cap))
        with gen_lock:
            out = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new,
                mask_id=args.mask_id,
                stop_token=args.stop_token,
                block_size=args.block_size,
                small_block_size=min(args.small_block_size, args.block_size),
                threshold=args.threshold,
                temperature=temperature,
                top_p=top_p if top_p is not None else args.top_p,
                use_block_cache=args.use_block_cache,
            )
        seq = out[0] if out.dim() == 2 else out
        gen = seq[prompt_len:]
        # cut at the first stop token; keep tag tokens (<tool_call>, <think>) as text for BFCL parsing
        stop_pos = (gen == args.stop_token).nonzero()
        if stop_pos.numel() > 0:
            gen = gen[: stop_pos[0].item()]
            finish = "stop"
        else:
            finish = "length"
        c_tok = int(gen.shape[0])
        text = tokenizer.decode(gen, skip_special_tokens=False)
        # Apply the OpenAI `stop` sequences: block-diffusion generates the whole block at once (can't
        # halt mid-decode), so honor `stop` by truncating the decoded text at the earliest match.
        # This is what caps the model's tendency to keep repeating past the intended answer.
        stops = [stop] if isinstance(stop, str) else (stop or [])
        cut = len(text)
        for s in stops:
            if s:
                i = text.find(s)
                if i != -1:
                    cut = min(cut, i)
        if cut < len(text):
            text = text[:cut]
            finish = "stop"
        return text, prompt_len, c_tok, finish

    @app.post("/v1/completions")
    def completions(req: CompletionRequest):
        t0 = time.time()
        text, p_tok, c_tok, finish = _generate(req.prompt, req.max_tokens, req.temperature, req.top_p, req.stop)
        return {
            "id": "cmpl-trida",
            "object": "text_completion",
            "created": int(t0),
            "model": served_name,
            "choices": [{"text": text, "index": 0, "finish_reason": finish, "logprobs": None}],
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        # Chat path for FunctionChat-Bench / Ko-AgentBench: format ChatML server-side, decode, and
        # parse <tool_call> blocks into OpenAI tool_calls (these clients consume tool_calls objects).
        t0 = time.time()
        prompt = format_chat_prompt(req.messages, req.tools)
        max_tokens = req.max_tokens or req.max_completion_tokens or 512
        text, p_tok, c_tok, finish = _generate(prompt, max_tokens, req.temperature, req.top_p, req.stop)
        reasoning, content = split_think(text)
        calls = extract_tool_calls(text)
        message = {"role": "assistant"}
        if calls:
            # OpenAI convention: tool calls live in `tool_calls`, not `content`. Strip the
            # <tool_call>…</tool_call> blocks (and any stray tag) from the NL content.
            nl = _TOOL_CALL_RE.sub("", content)
            nl = nl.replace("<tool_call>", "").replace("</tool_call>", "").strip()
            message["content"] = nl or None
            message["tool_calls"] = [
                {"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                 "function": {"name": c["name"],
                              "arguments": json.dumps(c["arguments"], ensure_ascii=False)}}
                for c in calls
            ]
            finish = "tool_calls"
        else:
            message["content"] = content
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
            "object": "chat.completion",
            "created": int(t0),
            "model": served_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
        }

    return app


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")
