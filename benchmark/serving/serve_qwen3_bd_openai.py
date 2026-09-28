#!/usr/bin/env python3
"""OpenAI-compatible server for a Qwen3-4B BLOCK-DIFFUSION checkpoint (trida hf_block_diffusion).

Unlike `serve_trida_openai.py` (which serves the released Trida-7B via its custom `trust_remote_code`
`model.generate(...)`), our training checkpoints are a STOCK `Qwen3ForCausalLM` (+ an added `<|mask|>`
token + a `block_diffusion.json` sidecar) with NO block-diffusion `generate`. So we load the stock
model with flex-attention and decode with `inference.bd_generate.block_diffusion_generate`,
which reuses the trainer's own forward + block-mask (Qwen3 QK-norm / RoPE / GQA correct by
construction). Endpoints, ChatML formatting, `<tool_call>` parsing, and OpenAI-`stop` truncation are
identical to `serve_trida_openai.py`, so the benchmarks run unchanged.

Usage:
    python serve_qwen3_bd_openai.py --model checkpoints/Qwen3-4B-32k-merged/step_4500 --device cuda:0 --port 8001
"""
import argparse
import json
import os
import re
import sys
import threading
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# make the repo root importable (for `inference.bd_generate`) regardless of CWD / venv
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from inference.bd_generate import (  # noqa: E402
    block_diffusion_generate_cached, block_speculative_generate_cached,
    block_diffusion_then_ar_generate, load_tokenizer_compat)

# --- chat helpers: reuse the verified ChatML formatter + tool-call parser from serve_trida_openai ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from serve_trida_openai import (  # noqa: E402
    _TOOL_CALL_RE, split_think, extract_tool_calls, format_chat_prompt,
    CompletionRequest, ChatCompletionRequest,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="local block-diffusion checkpoint dir")
    p.add_argument("--served-name", default=None, help="id reported by /v1/models (default: --model)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8001)
    p.add_argument("--block-size", type=int, default=None, help="bd_size (default: from block_diffusion.json)")
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--allow-eos", action="store_true",
                   help="allow EOS during unmasking (enables natural stopping; default masks EOS like "
                        "dInfer, since allowing it tends to blank out masked slots)")
    p.add_argument("--max-new-tokens-cap", type=int, default=1024)
    p.add_argument("--spec", action="store_true",
                   help="speculative decoding: KV-cached one-shot bd_size-token block draft (no "
                        "threshold) + causal AR verify (output == AR-greedy; "
                        "block_speculative_generate_cached). Default is pure block-diffusion.")
    p.add_argument("--hybrid", action="store_true",
                   help="hybrid decode: block-diffusion for the <think> reasoning, then greedy-AR after "
                        "</think> for the answer/tool-call (valid JSON). block_diffusion_then_ar_generate.")
    p.add_argument("--think-prefill", dest="think_prefill", action="store_true", default=True,
                   help="prefill '<think>\\n' onto the chat generation prompt (default): anchors the "
                        "structural start so the first diffusion block decodes reasoning, not the tag "
                        "(measurably better + more likely to stop cleanly).")
    p.add_argument("--no-think-prefill", dest="think_prefill", action="store_false")
    p.add_argument("--dtype", default="bfloat16")
    return p.parse_args()


def build_app(args):
    dtype = getattr(torch, args.dtype)
    device = torch.device(args.device)
    served_name = args.served_name or args.model

    # block-diffusion settings from the checkpoint sidecar
    bd_path = os.path.join(args.model, "block_diffusion.json")
    bd_cfg = {}
    if os.path.exists(bd_path):
        with open(bd_path) as f:
            bd_cfg = json.load(f)
    bd_size = args.block_size or int(bd_cfg.get("bd_size", 32))

    print(f"[serve] loading {args.model} on {device} ({args.dtype}) bd_size={bd_size}", flush=True)
    tokenizer = load_tokenizer_compat(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="eager"
    ).to(device).eval()

    # mask/eos/pad ids: prefer the sidecar mask_id, cross-check against the tokenizer's <|mask|>
    tok_mask = tokenizer.convert_tokens_to_ids("<|mask|>")
    mask_id = int(bd_cfg.get("mask_id", tok_mask))
    if tok_mask is not None and tok_mask != tokenizer.unk_token_id and mask_id != tok_mask:
        print(f"[serve] WARNING: block_diffusion.json mask_id={mask_id} != tokenizer <|mask|>={tok_mask}; "
              f"using {tok_mask}", flush=True)
        mask_id = tok_mask
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    think_end_id = tokenizer.convert_tokens_to_ids("</think>")  # for --hybrid handoff to AR
    gen_lock = threading.Lock()
    print(f"[serve] ready: {served_name} on {device} | mask_id={mask_id} eos_id={eos_id} pad_id={pad_id}",
          flush=True)

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
            if args.spec:   # KV-cached one-shot bd_size-token draft + causal AR verify (== AR-greedy)
                gen = block_speculative_generate_cached(
                    model, input_ids, max_new_tokens=max_new, mask_id=mask_id, eos_id=eos_id,
                    pad_id=pad_id, bd_size=bd_size)[0]
            elif args.hybrid:   # block-diffusion reasoning, then AR after </think> (valid structured out)
                gen = block_diffusion_then_ar_generate(
                    model, input_ids, gen_length=max_new, mask_id=mask_id, eos_id=eos_id, pad_id=pad_id,
                    bd_size=bd_size, threshold=args.threshold, think_end_id=think_end_id,
                    max_ar_tokens=max_new)[0]
            else:   # KV-cached block-diffusion (fix A: token-causal committed blocks + repeat-stop)
                gen = block_diffusion_generate_cached(
                    model, input_ids, gen_length=max_new, mask_id=mask_id, eos_id=eos_id, pad_id=pad_id,
                    bd_size=bd_size, threshold=args.threshold,
                    temperature=temperature, top_p=top_p if top_p is not None else args.top_p,
                    mask_eos_in_sampling=not args.allow_eos,
                )[0]
        # cut at the first EOS (present only if --allow-eos); keep tag tokens as text for parsing
        eos_pos = (gen == eos_id).nonzero()
        if eos_pos.numel() > 0:
            gen = gen[: eos_pos[0].item()]
            finish = "stop"
        else:
            finish = "length"
        c_tok = int(gen.shape[0])
        text = tokenizer.decode(gen, skip_special_tokens=False)
        # OpenAI `stop`: block-diffusion fills the whole block, so honor `stop` by truncating decoded text.
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
        prompt = req.prompt
        # BFCL etc. hit /v1/completions with a raw ChatML prompt ending at the assistant header; apply
        # the same <think>\n prefill as the chat route (anchors the reasoning start for block diffusion).
        if args.think_prefill and prompt.endswith("<|im_start|>assistant\n"):
            prompt += "<think>\n"
        text, p_tok, c_tok, finish = _generate(prompt, req.max_tokens, req.temperature, req.top_p, req.stop)
        return {
            "id": "cmpl-qwen3bd", "object": "text_completion", "created": int(t0), "model": served_name,
            "choices": [{"text": text, "index": 0, "finish_reason": finish, "logprobs": None}],
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        t0 = time.time()
        prompt = format_chat_prompt(req.messages, req.tools)
        if args.think_prefill:
            prompt += "<think>\n"   # anchor the reasoning start (see --think-prefill)
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
                for c in calls
            ]
            finish = "tool_calls"
        else:
            message["content"] = content
        if reasoning:
            message["reasoning_content"] = reasoning
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:24]}", "object": "chat.completion", "created": int(t0),
            "model": served_name,
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": p_tok, "completion_tokens": c_tok, "total_tokens": p_tok + c_tok},
        }

    return app


if __name__ == "__main__":
    args = parse_args()
    uvicorn.run(build_app(args), host=args.host, port=args.port, log_level="warning")
