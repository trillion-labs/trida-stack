"""Benchmark causal vs self-spec on device, and check that greedy self-spec is lossless.

    python -m trida_mlx.bench --model ./Trida2.0-4B-mlx-q8                # built-in prompts
    python -m trida_mlx.bench --model ./Trida2.0-4B-mlx-q8 --temperature 1.0 --repeats 3
    python -m trida_mlx.bench --model ./Trida2.0-4B-mlx-q8 --prompts my_prompts.jsonl --out results.json

Reports per prompt and in aggregate: decode tok/s, tokens per forward, accept histogram,
prefill time, and (greedy) whether self-spec output == causal output token for token.
Prompts file: JSONL with {"messages": [...]} or {"prompt": "..."} (+ optional "tools").
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import time

import mlx.core as mx

from .decode import DecodeStats, SamplingParams
from .engine import Engine
from .server import _normalize_messages

BUILTIN = [
    {"name": "gsm8k", "messages": [{"role": "user", "content":
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
        "How many clips did Natalia sell altogether in April and May? Reason step by step and put the "
        "final answer after '####'."}]},
    {"name": "code", "messages": [{"role": "user", "content":
        "Write a Python function that returns the n-th Fibonacci number iteratively, with a docstring "
        "and two doctests."}]},
    {"name": "tool_call", "tools": [{"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"},
                                                          "unit": {"type": "string", "enum": ["c", "f"]}},
                       "required": ["city"]}}}],
     "messages": [{"role": "user", "content": "What's the weather in Seoul right now, in celsius?"}]},
    {"name": "korean", "messages": [{"role": "user", "content":
        "온디바이스 LLM의 장점과 단점을 각각 세 가지씩 설명해 주세요."}]},
]


def _hw() -> str:
    try:
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout.strip() or 0)
        return f"{chip}, {mem / 2**30:.0f}GB"
    except Exception:
        return platform.platform()


def run_one(eng: Engine, prompt_ids, mode, sp, max_tokens):
    eng.reset_cache()  # cold prompt for a fair prefill number
    st = DecodeStats()
    ids = []
    t = time.perf_counter()
    for chunk in eng.generate_ids(prompt_ids, max_tokens=max_tokens, sampling=sp, mode=mode, stats=st):
        ids.extend(chunk)
    wall = time.perf_counter() - t
    d = st.as_dict()
    d["wall_s"] = round(wall, 3)
    return ids, d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trillionlabs/Trida2.0-4B")
    ap.add_argument("--prompts", default=None)
    ap.add_argument("--modes", default="causal,self-spec")
    ap.add_argument("--gen-block", type=int, default=None, help="N (canvas 2N-1); default 4")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-fused-gdn", action="store_true", help="A/B: unfused canvas GDN (2 kernels + readout)")
    ap.add_argument("--skip-rows", action="store_true", help="A/B: LM head on rows 0..N-1 only at cold start")
    a = ap.parse_args(argv)

    eng = Engine(a.model, gen_block=a.gen_block, fused_gdn=not a.no_fused_gdn)
    eng.rt.skip_rows = a.skip_rows
    items = BUILTIN
    if a.prompts:
        items = [json.loads(l) for l in open(a.prompts) if l.strip()]
    modes = [m.strip() for m in a.modes.split(",") if m.strip()]
    print(f"hardware: {_hw()}   model: {a.model}   load {eng.load_s:.1f}s   mask_id={eng.mask_id}")
    print("warming up ...", flush=True)
    for m in modes:
        for _ in eng.generate_ids(eng.encode("<|im_start|>user\nhello<|im_end|>\n<|im_start|>assistant\n"),
                                  max_tokens=16, sampling=SamplingParams(temperature=0.0), mode=m):
            pass
    results = []
    for it in items:
        images = []
        if it.get("prompt"):
            text = it["prompt"]
        else:  # OpenAI-style messages (e.g. captured with server --log-requests), normalized like the server
            msgs, srcs = _normalize_messages(it["messages"], vision=eng.supports_vision)
            images = [eng.add_image(s) for s in srcs]
            text = eng.render(msgs, tools=it.get("tools"), enable_thinking=not a.no_think)
        pids = eng.encode(text, images)
        row = {"name": it.get("name", f"p{len(results)}"), "prompt_tokens": len(pids), "runs": {}}
        outs = {}
        for m in modes:
            runs = []
            for r in range(a.repeats):
                sp = SamplingParams(temperature=a.temperature, top_k=a.top_k, top_p=a.top_p,
                                    seed=None if a.temperature <= 0 else r)
                ids, d = run_one(eng, pids, m, sp, a.max_tokens)
                runs.append(d)
                outs.setdefault(m, ids)
            row["runs"][m] = runs
        if a.temperature <= 0 and "causal" in outs and "self-spec" in outs:
            A, B = outs["causal"], outs["self-spec"]
            n = min(len(A), len(B))
            first = next((i for i in range(n) if A[i] != B[i]), None if len(A) == len(B) else n)
            row["identical"] = first is None
            row["first_diff"] = first
        row["sample"] = eng.tokenizer.decode(outs[modes[-1]][:160])
        results.append(row)
        line = f"[{row['name']:>9}] prompt={len(pids):5d}"
        for m in modes:
            tps = statistics.mean(r["decode_tok_s"] for r in row["runs"][m])
            tpf = statistics.mean(r["tokens_per_forward"] for r in row["runs"][m])
            nt = statistics.mean(r["new_tokens"] for r in row["runs"][m])
            line += f" | {m}: {tps:6.1f} tok/s  {tpf:4.2f} tok/fwd  {nt:5.0f} tok"
        if "identical" in row:
            line += f" | lossless={'YES' if row['identical'] else 'NO@' + str(row['first_diff'])}"
        print(line, flush=True)

    agg = {}
    for m in modes:
        tok = sum(r["new_tokens"] for row in results for r in row["runs"][m])
        sec = sum(r["decode_s"] for row in results for r in row["runs"][m])
        fw = sum(r["forwards"] for row in results for r in row["runs"][m])
        agg[m] = {"decode_tok_s": round(tok / sec, 2) if sec else 0, "tokens_per_forward": round(tok / fw, 3) if fw else 0}
    if "causal" in agg and "self-spec" in agg and agg["causal"]["decode_tok_s"]:
        agg["speedup"] = round(agg["self-spec"]["decode_tok_s"] / agg["causal"]["decode_tok_s"], 3)
    print("aggregate:", json.dumps(agg))
    if a.out:
        with open(a.out, "w") as f:
            json.dump({"hardware": _hw(), "model": a.model, "args": vars(a), "aggregate": agg,
                       "results": results, "peak_memory_gb": round(mx.get_peak_memory() / 1e9, 2)}, f,
                      indent=2, ensure_ascii=False)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
