"""Cross-engine fixed-output throughput sweep (vLLM + SGLang via OpenAI /v1/chat/completions).
Mirrors HybridDiffusion bench_gsm8k_fixed_output.py: release GSM8K zero-shot prompt, ignore_eos so
every request generates exactly max_tokens, strict concurrency waves. Not a quality benchmark."""
import argparse, asyncio, json, time, statistics
import aiohttp
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, required=True)
ap.add_argument("--model", default="trida-bd")
ap.add_argument("--limit", type=int, default=64)
ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--max_tokens", type=int, default=512)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--save", required=True)
ap.add_argument("--timeout", type=float, default=3600)
ap.add_argument("--no_sampling", action="store_true", help="omit temperature/top_p (vLLM diffusion rejects them)")
ap.add_argument("--show", type=int, default=0, help="print the first N generated texts (smoke sanity)")
a = ap.parse_args()

ds = load_dataset("gsm8k", "main", split="test")
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
qs = [ds[i]["question"] + SUFFIX for i in range(a.limit + a.warmup)]
URL = f"http://localhost:{a.port}/v1/chat/completions"


async def one(sess, q):
    body = {"model": a.model, "messages": [{"role": "user", "content": q}], "max_tokens": a.max_tokens,
            "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}}
    if not a.no_sampling:
        body.update({"temperature": 0, "top_p": 1.0})
    t0 = time.perf_counter()
    try:
        async with sess.post(URL, json=body, timeout=aiohttp.ClientTimeout(total=a.timeout)) as r:
            d = await r.json()
            status = r.status
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "s": time.perf_counter() - t0, "tok": 0, "err": repr(e)[:200]}
    dt = time.perf_counter() - t0
    if status != 200 or "usage" not in d:
        return {"ok": False, "s": dt, "tok": 0, "err": str(d)[:200]}
    # speculative-decoding counters when the server reports them (SGLang: spec_verify_ct in usage/meta)
    def find(o, key):
        if isinstance(o, dict):
            if key in o:
                return o[key]
            for v in o.values():
                r = find(v, key)
                if r is not None:
                    return r
        elif isinstance(o, list):
            for v in o:
                r = find(v, key)
                if r is not None:
                    return r
        return None
    return {"ok": True, "s": dt, "tok": d["usage"]["completion_tokens"], "spec_verify_ct": find(d, "spec_verify_ct"),
            "text": (d["choices"][0]["message"].get("content") or "")[:300] if a.show else None}


async def main():
    async with aiohttp.ClientSession() as sess:
        if a.warmup:
            await asyncio.gather(*[one(sess, q) for q in qs[:a.warmup]])
        work = qs[a.warmup:]
        res = []
        T0 = time.perf_counter()
        for i in range(0, len(work), a.concurrency):
            res += await asyncio.gather(*[one(sess, q) for q in work[i:i + a.concurrency]])
        wall = time.perf_counter() - T0
    ok = [r for r in res if r["ok"]]
    tok = sum(r["tok"] for r in ok)
    out = {"port": a.port, "concurrency": a.concurrency, "limit": a.limit, "max_tokens": a.max_tokens,
           "n_ok": len(ok), "n_err": len(res) - len(ok), "wall_s": round(wall, 3), "tokens": tok,
           "tok_per_s": round(tok / wall, 2) if wall else 0,
           "per_req_tok_per_s": round(statistics.mean(r["tok"] / r["s"] for r in ok), 2) if ok else 0,
           "p50_latency_s": round(statistics.median(r["s"] for r in ok), 3) if ok else 0,
           "errors": [r["err"] for r in res if not r["ok"]][:3]}
    ct = [r["spec_verify_ct"] for r in ok if r.get("spec_verify_ct")]
    if ct:
        out["mean_accept_len"] = round(sum(r["tok"] for r in ok if r.get("spec_verify_ct")) / sum(ct), 3)
    if a.show:
        out["samples"] = [r["text"] for r in ok[:a.show]]
    json.dump(out, open(a.save, "w"), indent=1)
    print(json.dumps(out))


asyncio.run(main())
