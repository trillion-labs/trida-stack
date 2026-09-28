#!/usr/bin/env python3
"""nano-inference eval — reproduce benchmark accuracy on a served diffusion LLM.

  python eval.py gsm8k                          # full test set
  python eval.py mmlu_pro --num-problems 1000   # random subsample (seed 42)
  python eval.py gsm8k --port 30000 --max-workers 64

Points at ONE served endpoint (see serve.py). One run reports, from the same requests:
accuracy, aggregate tok/s, per-request tok/s, latency p50/p90, mean output length —
and, given the server's log (`--server-log`, see `serve.py --log`), tokens per forward.
Every request is written to `<out>/requests.jsonl`; the totals to `<out>/summary.json`.
Sampling defaults: temp 1.0 / top_p 0.95 / top_k 50 / no presence penalty — the canonical
self-spec setting, identical to the SGLang self-spec config (trida_self_spec_b7_g4.yaml) so both
backends decode the same way; `enable_thinking` is on. IFEval lives in eval_ifeval.py.

Adding a task = adding one entry to TASKS below (loader, prompt, extract, gold).
"""
import argparse
import json
import re
import statistics
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from datasets import load_dataset

LETTERS = "ABCDEFGHIJ"


# ---- gsm8k --------------------------------------------------------------
def gsm8k_load():
    ds = load_dataset("openai/gsm8k", "main", split="test")  # namespaced id: bare "gsm8k" breaks on huggingface_hub>=1.x
    return [(x["question"], x["answer"].split("####")[-1].strip().replace(",", "")) for x in ds]


def gsm8k_prompt(q):
    return f"{q}\nPlease reason step by step, and put your final answer within \\boxed{{}}."


def gsm8k_extract(c):
    boxed = re.findall(r"\\boxed\{([^}]+)\}", c)
    if boxed:
        return boxed[-1].replace(",", "").replace("$", "").replace("\\", "").strip()
    after = c.split("</think>")[-1] if "</think>" in c else c
    nums = re.findall(r"[\d,]+", after)
    return nums[-1].replace(",", "") if nums else "?"


# ---- mmlu_pro -----------------------------------------------------------
def mmlu_load():
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
    out = []
    for x in ds:
        choices = "\n".join(f"{LETTERS[i]}. {o}" for i, o in enumerate(x["options"]) if i < len(LETTERS))
        gold = LETTERS[x["answer_index"]] if isinstance(x["answer_index"], int) else str(x["answer"])
        out.append((f'{x["question"]}\n\n{choices}', gold))
    return out


def mmlu_prompt(qc):
    return f'{qc}\n\nThink step by step, then give your answer as "The answer is (X)".'


def mmlu_extract(c):
    after = re.sub(r"^.*?</think>\s*", "", c, count=1, flags=re.DOTALL) if "</think>" in c else c[-500:]
    for pat in (r"[Aa]nswer is:?\s*\*{0,2}\(?([A-Ja-j])\)?\*{0,2}",
                r"\\boxed\{[^}]*?([A-Ja-j])[^A-Ja-j}]*\}",
                r"[Aa]nswer:?\s*\(?([A-Ja-j])\)?"):
        m = re.search(pat, after)
        if m:
            return m.group(1).upper()
    return "?"


TASKS = {
    "gsm8k":    dict(load=gsm8k_load, prompt=gsm8k_prompt, extract=gsm8k_extract),
    "mmlu_pro": dict(load=mmlu_load,  prompt=mmlu_prompt,  extract=mmlu_extract),
}


def served_model_id(base):
    """The id the server exposes (vLLM rejects unknown model names; SGLang accepts any)."""
    try:
        return requests.get(f"{base}/v1/models", timeout=10).json()["data"][0]["id"]
    except Exception:
        return "default"


def run_one(base, model, spec, idx, item, args):
    """One request -> a record with correctness, token counts and wall-clock latency."""
    prompt_text, gold = item
    body = {
        "model": model,
        "messages": [{"role": "user", "content": spec["prompt"](prompt_text)}],
        "max_tokens": args.max_tokens, "temperature": args.temperature,
        "top_p": args.top_p, "top_k": args.top_k, "presence_penalty": args.presence_penalty,
        "chat_template_kwargs": {"enable_thinking": args.think},
    }
    rec = {"idx": idx, "gold": gold, "pred": None, "correct": False,
           "prompt_tokens": 0, "completion_tokens": 0, "latency_s": 0.0, "finish_reason": None, "error": None}
    t0 = time.perf_counter()
    try:
        r = requests.post(f"{base}/v1/chat/completions", json=body, timeout=args.timeout).json()
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        if "choices" not in r:  # server-side rejection (e.g. max_tokens > max-model-len): keep its message
            raise RuntimeError(str(r.get("error", r))[:300])
        c = r["choices"][0]["message"]["content"]
        rec["finish_reason"] = r["choices"][0].get("finish_reason")  # "length" = hit max_tokens
        rec["pred"] = spec["extract"](c)
        rec["correct"] = rec["pred"] == gold
        rec["prompt_tokens"] = r["usage"].get("prompt_tokens", 0)
        rec["completion_tokens"] = r["usage"]["completion_tokens"]
    except Exception as e:
        rec["latency_s"] = round(time.perf_counter() - t0, 3)
        rec["error"] = str(e)[:300]
    return rec


# ---- tokens per forward, from the server's own counters -----------------------
# Client-side timing cannot see forwards; the backends log them, cumulatively:
#   vLLM plugin       "DiffusionDecoding metrics: ... Committed: N tokens, Denoising steps: M"  (N excludes the bonus token)
#   SGLang self-spec  "[HybridDiffusionSelfSpec] ... fwd=N, ... tok/fwd=X ..."  (serve.py turns this on)
#   SGLang diffusion / causal: no counter is logged -> n/a (causal is 1 token per forward by definition)
# We snapshot the counters before and after the run and report the delta.
def forward_counters(log_path):
    """-> (committed_tokens, forwards, source) cumulative since server start, or None."""
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return None
    m = re.findall(r"Committed: (\d+) tokens, Denoising steps: (\d+)", text)
    if m:
        # "Committed" counts ACCEPTED DRAFTS only; every step also emits one bonus/recovery token,
        # so tokens = committed + steps (verified against per-request output counts: 135,680 emitted
        # over 59,451 steps with 76,234 committed).
        fwd = sum(int(b) for _, b in m)
        return sum(int(a) for a, _ in m) + fwd, fwd, "vLLM DiffusionDecoding counters (+1 bonus token per step)"
    m = re.findall(r"\[HybridDiffusionSelfSpec\][^\n]*?fwd=(\d+)[^\n]*?tok/fwd=([\d.]+)", text)
    if m:
        fwd, tpf = int(m[-1][0]), float(m[-1][1])
        return round(fwd * tpf), fwd, "SGLang HybridDiffusionSelfSpec stats"
    return None


def tokens_per_forward(before, after):
    if after is None:
        return {"tok_per_fwd": None, "source": "no forward counter in server log "
                "(SGLang causal/diffusion do not log one; causal AR is 1 token per forward)"}
    tok0, fwd0 = (before[0], before[1]) if before else (0, 0)
    tok, fwd = after[0] - tok0, after[1] - fwd0
    if fwd <= 0:
        return {"tok_per_fwd": None, "source": f"{after[2]}: counter did not advance during the run"}
    return {"tok_per_fwd": round(tok / fwd, 3), "forwards": fwd, "committed_tokens": tok, "source": after[2]}


def merge(dirs):
    """Combine shard result dirs (eval.py --shard) into one summary; wall/throughput are per-shard maxima/sums."""
    if not dirs:
        sys.exit("usage: eval.py merge <out-dir> [<out-dir> ...]   (dirs written by eval.py --shard)")
    recs, sums = [], []
    for d in dirs:
        recs += [json.loads(line) for line in open(Path(d) / "requests.jsonl")]
        sums.append(json.loads((Path(d) / "summary.json").read_text()))
    ok = [r for r in recs if r["error"] is None]
    n = len(recs); correct = sum(r["correct"] for r in recs)
    lat = sorted(r["latency_s"] for r in ok)
    q = lambda p: lat[min(int(p * len(lat)), len(lat) - 1)] if lat else 0.0
    wall = max(s["wall_s"] for s in sums)
    # A dead replica (all of its requests errored) is the failure mode sharding exists for, so every
    # aggregate below tolerates an empty `ok`/`lat` rather than taking the other shards down with it.
    per_req = [r["completion_tokens"] / r["latency_s"] for r in ok if r["latency_s"] > 0]
    out = {"task": sums[0]["task"], "shards": len(dirs), "n": n, "correct": correct,
           "accuracy_pct": round(correct / n * 100, 2) if n else 0.0, "errors": n - len(ok), "wall_s": wall,
           "tok_per_s_aggregate": round(sum(s["tok_per_s_aggregate"] for s in sums), 1),
           "tok_per_s_per_request": round(statistics.mean(per_req), 1) if per_req else 0.0,
           "latency_s": {"mean": round(statistics.mean(lat), 2) if lat else 0.0, "p50": q(0.5), "p90": q(0.9),
                         "max": lat[-1] if lat else 0.0},
           "completion_tokens_mean": round(sum(r["completion_tokens"] for r in ok) / max(len(ok), 1), 1),
           "truncated_pct": round(100 * sum(r.get("finish_reason") == "length" for r in ok) / max(len(ok), 1), 1),
           "settings": sums[0]["settings"]}
    fw = [s["forward"] for s in sums if s.get("forward", {}).get("tok_per_fwd")]
    if fw:
        tok = sum(f["committed_tokens"] for f in fw); f_ = sum(f["forwards"] for f in fw)
        out["forward"] = {"tok_per_fwd": round(tok / f_, 3), "forwards": f_, "committed_tokens": tok, "source": fw[0]["source"]}
    print(json.dumps(out, indent=1))
    return out


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "merge":
        merge(sys.argv[2:]); return
    ap = argparse.ArgumentParser(description="Reproduce benchmark accuracy on a served diffusion LLM.")
    ap.add_argument("task", choices=list(TASKS))
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--num-problems", type=int, default=0, help="0 = full set; else random subsample (seed 42)")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--presence-penalty", type=float, default=0.0)
    ap.add_argument("--no-think", dest="think", action="store_false")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--max-workers", type=int, default=64, help="client concurrency")
    ap.add_argument("--shard", default=None, metavar="I/N",
                    help="run only slice I of N of the problem set (0-based), e.g. 0/4 .. 3/4 against 4 replicas; "
                         "then `eval.py merge <out1> <out2> ...` combines the shards")
    ap.add_argument("--out", default=None, help="results dir (default results/<task>_<timestamp>)")
    ap.add_argument("--server-log", default=None,
                    help="server log (serve.py --log FILE): adds tokens-per-forward to the summary")
    args = ap.parse_args()

    spec = TASKS[args.task]
    items = spec["load"]()
    if 0 < args.num_problems < len(items):
        import random
        random.seed(42)
        items = [items[i] for i in random.sample(range(len(items)), args.num_problems)]
    if args.shard:
        try:
            i, n = (int(x) for x in args.shard.split("/"))
        except ValueError:
            ap.error(f"--shard must look like I/N (0-based), got {args.shard!r}")
        if n < 1 or not (0 <= i < n):
            ap.error(f"--shard I/N needs N >= 1 and 0 <= I < N (0-based), got {args.shard!r}; "
                     f"for 4 replicas use 0/4 1/4 2/4 3/4")
        items = items[i::n]
    if not items:
        ap.error("no problems selected (check --num-problems / --shard)")
    base = f"http://{args.host}:{args.port}"
    model = served_model_id(base)
    out = Path(args.out or f"results/{args.task}_{time.strftime('%Y%m%d-%H%M%S')}")
    out.mkdir(parents=True, exist_ok=True)
    print(f"{args.task}: {len(items)} problems -> {base} (model={model}, workers={args.max_workers}) -> {out}/", flush=True)

    counters0 = forward_counters(args.server_log) if args.server_log else None
    t0 = time.time()
    recs = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as pool, open(out / "requests.jsonl", "w") as fh:
        futs = [pool.submit(run_one, base, model, spec, i, it, args) for i, it in enumerate(items)]
        for i, f in enumerate(as_completed(futs), 1):
            rec = f.result(); recs.append(rec)
            fh.write(json.dumps(rec) + "\n")
            if i % max(len(items) // 10, 1) == 0:
                acc = sum(r["correct"] for r in recs) / i * 100
                print(f"  {i}/{len(items)}  acc={acc:.1f}%  ({time.time()-t0:.0f}s)", flush=True)

    el = time.time() - t0
    n = len(items)
    ok = [r for r in recs if r["error"] is None]
    correct = sum(r["correct"] for r in recs)
    tokens = sum(r["completion_tokens"] for r in ok)
    lat = sorted(r["latency_s"] for r in ok)
    q = lambda p: lat[min(int(p * len(lat)), len(lat) - 1)] if lat else 0.0
    summary = {
        "task": args.task, "endpoint": base, "model": model, "n": n, "correct": correct,
        "accuracy_pct": round(correct / n * 100, 2), "errors": n - len(ok),
        "wall_s": round(el, 1), "client_concurrency": args.max_workers,
        "tok_per_s_aggregate": round(tokens / el, 1),
        "tok_per_s_per_request": round(statistics.mean(r["completion_tokens"] / r["latency_s"]
                                                       for r in ok if r["latency_s"] > 0), 1) if ok else 0,
        "latency_s": {"mean": round(statistics.mean(lat), 2) if lat else 0, "p50": q(0.5), "p90": q(0.9), "max": lat[-1] if lat else 0},
        "completion_tokens_mean": round(tokens / max(len(ok), 1), 1),
        "truncated_pct": round(100 * sum(r["finish_reason"] == "length" for r in ok) / max(len(ok), 1), 1),
        "settings": {"max_tokens": args.max_tokens, "temperature": args.temperature, "top_p": args.top_p,
                     "top_k": args.top_k, "presence_penalty": args.presence_penalty, "thinking": args.think},
    }
    if args.server_log:
        time.sleep(1)  # let the backend flush its last stats line
        summary["forward"] = tokens_per_forward(counters0, forward_counters(args.server_log))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))

    L = summary["latency_s"]
    print(f"\n{args.task}: {correct}/{n} = {summary['accuracy_pct']:.2f}%   errors={summary['errors']}   wall={el:.0f}s")
    print(f"  throughput : {summary['tok_per_s_aggregate']:.0f} tok/s aggregate (C={args.max_workers}),"
          f" {summary['tok_per_s_per_request']:.0f} tok/s per request")
    print(f"  latency    : p50 {L['p50']:.2f}s  p90 {L['p90']:.2f}s  mean {L['mean']:.2f}s"
          f"   (mean output {summary['completion_tokens_mean']:.0f} tokens,"
          f" {summary['truncated_pct']:.1f}% hit max_tokens={args.max_tokens})")
    if "forward" in summary:
        fw = summary["forward"]
        print(f"  tok/forward: {fw['tok_per_fwd'] if fw['tok_per_fwd'] is not None else 'n/a'}   [{fw['source']}]")
    print(f"  -> {out}/summary.json, {out}/requests.jsonl")


if __name__ == "__main__":
    main()
