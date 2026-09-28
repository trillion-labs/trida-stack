#!/usr/bin/env python3
"""Self-distillation data for the draft-aligned fine-tune: keep v6 prompts/history/tools, regenerate every assistant turn
with the model's own AR greedy path (teacher-forced on the original history), via vLLM /v1/completions on token ids
rendered with the trainer's own chat-template helper. Usage:
  python gen_selfdistill.py --src DIR --out shard.jsonl --ckpt CKPT --ports 30000 30001 ... --stride 17 --offset 0 --count 20000
"""
import argparse, asyncio, glob, json, os, sys, time
import aiohttp
sys.path.insert(0, os.environ.get("TRIDA_REPO", "$SCRATCH/code/trida-stack-main"))
from train.data.text_sft_data import _normalize_messages, _apply_template  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ckpt", required=True)
ap.add_argument("--ports", type=int, nargs="+", required=True); ap.add_argument("--model", default="trida-bd")
ap.add_argument("--stride", type=int, default=17); ap.add_argument("--offset", type=int, default=0); ap.add_argument("--count", type=int, default=20000)
ap.add_argument("--max_tokens", type=int, default=16384); ap.add_argument("--max_prompt", type=int, default=24000)
ap.add_argument("--ctx_len", type=int, default=32768, help="server --max-model-len; per-request max_tokens is clipped to ctx_len - prompt")
ap.add_argument("--per_port", type=int, default=16); ap.add_argument("--timeout", type=float, default=3600)
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.ckpt, trust_remote_code=True)
GEN_TAIL = "<|im_start|>assistant\n"

def select_rows():
    """Every `stride`-th conversation over all chunks (global index), then this shard's [offset, offset+count)."""
    k = 0; taken = 0
    for f in sorted(glob.glob(os.path.join(a.src, "chunk-*.jsonl"))):
        with open(f) as fh:
            for i, line in enumerate(fh):
                if (i % a.stride) != 0:
                    continue
                if k >= a.offset and taken < a.count:
                    taken += 1; yield json.loads(line)
                k += 1
                if taken >= a.count:
                    return

def render(messages, tools, i, gen):
    try:
        return _apply_template(tok, messages[:i] if gen else messages[:i + 1], tools, gen, keep_all_reasoning=True), tools
    except Exception:
        return _apply_template(tok, messages[:i] if gen else messages[:i + 1], None, gen, keep_all_reasoning=True), None

stats = {"conv": 0, "kept": 0, "turns": 0, "truncated": 0, "long_prompt": 0, "roundtrip_mismatch": 0, "errors": 0, "gen_tokens": 0}

async def gen_turn(sess, sem, port, prompt_ids):
    # budget: never exceed the server context (32768) — prompt + 16k used to error out on ~3% of long prompts
    body = {"model": a.model, "prompt": prompt_ids, "max_tokens": max(256, min(a.max_tokens, a.ctx_len - len(prompt_ids) - 16)), "temperature": 0, "skip_special_tokens": False, "return_token_ids": True}
    async with sem:
        async with sess.post(f"http://localhost:{port}/v1/completions", json=body, timeout=aiohttp.ClientTimeout(total=a.timeout)) as r:
            d = await r.json()
            if r.status != 200:
                raise RuntimeError(str(d)[:200])
            c = d["choices"][0]
            return c["text"], c.get("finish_reason"), c.get("token_ids") or d.get("token_ids")

async def process(sess, sems, row, rr):
    """Regenerate every assistant turn SEQUENTIALLY, each conditioned on the already-regenerated earlier turns
    (what the verifier sees at inference), teacher-forcing only the user/tool/system turns."""
    msgs = _normalize_messages(row.get("messages") or [])
    tools = row.get("tools")
    asst = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
    if not asst:
        return None
    new_msgs = [dict(m) for m in msgs]
    for i in asst:
        prompt_ids, tools_used = render(new_msgs, tools, i, True)
        if len(prompt_ids) > a.max_prompt:
            stats["long_prompt"] += 1; return None
        port = a.ports[rr[0] % len(a.ports)]; rr[0] += 1
        try:
            text, finish, gen_ids = await gen_turn(sess, sems[port], port, prompt_ids)
        except Exception as e:  # noqa: BLE001
            stats["errors"] += 1; stats.setdefault("error_samples", []);
            if len(stats["error_samples"]) < 5: stats["error_samples"].append(str(e)[:160])
            return None
        if finish == "length":
            stats["truncated"] += 1
            with open(a.out + ".truncated.jsonl", "a") as ft:   # inspect runaway generations later
                ft.write(json.dumps({"id": row.get("id"), "turn": i, "head": text[:300], "tail": text[-300:]}, ensure_ascii=False) + "\n")
            return None
        text = text.split("<|im_end|>")[0]
        content = "<think>\n" + text if not text.startswith("<think>") else text   # generation prompt ends with '<think>\n'
        new_msgs[i] = {"role": "assistant", "content": content}
        stats["turns"] += 1; stats["gen_tokens"] += len(gen_ids) if gen_ids else 0
        # round-trip against the SAME context: the re-rendered turn must start with prompt + generated ids
        try:
            full = _apply_template(tok, new_msgs[: i + 1], tools_used, False, keep_all_reasoning=True)
            expect = list(prompt_ids) + (list(gen_ids) if gen_ids else tok(text, add_special_tokens=False)["input_ids"])
            if full[: len(expect)] != expect:
                stats["roundtrip_mismatch"] += 1
        except Exception:
            stats["roundtrip_mismatch"] += 1
    out = dict(row); out["messages"] = new_msgs
    out["meta"] = dict(row.get("meta") or {}, generated={"ckpt": a.ckpt, "max_tokens": a.max_tokens, "greedy": True, "thinking": True, "sequential_history": True})
    return out

async def main():
    sems = {p: asyncio.Semaphore(a.per_port) for p in a.ports}
    rr = [0]; t0 = time.time(); pending = set()
    with open(a.out, "w") as fo:
        async with aiohttp.ClientSession() as sess:
            async def run_one(row):
                stats["conv"] += 1
                res = await process(sess, sems, row, rr)
                if res is not None:
                    stats["kept"] += 1; fo.write(json.dumps(res, ensure_ascii=False) + "\n")
                if stats["conv"] % 200 == 0:
                    print(f"[{time.time()-t0:7.0f}s] {stats}", file=sys.stderr, flush=True)
            for row in select_rows():
                while len(pending) >= len(a.ports) * a.per_port * 2:      # turns are sequential per conversation -> keep many conversations in flight
                    done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                pending.add(asyncio.create_task(run_one(row)))
            if pending:
                await asyncio.gather(*pending)
    print(json.dumps(stats), file=sys.stderr); json.dump(stats, open(a.out + ".stats.json", "w"))

asyncio.run(main())
