#!/usr/bin/env python3
"""Replay recorded FunctionChat requests (model_request from a *.eval.jsonl) against one server, sequentially.
Reports per-request latency, prompt/completion tokens, and saves outputs for identity checks."""
import ast, json, sys, time, urllib.request, statistics as st
port, model, src, n, out = int(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4]), sys.argv[5]
rows = [json.loads(l) for l in open(src)][:n]
res = []
for i, r in enumerate(rows):
    mr = r["model_request"]
    msgs = mr["messages"] if isinstance(mr["messages"], list) else ast.literal_eval(mr["messages"])
    tools = mr["tools"] if isinstance(mr["tools"], list) else ast.literal_eval(mr["tools"])
    body = {"model": model, "messages": msgs, "tools": tools, "tool_choice": "auto", "temperature": 0.0, "max_tokens": 4096}
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(f"http://localhost:{port}/v1/chat/completions", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        d = json.load(urllib.request.urlopen(req, timeout=900)); dt = time.perf_counter() - t0
        m = d["choices"][0]["message"]; u = d.get("usage", {})
        res.append({"i": i, "lat_s": dt, "prompt_tokens": u.get("prompt_tokens"), "completion_tokens": u.get("completion_tokens"),
                    "tool_calls": [(c["function"]["name"], c["function"]["arguments"]) for c in (m.get("tool_calls") or [])], "content": (m.get("content") or "")[:400], "reasoning_len": len(m.get("reasoning") or m.get("reasoning_content") or "")})
    except Exception as e:
        res.append({"i": i, "error": str(e)[:200], "lat_s": time.perf_counter() - t0})
ok = [x for x in res if "error" not in x]
json.dump(res, open(out, "w"), ensure_ascii=False, indent=0)
if ok:
    lat = [x["lat_s"] for x in ok]; pt = [x["prompt_tokens"] or 0 for x in ok]; ct = [x["completion_tokens"] or 0 for x in ok]
    print(f"{model}: n={len(ok)} err={len(res)-len(ok)} wall={sum(lat):.1f}s lat mean {st.mean(lat):.2f}s p50 {st.median(lat):.2f}s max {max(lat):.2f}s | prompt tok mean {st.mean(pt):.0f} | completion tok mean {st.mean(ct):.0f} max {max(ct)} | gen tok/s {sum(ct)/sum(lat):.0f}")
else:
    print(f"{model}: all failed: {res[0].get('error')}")
