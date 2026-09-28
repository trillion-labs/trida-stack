#!/usr/bin/env python3
"""Summarize a DFlash comparison run dir: per engine job, tok/s per config x concurrency, speedup vs that engine's AR,
mean acceptance length (vLLM: parsed from the server log; SGLang: spec_verify_ct if reported). Writes summary_dflash.json."""
import glob, json, os, re, sys
RUN = sys.argv[1]; JOBS = sys.argv[2:] or ["sweep-vllm", "sweep-sglang"]
out = {"run": RUN, "jobs": {}}
for job in JOBS:
    d = f"{RUN}/{job}"
    if not os.path.isdir(d):
        continue
    res = {}
    for f in glob.glob(f"{d}/bench_*_c*.json"):
        m = re.match(r"bench_(.+)_c(\d+)\.json", os.path.basename(f)); b = json.load(open(f))
        res.setdefault(m.group(1), {})[int(m.group(2))] = {"tok_per_s": b["tok_per_s"], "n_ok": b["n_ok"], "n_err": b["n_err"], "accept": b.get("mean_accept_len")}
    acc = {}
    for f in glob.glob(f"{d}/serve_*.log"):
        n = os.path.basename(f)[6:-4]; txt = open(f, errors="ignore").read()
        v = re.findall(r"Mean acceptance length: ([\d.]+)", txt)            # vLLM: cumulative, take the last
        if v:
            acc[n] = float(v[-1]); continue
        v = [float(x) for x in re.findall(r"accept len: ([\d.]+)", txt)]    # SGLang: per decode batch, average
        if v:
            acc[n] = round(sum(v) / len(v), 2)
    out["jobs"][job] = {"results": res, "accept_len_serverlog": acc}
    ar = res.get("ar", {}); cs = sorted({c for v in res.values() for c in v})
    print(f"\n## {job}"); print("| config | " + " | ".join(f"C={c}" for c in cs) + " | accept len |"); print("|---|" + "---:|" * (len(cs) + 1))
    for n in sorted(res, key=lambda x: (x != "ar", x)):
        cells = []
        for c in cs:
            r = res[n].get(c)
            if not r: cells.append("—"); continue
            sp = f" ({r['tok_per_s']/ar[c]['tok_per_s']:.2f}x)" if c in ar and n != "ar" else ""
            err = f" err{r['n_err']}" if r["n_err"] else ""
            cells.append(f"{r['tok_per_s']:.0f}{sp}{err}")
        a = acc.get(n) or next((res[n][c]["accept"] for c in cs if res[n].get(c, {}).get("accept")), None)
        print(f"| {n} | " + " | ".join(cells) + f" | {a if a else '—'} |")
json.dump(out, open(f"{RUN}/summary_dflash.json", "w"), indent=1)
