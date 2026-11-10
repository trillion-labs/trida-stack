"""Analyze a comprehensive run directory -> REPORT.md + summary.json.

Inputs (per job subdir <run>/<job>/): job.json, job_end.json, eval.log, gsm8k_summary_*.json,
gsm8k_details_*.json, dllm_stats_{before,after}.json (sglang), trace_rep*.jsonl (vllm trace),
serve_rep*.log ([TIME] lines for the timers job), <run>/sweep/bench_*_c*.json.
"""
import glob, gzip, json, math, os, re, statistics as st, sys, collections

RUN = sys.argv[1]
THR = 0.90
out = {"run": RUN}
lines = []
P = lines.append


def load(p):
    with open(p) as f:
        return json.load(f)


def ci95(p, n):
    return 196 * math.sqrt(max(p * (1 - p), 0) / max(n, 1))


# ---------------------------------------------------------------- accuracy / TPS matrix
P("# Comprehensive run report\n")
man = load(os.path.join(RUN, "manifest.json")) if os.path.exists(os.path.join(RUN, "manifest.json")) else {}
out["manifest"] = man
import datetime as _dt; out["generated"] = _dt.datetime.now().isoformat(timespec="minutes")
out["jobs_present"] = sorted(os.path.basename(os.path.dirname(x)) for x in glob.glob(os.path.join(RUN, "*/job.json")))
out["jobs_done"] = sorted(os.path.basename(os.path.dirname(x)) for x in glob.glob(os.path.join(RUN, "*/job_end.json")))
out["sweep_done"] = os.path.exists(os.path.join(RUN, "sweep")) and bool(glob.glob(os.path.join(RUN, "sweep/bench_*_c*.json")))
P(f"run dir `{RUN}`  ·  git `{man.get('git_sha','?')}`  ·  code md5 `{man.get('code_md5','?')[:8]}`  ·  "
  f"vLLM {man.get('vllm_config',{})}  ·  eval: {man.get('eval','?')}\n")
P("## 1. Accuracy and throughput (GSM8K, greedy, no-think, max 1024 tok)\n")
P("| job | engine/mode | replicas | n | acc | ±95% | per-GPU tok/s | avg tok | hit max | errs | wall min |")
P("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
acc = {}
import re as _re
_groups = collections.defaultdict(list)
for jd in sorted(glob.glob(os.path.join(RUN, "*/job.json"))):
    _groups[_re.sub(r"-n\d+$", "", load(jd)["job"])].append(jd)
for name, jds in sorted(_groups.items()):
    job = load(jds[0]); dirs = [os.path.dirname(x) for x in jds]
    sums = [load(f) for d0 in dirs for f in glob.glob(os.path.join(d0, "gsm8k_summary_*.json"))]
    if not sums:
        P(f"| {name} | {job['engine']}/{job['mode']} | {job['nrep']} | — | (no result) | | | | | | |"); continue
    s = {"total": sum(x["total"] for x in sums), "correct": sum(x["correct"] for x in sums), "total_tok": sum(x["total_tok"] for x in sums),
         "wall_seconds": max(x["wall_seconds"] for x in sums), "length_limited": sum(x["length_limited"] for x in sums), "errors": sum(x["errors"] for x in sums)}
    n = s["total"]; a = s["correct"] / max(n, 1)
    nrep = sum(len(load(os.path.join(d0, "job_end.json"))["healthy_ports"].split()) if os.path.exists(os.path.join(d0, "job_end.json")) else job["nrep"] for d0 in dirs)
    d = dirs[0]
    tps = s["total_tok"] / s["wall_seconds"] / max(nrep, 1)
    acc[name] = {"engine": job["engine"], "mode": job["mode"], "trace": job["trace"], "n": n, "acc": a, "ci": ci95(a, n) / 100,
                 "per_gpu_tok_s": tps, "avg_tok": s["total_tok"] / max(n, 1), "hit_max": s["length_limited"], "errors": s["errors"],
                 "wall_s": s["wall_seconds"], "nrep": nrep, "cl": job.get("cl", 4), "thresh": job.get("thresh", 0.9)}
    P(f"| {name} | {job['engine']}/{job['mode']} | {nrep} | {n} | {100*a:.1f}% | {ci95(a,n):.1f} | {tps:.1f} | "
      f"{s['total_tok']/max(n,1):.0f} | {s['length_limited']} | {s['errors']} | {s['wall_seconds']/60:.1f} |")
out["accuracy"] = acc
_ref = os.environ.get("REF_DETAILS")
if _ref and os.path.exists(_ref):
    _refd = {x["index"]: x for x in load(_ref)}; out["identity_vs_ref"] = {}
    for name in acc:
        fs = glob.glob(os.path.join(RUN, name, "gsm8k_details_*.json")) + glob.glob(os.path.join(RUN, name + "-n*", "gsm8k_details_*.json"))
        dd = {x["index"]: x for f in fs for x in load(f)}; sh = [i for i in dd if i in _refd]
        if sh:
            same = sum(dd[i]["generation"] == _refd[i]["generation"] for i in sh); ra = sum(_refd[i]["correct"] for i in sh) / len(sh)
            out["identity_vs_ref"][name] = {"n": len(sh), "identical": same, "ref_acc": ra}
            P(f"identity vs reference ({os.path.basename(_ref)}): {name}: {same}/{len(sh)} identical, ref acc on shared {100*ra:.1f}%\n")
P("\nper-GPU tok/s = total completion tokens / wall / healthy replicas (one request in flight per replica). "
  "Trace/timer jobs carry diagnostic syncs; use the *clean* rows for speed.\n")

# ---------------------------------------------------------------- paired accuracy on shared items
P("## 2. Paired item-level agreement (clean jobs)\n")
det = {}
for name in acc:
    fs = glob.glob(os.path.join(RUN, name, "gsm8k_details_*.json")) + glob.glob(os.path.join(RUN, name + "-n*", "gsm8k_details_*.json"))
    if fs:
        det[name] = {x["index"]: x for f in fs for x in load(f)}
names = [n for n in det if acc[n]["trace"] == "none"]
if len(names) >= 2:
    P("| A | B | shared n | both right | A only | B only | both wrong |"); P("|---|---|---:|---:|---:|---:|---:|")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            A, B = names[i], names[j]; sh = set(det[A]) & set(det[B])
            bb = sum(det[A][k]["correct"] and det[B][k]["correct"] for k in sh)
            ao = sum(det[A][k]["correct"] and not det[B][k]["correct"] for k in sh)
            bo = sum(det[B][k]["correct"] and not det[A][k]["correct"] for k in sh)
            P(f"| {A} | {B} | {len(sh)} | {bb} | {ao} | {bo} | {len(sh)-bb-ao-bo} |")
P("")

# ---------------------------------------------------------------- tokens per forward
P("## 3. Tokens per forward (decode: denoise + commit forwards)\n")
tpf = {}
for d in sorted(glob.glob(os.path.join(RUN, "*/"))):
    name = _re.sub(r"-n\d+$", "", os.path.basename(d.rstrip("/")))
    b, a = os.path.join(d, "dllm_stats_before.json"), os.path.join(d, "dllm_stats_after.json")
    if os.path.exists(b) and os.path.exists(a):
        B, A = load(b), load(a); tok = fw = pf = 0
        for p in A:
            if p == "_" or not isinstance(A[p], dict) or not A[p]:
                continue
            tok += A[p].get("total_tokens", 0) - B[p].get("total_tokens", 0)
            fw += A[p].get("decode_forwards", 0) - B[p].get("decode_forwards", 0)
            pf += A[p].get("prefill_forwards", 0) - B[p].get("prefill_forwards", 0)
        if fw:
            prev = tpf.get(name, {"tokens": 0, "decode_fwd": 0, "prefill_fwd": 0})
            tok += prev["tokens"]; fw += prev["decode_fwd"]; pf += prev["prefill_fwd"]
            tpf[name] = {"tokens": tok, "decode_fwd": fw, "prefill_fwd": pf, "tok_per_fwd": tok / fw}
    tr = sorted(glob.glob(os.path.join(d, "trace_rep*.jsonl")))
    if tr:
        recs = [json.loads(l) for f in tr for l in open(f) if l.strip()]
        tok = sum(r["tokens"] for r in recs); fw = sum(r["decode_fwd"] for r in recs)
        if fw:
            prev = tpf.get(name);
            if prev and "n_req" in prev:
                tok += prev["tokens"]; fw += prev["decode_fwd"]; nreq = prev["n_req"] + len(recs); pfw = prev["prefill_fwd"] + sum(r["prefill_fwd"] for r in recs)
            else:
                nreq = len(recs); pfw = sum(r["prefill_fwd"] for r in recs)
            tpf[name] = {"tokens": tok, "decode_fwd": fw, "prefill_fwd": pfw, "tok_per_fwd": tok / fw, "n_req": nreq}
P("| job | tokens | decode fwds | tok/fwd | note |"); P("|---|---:|---:|---:|---|")
for k, v in tpf.items():
    P(f"| {k} | {v['tokens']} | {v['decode_fwd']} | **{v['tok_per_fwd']:.3f}** | {'vLLM trace (per-request)' if 'n_req' in v else 'SGLang dllm_stats deltas'} |")
out["tok_per_fwd"] = tpf
P("\nAR = 1.0. Speed-up vs AR at equal per-forward cost = tok/fwd; actual = tok/fwd × (AR step ms / diffusion step ms).\n")

# ---------------------------------------------------------------- traces: rounds, calibration, position, knobs
P("## 4. Decode traces (vLLM trace servers)\n")
_tgroups = collections.defaultdict(list)
for f in glob.glob(os.path.join(RUN, "*/trace_rep*.jsonl")):
    _tgroups[_re.sub(r"-n\d+$", "", os.path.basename(os.path.dirname(f)))].append(f)
out["traces"] = {}
for _ti, (_job, traces) in enumerate(sorted(_tgroups.items())):
    traces = sorted(traces)
    recs = [json.loads(l) for f in traces for l in open(f) if l.strip()]
    if not recs:
        continue
    CLj = max((len(b["ids"]) for r in recs for b in r["blocks"]), default=4)
    T = {"job": _job, "cl": CLj}
    P(f"### Trace: {_job} (CL={CLj})\n")
    recs = [json.loads(l) for f in traces for l in open(f) if l.strip()]
    P(f"{len(recs)} requests, {sum(r['n_blocks'] for r in recs)} blocks\n")
    rounds = collections.Counter(b["rounds"] for r in recs for b in r["blocks"])
    tot = sum(rounds.values())
    P(f"**Denoise rounds per block** (CL={CLj} → {CLj-1} masks):\n")
    P("| rounds | blocks | share |"); P("|---:|---:|---:|")
    for k in sorted(rounds):
        P(f"| {k} | {rounds[k]} | {100*rounds[k]/tot:.1f}% |")
    mean_rounds = sum(k * v for k, v in rounds.items()) / tot
    is_spec = any("spec_steps" in r for r in recs)
    if is_spec:
        acc_h = collections.Counter(a for r in recs for _, a in r.get("spec_steps", []))
        tot_s = sum(acc_h.values()); T["spec_accept_hist"] = dict(acc_h); T["is_spec"] = True
        P("**Self-spec (AR-Trust): accepted specs per forward** " + ", ".join(f"{k}: {100*v/tot_s:.1f}%" for k, v in sorted(acc_h.items())) +
          f" → tok/fwd = 1 + mean accepted = **{1 + sum(k*v for k,v in acc_h.items())/tot_s:.3f}**\n")
    elif mean_rounds > 0:
        P(f"\nmean rounds/block **{mean_rounds:.2f}** → forwards/block = {mean_rounds:.2f} + 1 commit = {mean_rounds+1:.2f}; "
          f"tok/fwd = {CLj}/{mean_rounds+1:.2f} = **{CLj/(mean_rounds+1):.3f}**. "
          f"Fusing the commit into the next denoise (Fix C) would give {CLj}/{mean_rounds:.2f} = **{CLj/mean_rounds:.3f}** (+{100*(1/mean_rounds*(mean_rounds+1)-1):.0f}%).\n")
    T["rounds_hist"] = dict(rounds); T["mean_rounds"] = mean_rounds

    # commit types and calibration: for every (block, round, position) record, was the prediction == final committed token?
    types = collections.Counter(); bins = collections.defaultdict(lambda: [0, 0])  # conf bin -> [n, correct] for REJECTED preds
    pos_first = collections.defaultdict(lambda: [0, 0])  # position -> [n_blocks, committed in round 1]
    margin_stats = []  # (committed?, correct?, conf, margin)
    knob = collections.defaultdict(lambda: [0, 0, 0])  # knob -> [extra_commits, extra_correct, total_rejected]
    for r in recs:
        for b in r["blocks"]:
            final = b["ids"]
            for ri, rd in enumerate(b["r"]):
                for pos_s, e in rd.items():
                    pos = int(pos_s); ok = (e["p"] == final[pos]); types[e["t"]] += 1
                    top = e["top"]; margin = math.exp(top[0][1]) - (math.exp(top[1][1]) if len(top) > 1 else 0)
                    if ri == 0:
                        pos_first[pos][0] += 1; pos_first[pos][1] += int(e["t"] in ("committed", "forced"))
                    if e["t"] == "rejected":
                        bi = min(int(e["c"] * 20) / 20, 0.85); bins[bi][0] += 1; bins[bi][1] += int(ok)
                        # knob simulations: would this rejected prediction have been committed, and was it right?
                        for nm, cond in (("thr0.85", e["c"] > 0.85), ("thr0.80", e["c"] > 0.80), ("thr0.70", e["c"] > 0.70),
                                         ("margin>0.5", margin > 0.5), ("margin>0.3", margin > 0.3),
                                         ("top1==final (oracle upper bound)", ok)):
                            knob[nm][2] += 1
                            if cond:
                                knob[nm][0] += 1; knob[nm][1] += int(ok)
    P("**Commit decision types over all (round, masked position) records:** " + ", ".join(f"{k} {v}" for k, v in types.most_common()) + "\n")
    P("**Calibration of REJECTED predictions** (conf < 0.9): fraction whose argmax equalled the token finally committed at that position:\n")
    P("| conf bin | rejected preds | later committed same token | note |"); P("|---:|---:|---:|---|")
    for bi in sorted(bins):
        n, c = bins[bi]
        P(f"| {bi:.2f}–{bi+0.05:.2f} | {n} | {100*c/max(n,1):.1f}% | {'gate too strict here' if c/max(n,1) > 0.9 and n >= 20 else ''} |")
    P("\n**First-round acceptance by canvas position** (share of blocks where the position committed in round 1):\n")
    P("| position | blocks | committed round 1 |"); P("|---:|---:|---:|")
    for pos in sorted(pos_first):
        n, c = pos_first[pos]; P(f"| {pos} | {n} | {100*c/max(n,1):.1f}% |")
    P("\n**Knob simulation (first-order, offline):** of the rejected predictions, how many would each relaxed gate have committed, "
      "and how many of those matched the final token. Extra commits reduce rounds; wrong extra commits change the output.\n")
    P("| knob | extra commits | of which correct | precision | share of rejected recovered |"); P("|---|---:|---:|---:|---:|")
    for nm, (x, c, t) in knob.items():
        P(f"| {nm} | {x} | {c} | {100*c/max(x,1):.1f}% | {100*x/max(t,1):.1f}% |")
    T["types"] = dict(types); T["calibration"] = {f"{k:.2f}": v for k, v in bins.items()}; T["knobs"] = dict(knob)
    T["pos_first"] = {str(k): v for k, v in pos_first.items()}; T["trace_n_req"] = len(recs); T["trace_n_blocks"] = tot

    # per-step host time from the trace (no syncs)
    dec = [s[1] for r in recs for s in r["steps"] if s[0] == "decode"]
    if dec:
        P(f"\n**Per-step host wall time (trace server, decode steps, n={len(dec)}):** median {st.median(dec):.2f} ms, "
          f"p90 {sorted(dec)[int(0.9*len(dec))]:.2f} ms, mean {st.mean(dec):.2f} ms.\n")
        T["step_ms_median"] = st.median(dec); T["step_ms_p90"] = sorted(dec)[int(0.9*len(dec))]; T["step_ms_mean"] = st.mean(dec)
    # seed oracle: next_seed of block k must equal ids[0] of block k+1
    mism = tot_b = 0
    for r in recs:
        for k in range(len(r["blocks"]) - 1):
            tot_b += 1; mism += int(r["blocks"][k]["next_seed"] != r["blocks"][k + 1]["ids"][0])
    P(f"**Seed chain consistency:** {mism} / {tot_b} block boundaries where the committed next seed differs from the next block's slot 0 (expect 0).\n")
    T["seed_mismatch"] = [mism, tot_b]

    out["traces"][_job] = T
    if _ti == 0:
        out.update({k: v for k, v in T.items() if k not in ("job", "cl")})
# ---------------------------------------------------------------- timers
tl = []
for f in glob.glob(os.path.join(RUN, "*timers*/serve_rep*.log")):
    for ln in open(f, errors="replace"):
        m = re.search(r"\[TIME\] snap_ms=([\d.]+) fwd_ms=([\d.]+) samp_ms=([\d.]+) step_ms=([\d.]+) gdn_ms=([\d.]+) gdn_n=(\d+)", ln)
        if m:
            tl.append(tuple(float(x) for x in m.groups()))
if tl:
    P(f"## 5. Per-step phase cost (timers server, {len(tl)} decode steps, CUDA-synced)\n")
    P("| phase | median ms | mean ms | share |"); P("|---|---:|---:|---:|")
    med = [st.median(c) for c in zip(*tl)]
    for nm, i in (("snapshot/restore", 0), ("forward (model)", 1), ("sampler/gate", 2), ("TOTAL step", 3), ("  of which GDN override (24 layers)", 4)):
        P(f"| {nm} | {med[i]:.2f} | {st.mean([t[i] for t in tl]):.2f} | {100*med[i]/med[3]:.0f}% |")
    P(f"\nAR step ≈ 4.8 ms → break-even tok/fwd = {med[3]/4.8:.2f}.\n")
    out["phase_ms_median"] = {"snap": med[0], "fwd": med[1], "samp": med[2], "step": med[3], "gdn": med[4]}; out["timer_steps"] = len(tl)

# ---------------------------------------------------------------- sweep
# extra sweeps in variant subdirs (<RUN>/<variant>/sweep/bench_*): stored as out["sweep_<variant>"]
for _sd in sorted(glob.glob(os.path.join(RUN, "*/sweep/bench_*_c*.json"))):
    _var = os.path.basename(os.path.dirname(os.path.dirname(_sd))); m = re.search(r"bench_(.+)_c(\d+)\.json", os.path.basename(_sd)); dd = load(_sd)
    out.setdefault("sweep_" + _var, {}).setdefault(m.group(1), {})[int(m.group(2))] = {"tok_per_s": dd["tok_per_s"], "n_ok": dd["n_ok"], "n_err": dd["n_err"]}
sw = sorted(glob.glob(os.path.join(RUN, "sweep/bench_*_c*.json")))
if sw:
    P("## 6. Serving throughput vs concurrency (fixed 512-token outputs, 64 prompts)\n")
    tab = collections.defaultdict(dict)
    for f in sw:
        m = re.search(r"bench_(.+)_c(\d+)\.json", os.path.basename(f)); d = load(f)
        tab[m.group(1)][int(m.group(2))] = d
    cs = sorted({c for v in tab.values() for c in v})
    P("| engine | " + " | ".join(f"C={c} tok/s" for c in cs) + " | " + " | ".join(f"C={c} p50 s" for c in cs) + " |")
    P("|---|" + "---:|" * (2 * len(cs)))
    for n, v in tab.items():
        P(f"| {n} | " + " | ".join(f"{v[c]['tok_per_s']:.0f}" if c in v and v[c]['n_ok'] else "err" for c in cs) + " | "
          + " | ".join(f"{v[c]['p50_latency_s']:.1f}" if c in v and v[c]['n_ok'] else "—" for c in cs) + " |")
    out["sweep"] = {n: {c: {"tok_per_s": v[c]["tok_per_s"], "n_ok": v[c]["n_ok"], "n_err": v[c]["n_err"]} for c in v} for n, v in tab.items()}
    P("")

open(os.path.join(RUN, "REPORT.md"), "w").write("\n".join(lines) + "\n")
json.dump(out, open(os.path.join(RUN, "summary.json"), "w"), indent=1, default=str)
print("\n".join(lines))
