#!/usr/bin/env python3
"""Report one checkpoint's eval: tok/fwd + accept histogram (N=4, N=8) and AR identity/accuracy vs the step_18000 references.
usage: eval_ckpt_report.py <tag>"""
import collections, json, sys, glob
TAG = sys.argv[1]; R = "$SCRATCH/runs/draftalign_eval"
REF_AR = "$SCRATCH/runs/vspec_20260909_172532/s3-n4-30"          # step_18000 self-spec == AR greedy path
ref = json.load(open(glob.glob(f"{REF_AR}/gsm8k_details_*.json")[0]))
for job, label in ((f"{TAG}-ss4", "self-spec N=4"), (f"{TAG}-ss8", "self-spec N=8"), (f"{TAG}-ar", "AR greedy")):
    d = f"{R}/{job}"
    try:
        det = json.load(open(glob.glob(f"{d}/gsm8k_details_*.json")[0])); summ = json.load(open(glob.glob(f"{d}/gsm8k_summary_*.json")[0]))
    except Exception as e:
        print(f"{label}: missing ({e})"); continue
    same = sum(a["generation"] == b["generation"] for a, b in zip(det, ref))
    line = f"{label}: acc {summ['correct']}/{summ['total']}  identical-to-step18000 {same}/{len(det)}"
    tr = glob.glob(f"{d}/trace_rep*.jsonl")
    if tr:
        T = [json.loads(l) for l in open(tr[0])]; st = [s for t in T for s in t.get("spec_steps", [])]
        if st:
            h = collections.Counter(a for _, a in st); n = len(st)
            line += f"  tok/fwd {sum(1+a for _,a in st)/n:.3f}  accept-hist " + " ".join(f"{k}:{100*v/n:.0f}%" for k, v in sorted(h.items()))
    print(line)
