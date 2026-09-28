#!/usr/bin/env python3
"""Summarize a fulleval run dir: GSM8K (2-node halves merged), FunctionChat eval_score.json, Ko-AgentBench summary.csv per mode."""
import csv, glob, json, os, sys
RUN = sys.argv[1]; MODES = ["causal", "selfspec-n4", "selfspec-n8", "selfspec-n32"]
out = {"run": RUN, "gsm8k": {}, "fc": {}, "koab": {}}
for m in MODES:
    c = t = tok = 0; wall = 0.0
    for f in glob.glob(f"{RUN}/gsm8k-{m}-n*/gsm8k_summary_*.json"):
        s = json.load(open(f)); c += s["correct"]; t += s["total"]; tok += s.get("total_tok", 0); wall = max(wall, s.get("wall_seconds", 0))
    if t: out["gsm8k"][m] = {"correct": c, "total": t, "acc": c / t, "tokens": tok, "wall_s": wall, "tok_s_16gpu": tok / wall if wall else None}
    fs = glob.glob(f"{RUN}/fc-{m}/fc_output/*eval_score.json")
    if fs:
        s = json.load(open(fs[0])); out["fc"][m] = {k: s.get(k) for k in ("singlecall_score", "dialog_score", "calldecision_score")}
    ks = glob.glob(f"{RUN}/koab-{m}/koab_report/evaluation_summary.csv")
    if ks:
        rows = list(csv.DictReader(open(ks[0])))
        lv = {r["Level"]: {k: (float(r[k]) if k != "Level" else r[k]) for k in r if k in ("Total_Tasks", "Evaluated_Tasks", "SR", "pass@k", "RRR", "EPR_CVR", "Avg_Tokens", "Avg_TPS", "Avg_Exec_Time")} for r in rows}
        tot = sum(v["Total_Tasks"] for v in lv.values())
        out["koab"][m] = {"levels": lv, "sr_weighted": sum(v["SR"] * v["Total_Tasks"] for v in lv.values()) / tot, "sr_mean": sum(v["SR"] for v in lv.values()) / len(lv), "tasks": tot,
                          "tps_mean": sum(v["Avg_TPS"] for v in lv.values()) / len(lv)}
    koab_log = f"{RUN}/koab-{m}/koab.log"
    if os.path.exists(koab_log):
        txt = open(koab_log).read(); out["koab"].setdefault(m, {})["api_timeouts"] = txt.count("APITimeoutError"); out["koab"][m]["context_exceeded"] = txt.count("ContextWindowExceeded")
json.dump(out, open(f"{RUN}/summary_fulleval.json", "w"), indent=1)
print("| mode | GSM8K acc | tok/s (16 GPU) | FC singlecall | FC dialog avg | FC calldecision | KoAB SR task-weighted (L1..L7) |"); print("|---|---:|---:|---:|---:|---:|---|")
for m in MODES:
    g = out["gsm8k"].get(m); f = out["fc"].get(m); k = out["koab"].get(m)
    def sc(d, key):
        return f"{d[key]:.3f}" if d and isinstance(d.get(key), (int, float)) else "—"
    fcs = sc(f['singlecall_score'], 'total_pass_rate') if f and f.get('singlecall_score') else "—"
    fcd = f"{f['dialog_score']['avg(micro)']:.3f}" if f and f.get("dialog_score") and 'avg(micro)' in f['dialog_score'] else "—"
    fcc = sc(f['calldecision_score'], 'total_pass_rate') if f and f.get('calldecision_score') else "—"
    ko = f"{k['sr_weighted']:.3f} (" + " ".join(f"{k['levels'][l]['SR']:.2f}" for l in sorted(k['levels'])) + ")" if k and "levels" in k else "—"
    print(f"| {m} | {g['acc']*100:.1f}% ({g['correct']}/{g['total']}) | {g['tok_s_16gpu']:.0f} | {fcs} | {fcd} | {fcc} | {ko} |" if g else f"| {m} | — | — | {fcs} | {fcd} | {fcc} | {ko} |")
