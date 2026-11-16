"""Score an OCR model on docr-ko-eval with the OmniDocBench metric policy.

Metrics come from OmniDocBench's own code (github.com/opendatalab/OmniDocBench), imported from a local
checkout given by --omnidocbench (or $OMNIDOCBENCH_ROOT):

  text     Edit_dist   OmniDocBench `normalized_text` on both sides (inline LaTeX -> unicode, then only
                       letters/digits/CJK/Hangul kept: spaces and punctuation dropped), Levenshtein / max len.
                       Reported like OmniDocBench: edit_sample_avg (mean of per-crop ratios) and edit_whole
                       (sum of edits / sum of lengths).
           BLEU        corpus BLEU-4 (sacreBLEU, 13a tokenization, no smoothing) on lightly normalized text
                       (`textblock2unicode`, whitespace collapsed). OmniDocBench's own call_BLEU feeds it the
                       space-free normalized strings, which makes word-level BLEU degenerate, so we don't.
  table    TEDS        OmniDocBench TEDS (+ TEDS_structure_only) on `normalize_table_match_content` of both
                       sides (th->td, thead/tbody dropped, cell clean-up). Markdown pipe tables are converted
                       to HTML first. Edit_dist on the same normalized HTML is also reported.
  formula  CDM         OmniDocBench CDM: both LaTeX strings are rendered with pdflatex and the rendered
                       characters are matched (F1). Wrappers ($$, \\[ \\], \\( \\)) and \\tag are stripped first.
           Edit_dist   on OmniDocBench `normalized_formula`.
  Overall  = mean of (1 - text Edit_dist[sample avg]), table TEDS, formula CDM   (OmniDocBench v1.5 style), x100

CDM needs pdflatex (+ CJK for Hangul \\text{}), Ghostscript and ImageMagick's `magick` (IM6's `convert`
works through a one-line `magick` shim on PATH), and node.

Usage
  python score.py --data data --endpoint http://localhost:30500/v1 --model gravityocr --out runs/spec
  python score.py --data data --preds preds.jsonl --out runs/x        # {"id", "pred"} per line
  python score.py --data data --self_check                            # targets vs themselves
Writes preds.jsonl (endpoint mode), scores.jsonl (per crop; `score` = the task's headline metric, used by the
viewer), summary.json and summary.md.
"""
import argparse
import base64
import json
import os
import re
import sys
import tempfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

MAX_TOKENS = {"text": 512, "table": 4096, "formula": 4096}   # = infer_omnidocbench.MAX_NEW_TOKENS
ODB = None   # OmniDocBench root, set in main()


def _odb():
    """Import OmniDocBench metric code lazily (also inside worker processes)."""
    import warnings
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    if ODB and ODB not in sys.path:
        sys.path.insert(0, ODB)
    from src.core.preprocess.data_preprocess import normalized_text, normalized_formula, textblock2unicode
    from src.core.matching.match import normalize_table_match_content
    from src.metrics.cal_metric import _strip_cdm_math_wrappers
    from src.metrics.table_metric import TEDS
    from src.metrics.cdm_metric import CDM
    return dict(normalized_text=normalized_text, normalized_formula=normalized_formula,
                textblock2unicode=textblock2unicode, norm_table=normalize_table_match_content,
                strip_cdm=_strip_cdm_math_wrappers, TEDS=TEDS, CDM=CDM)


def _edit(a, b):
    import Levenshtein
    n = max(len(a), len(b))
    return (Levenshtein.distance(a, b), n)


def md_table_to_html(s):
    rows = [r for r in s.strip().splitlines() if r.strip().startswith("|")]
    rows = [r for r in rows if not re.fullmatch(r"\s*\|?[\s:\-|]+\|?\s*", r)]
    if not rows:
        return s
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    return "<table>" + "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in cells) + "</table>"


# ------------------------------------------------------------------ per-crop scoring (worker)
def score_one(args):
    rec, pred, cdm_root = args
    m = _odb()
    t = rec["task_type"]
    pred = pred or ""
    out = {}
    if t == "text":
        g, p = m["normalized_text"](rec["target"]), m["normalized_text"](pred)
        d, n = _edit(p, g)
        out.update(edit=d / n if n else 0.0, edit_num=d, upper_len=n, score=1 - (d / n if n else 0.0))
    elif t == "table":
        p_html = pred if "<table" in pred.lower() else md_table_to_html(pred)
        g, p = m["norm_table"](rec["target"]), m["norm_table"](p_html)
        try:
            teds = m["TEDS"](structure_only=False).evaluate(p, g)
            teds_s = m["TEDS"](structure_only=True).evaluate(p, g)
        except Exception as e:          # OmniDocBench sets failures to 0
            teds, teds_s, out["error"] = 0.0, 0.0, f"{type(e).__name__}: {e}"
        d, n = _edit(p, g)
        out.update(teds=teds, teds_struct=teds_s, edit=d / n if n else 0.0, score=teds)
    else:
        gt_c = m["strip_cdm"](rec["target"])
        pr_c = m["strip_cdm"](pred.split("```latex")[-1].split("```")[0])
        cdm = m["CDM"](output_root=cdm_root).evaluate(gt_c, pr_c, rec["id"])
        g, p = m["normalized_formula"](rec["target"]), m["normalized_formula"](pred)
        d, n = _edit(p, g)
        out.update(cdm=cdm["F1_score"], cdm_recall=cdm["recall"], cdm_precision=cdm["precision"],
                   edit=d / n if n else 0.0, score=cdm["F1_score"])
        if cdm.get("cdm_eval_error"):
            out["error"] = cdm["cdm_eval_error"]
    return rec["id"], out


def corpus_bleu(recs, preds):
    import sacrebleu
    m = _odb()
    clean = lambda s: re.sub(r"\s+", " ", m["textblock2unicode"](s or "")).strip()
    hyp = [clean(preds.get(r["id"], "")) for r in recs]
    ref = [clean(r["target"]) for r in recs]
    return sacrebleu.corpus_bleu(hyp, [ref], tokenize="13a", smooth_method="none").score / 100


# ------------------------------------------------------------------ inference
def predict(rec, data_dir, endpoint, model, timeout):
    import urllib.request
    b64 = base64.b64encode((Path(data_dir) / "images" / f"{rec['id']}.png").read_bytes()).decode()
    body = {"model": model, "temperature": 0.0, "max_tokens": MAX_TOKENS[rec["task_type"]],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                {"type": "text", "text": rec["prompt"]}]}]}
    req = urllib.request.Request(f"{endpoint.rstrip('/')}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.load(r)
    return out["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------------ summary
def summarize(rows, recs, preds):
    by = defaultdict(list)
    for r in rows:
        by[r["task_type"]].append(r)
    s = {"protocol": "OmniDocBench metric code (Edit_dist / BLEU / TEDS / CDM)", "by_task": {}}
    if by["text"]:
        tx = by["text"]
        s["by_task"]["text"] = {
            "n": len(tx),
            "Edit_dist": round(sum(r["edit"] for r in tx) / len(tx), 4),                      # edit_sample_avg
            "Edit_dist_whole": round(sum(r["edit_num"] for r in tx) / max(1, sum(r["upper_len"] for r in tx)), 4),
            "BLEU": round(corpus_bleu([x for x in recs if x["task_type"] == "text"], preds), 4)}
    if by["table"]:
        tb = by["table"]
        s["by_task"]["table"] = {"n": len(tb), "TEDS": round(sum(r["teds"] for r in tb) / len(tb), 4),
                                 "TEDS_structure_only": round(sum(r["teds_struct"] for r in tb) / len(tb), 4),
                                 "Edit_dist": round(sum(r["edit"] for r in tb) / len(tb), 4)}
    if by["formula"]:
        fm = by["formula"]
        s["by_task"]["formula"] = {"n": len(fm), "CDM": round(sum(r["cdm"] for r in fm) / len(fm), 4),
                                   "Edit_dist": round(sum(r["edit"] for r in fm) / len(fm), 4),
                                   "cdm_errors": sum(1 for r in fm if r.get("error"))}
    bt = s["by_task"]
    if all(k in bt for k in ("text", "table", "formula")):
        s["Overall"] = round(((1 - bt["text"]["Edit_dist"]) + bt["table"]["TEDS"] + bt["formula"]["CDM"]) / 3 * 100, 2)
    for key in ("kind", "source", "degradation", "lang"):
        d = defaultdict(list)
        for r in rows:
            d[(r["task_type"], r.get(key))].append(r["score"])
        s[f"by_{key}"] = {f"{t}/{k}": {"n": len(v), "mean": round(sum(v) / len(v), 4)} for (t, k), v in sorted(d.items(), key=str)}
    return s


def to_md(s):
    head = {"text": "1-Edit_dist", "table": "TEDS", "formula": "CDM"}
    lines = [f"Protocol: {s['protocol']}", "", "| task | n | metrics |", "|---|---|---|"]
    for t, e in s["by_task"].items():
        lines.append(f"| {t} | {e['n']} | " + ", ".join(f"{k}={v}" for k, v in e.items() if k != "n") + " |")
    if "Overall" in s:
        lines.append(f"\n**Overall** = mean(1 − text Edit_dist, table TEDS, formula CDM) × 100 = **{s['Overall']}**\n")
    for key in ("source", "degradation", "kind", "lang"):
        lines += [f"\n### by {key} (headline metric: " + ", ".join(f"{t}={h}" for t, h in head.items()) + ")",
                  "| group | n | mean |", "|---|---|---|"]
        lines += [f"| {g} | {v['n']} | {v['mean']} |" for g, v in s[f"by_{key}"].items()]
    return "\n".join(lines)


def main():
    global ODB
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset dir (data.jsonl + images/)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--endpoint")
    src.add_argument("--preds")
    src.add_argument("--self_check", action="store_true")
    ap.add_argument("--omnidocbench", default=os.environ.get("OMNIDOCBENCH_ROOT"), help="OmniDocBench checkout")
    ap.add_argument("--model", default="gravityocr")
    ap.add_argument("--workers", type=int, default=1, help="inference concurrency (1 = single-stream)")
    ap.add_argument("--score_workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--timeout", type=int, default=600)
    ap.add_argument("--out", default="runs/latest")
    ap.add_argument("--split", choices=["all", "dev", "test"], default="all",
                    help="score only the dev or test half (ids from <data>/splits.json)")
    a = ap.parse_args()
    if not a.omnidocbench or not (Path(a.omnidocbench) / "src" / "metrics").is_dir():
        sys.exit("--omnidocbench must point to an OmniDocBench checkout (git clone https://github.com/opendatalab/OmniDocBench)")
    ODB = os.path.abspath(a.omnidocbench)
    os.environ["OMNIDOCBENCH_ROOT"] = ODB
    os.environ.setdefault("CDM_SAVE_VIS", "0")

    recs = [json.loads(l) for l in open(Path(a.data) / "data.jsonl", encoding="utf-8")]
    if a.split != "all":
        split_of = json.load(open(Path(a.data) / "splits.json", encoding="utf-8"))
        recs = [r for r in recs if split_of.get(r["id"]) == a.split]
        print(f"split {a.split}: {len(recs)} crops")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.self_check:
        preds = {r["id"]: r["target"] for r in recs}
    elif a.preds:
        preds = {d["id"]: d["pred"] for d in map(json.loads, open(a.preds, encoding="utf-8"))}
    else:
        with ThreadPoolExecutor(a.workers) as ex:
            res = list(ex.map(lambda r: predict(r, a.data, a.endpoint, a.model, a.timeout), recs))
        preds = {r["id"]: p for r, p in zip(recs, res)}
        with open(out / "preds.jsonl", "w", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps({"id": r["id"], "pred": preds[r["id"]]}, ensure_ascii=False) + "\n")

    cdm_root = tempfile.mkdtemp(prefix="cdm_", dir=out)
    todo = [(r, preds.get(r["id"], ""), cdm_root) for r in recs if r["id"] in preds]
    with ProcessPoolExecutor(a.score_workers, initializer=_init_worker, initargs=(ODB,)) as ex:
        scored = dict(ex.map(score_one, todo, chunksize=4))
    rows = []
    for r in recs:
        if r["id"] in scored:
            rows.append({**{k: r.get(k) for k in ("id", "task_type", "kind", "degradation", "post", "lang")},
                         "source": r.get("post") if r.get("post") in ("screenshot", "authored", "synthetic") else "blog",
                         **scored[r["id"]]})
    with open(out / "scores.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    summ = summarize(rows, recs, preds)
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=2))
    (out / "summary.md").write_text(to_md(summ))
    print(json.dumps({"by_task": summ["by_task"], "Overall": summ.get("Overall")}, ensure_ascii=False, indent=2))


def _init_worker(root):
    global ODB
    ODB = root


if __name__ == "__main__":
    main()
