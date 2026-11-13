#!/usr/bin/env bash
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# Evaluate a GravityOCR checkpoint on a docr-ko-eval dataset (region crops) on one NVIDIA GPU, in one command:
#
#   export the dataset -> serve the checkpoint (inference/gravityocr/serve.sh: SGLang + the GravityOCR patch)
#   -> warm-up -> one measured pass per concurrency level -> stop the server
#   -> CDM self-check -> score each split (OmniDocBench Edit_dist / BLEU / TEDS / CDM) -> report.md, report.json
#
# Set up once with setup.sh (it writes $EVAL_HOME/env.sh, which run.sh reads), then:
#
#     MODEL=trillionlabs/GravityOCR-KO bash run.sh                   # DATASET defaults to trillionlabs/gravityocr-dev
#     MODEL=/ckpt/my_checkpoint MODE=causal CONCURRENCY="1 8 64" bash run.sh
#     MODEL=... DATASET=/data/my-eval bash run.sh                     # a local datasets.save_to_disk directory
#     ENDPOINT=http://127.0.0.1:30000/v1 bash run.sh                  # use a running server
#     bash run.sh MODEL=...                                           # KEY=VALUE arguments work too
#
# Settings (environment or KEY=VALUE arguments):
#   MODEL          checkpoint dir or Hub id (required unless ENDPOINT is set)
#   DATASET        Hub dataset id or datasets.save_to_disk directory (default trillionlabs/gravityocr-dev, the
#                  dev split of docr-ko-eval-v2; it is private: log in with `hf auth login` or set HF_TOKEN)
#   OUT            output directory (default runs/<model>-<mode>-<YYYYmmdd-HHMMSS>)
#   MODE           self-spec (default) | causal
#   CONCURRENCY    concurrency levels, space-separated (default 128)
#   WARMUP         1 (default): one crop per task, then one full pass at the first level, both discarded.
#                  The first high-concurrency pass on a fresh server is ~2x slower; 0 skips the warm-up
#   SPLITS         splits to score (default: every split in DATASET)
#   SCORE_LEVELS   levels whose predictions are scored (default: all of CONCURRENCY; "none" skips scoring)
#   ENDPOINT       OpenAI base URL of a running server; run.sh then starts and stops nothing
#   PORT           server port (default 30500)
#   EVAL_HOME      setup.sh's install directory (default $HOME/.gravityocr-eval); its env.sh sets the defaults
#                  of VBIN, PYTHON and OMNIDOCBENCH and puts its TeX Live on PATH
#   VBIN           bin/ of the patched SGLang venv, for serve.sh (see inference/gravityocr/sglang/setup.sh)
#   PYTHON         interpreter for export, client, scoring and report (default python3); needs datasets,
#                  Pillow, pymupdf and OmniDocBench's requirements
#   OMNIDOCBENCH   OmniDocBench checkout (default $OMNIDOCBENCH_ROOT)
#   SCORE_WORKERS  scoring processes (default: number of CPUs)
#   SERVE_TIMEOUT  seconds to wait for the server to come up (default 900)
#   Other serve.sh settings (GPU, MEMFRAC, CUDA_HOME, SKIP_PREFLIGHT, HF_HOME, ...) pass through.
#
# CDM renders formulas with pdflatex: setup.sh installs TeX Live 2022.06 or newer (xcolor 2.14, for \mathcolor;
# with Ubuntu 22.04's apt TeX Live every formula scores 0). `magick` is used if it is on PATH, otherwise run.sh
# puts a PyMuPDF stand-in in $OUT/.bin. A CDM self-check runs before scoring and stops the run if CDM does not
# score a formula against itself as 1.0.
#
# Output in $OUT: data/ (exported crops), preds_c<N>.jsonl, sweep.json, serve.log, c<N>/<split>/summary.{json,md}
# and scores.jsonl, timing.json, report.md, report.json.
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    -h|--help) sed -n '/^# Evaluate /,/^set -euo pipefail$/{/^set /d;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
    [A-Z_]*=*) export "${kv?}" ;;
    *) echo "[run] error: unknown argument: $kv (use KEY=VALUE)" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
die() { echo "[run] error: $*" >&2; exit 1; }
log() { echo "[run $(date +%H:%M:%S)] $*" >&2; }
now() { date +%s.%N; }
EVAL_HOME=${EVAL_HOME:-$HOME/.gravityocr-eval}
# shellcheck source=/dev/null
[ ! -f "$EVAL_HOME/env.sh" ] || . "$EVAL_HOME/env.sh"

DATASET=${DATASET:-trillionlabs/gravityocr-dev}
[ -n "${ENDPOINT:-}" ] || : "${MODEL:?set MODEL to a checkpoint dir or Hub id (or ENDPOINT to a running server)}"
MODE=${MODE:-self-spec}
CONCURRENCY=${CONCURRENCY:-128}
WARMUP=${WARMUP:-1}
SCORE_LEVELS=${SCORE_LEVELS:-$CONCURRENCY}
PORT=${PORT:-30500}
PYTHON=${PYTHON:-python3}
OMNIDOCBENCH=${OMNIDOCBENCH:-${OMNIDOCBENCH_ROOT:-}}
SCORE_WORKERS=${SCORE_WORKERS:-$(nproc 2>/dev/null || sysctl -n hw.ncpu)}
SERVE_TIMEOUT=${SERVE_TIMEOUT:-900}
SCORE_PY=${SCORE_PY:-$HERE/score.py}
SERVE_SH=${SERVE_SH:-$REPO/inference/gravityocr/serve.sh}
case "$MODE" in self-spec|causal) ;; *) die "MODE must be self-spec or causal" ;; esac
NAME=$(basename "${MODEL:-endpoint}")
OUT=${OUT:-runs/$NAME-$MODE-$(date +%Y%m%d-%H%M%S)}

[ -f "$SCORE_PY" ] || die "score.py not found at $SCORE_PY"
if [ "$SCORE_LEVELS" != none ]; then
  [ -n "$OMNIDOCBENCH" ] && [ -d "$OMNIDOCBENCH/src/metrics" ] || die "set OMNIDOCBENCH to an OmniDocBench checkout (or run setup.sh)"
  command -v pdflatex >/dev/null || die "pdflatex not on PATH (run setup.sh: CDM needs TeX Live 2022.06 or newer)"
fi
mkdir -p "$OUT"
OUT=$(cd "$OUT" && pwd)
log "out=$OUT model=${MODEL:-$ENDPOINT} mode=$MODE concurrency=[$CONCURRENCY] dataset=$DATASET"

# ---- 1. export: Hub or save_to_disk -> data.jsonl + images/<id>.png + splits.json (the layout score.py reads) -------
t0=$(now)
"$PYTHON" - "$DATASET" "$OUT/data" <<'PY'
import io, json, sys
from pathlib import Path
import datasets
from PIL import Image
src, out = sys.argv[1], Path(sys.argv[2])
(out / "images").mkdir(parents=True, exist_ok=True)
# A local save_to_disk directory, else a Hub dataset id (a private one needs `hf auth login` or HF_TOKEN)
dd = datasets.load_from_disk(src) if Path(src).is_dir() else datasets.load_dataset(src)
if isinstance(dd, datasets.Dataset):  # one unsplit Dataset: score it whole
    dd = {"all": dd}
rows, split_of = [], {}
keep = ("id", "prompt", "target", "task_type", "kind", "lang", "degradation", "width", "height", "gt_status",
        "domain", "difficulty", "source")
for split, ds in dd.items():
    for r in ds.cast_column("image", datasets.Image(decode=False)):
        img = r["image"]["bytes"]
        if not img.startswith(b"\x89PNG"):  # the client sends every crop as image/png
            buf = io.BytesIO(); Image.open(io.BytesIO(img)).save(buf, "PNG"); img = buf.getvalue()
        (out / "images" / f"{r['id']}.png").write_bytes(img)
        # score.py's `post`: the post name for blog crops, the source otherwise
        post = r.get("source_detail") if r.get("source") == "blog" else r.get("source")
        rows.append({k: r.get(k) for k in keep} | {"post": post})
        split_of[r["id"]] = split
rows.sort(key=lambda r: r["id"])
with open(out / "data.jsonl", "w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
json.dump(split_of, open(out / "splits.json", "w"), indent=0)
by = {}
for r in rows:
    by[r["task_type"]] = by.get(r["task_type"], 0) + 1
print(f"[run] exported {len(rows)} crops {by}, splits {sorted(set(split_of.values()))}", file=sys.stderr)
PY
t_export=$(now)
SPLITS=${SPLITS:-$("$PYTHON" -c 'import json, sys; print(" ".join(sorted(set(json.load(open(sys.argv[1])).values()))))' "$OUT/data/splits.json")}

# ---- 2. serve ----------------------------------------------------------------------------------------------------
SRV=
stop_server() {
  [ -n "$SRV" ] || return 0
  log "stopping the server"
  kill -TERM -- "-$SRV" 2>/dev/null || true
  for _ in $(seq 30); do kill -0 -- "-$SRV" 2>/dev/null || break; sleep 1; done
  kill -KILL -- "-$SRV" 2>/dev/null || true
  SRV=
}
trap stop_server EXIT
trap 'exit 130' INT TERM

t_up=$t_export
if [ -z "${ENDPOINT:-}" ]; then
  [ -f "$SERVE_SH" ] || die "serve.sh not found at $SERVE_SH"
  ENDPOINT=http://127.0.0.1:$PORT/v1
  log "starting serve.sh (log: $OUT/serve.log)"
  # setsid: the server gets its own process group, so stop_server can take down SGLang's children too
  CKPT=$MODEL MODE=$MODE PORT=$PORT SERVE_HOST=127.0.0.1 SERVED_NAME=glm-ocr \
    setsid bash "$SERVE_SH" > "$OUT/serve.log" 2>&1 < /dev/null &
  SRV=$!
  deadline=$(( $(date +%s) + SERVE_TIMEOUT ))
  until curl -sf "$ENDPOINT/models" > /dev/null; do
    kill -0 "$SRV" 2>/dev/null || { tail -30 "$OUT/serve.log" >&2; SRV=; die "the server exited; see $OUT/serve.log"; }
    [ "$(date +%s)" -lt "$deadline" ] || { tail -30 "$OUT/serve.log" >&2; die "the server did not come up in ${SERVE_TIMEOUT}s"; }
    sleep 2
  done
  t_up=$(now)
  log "server up in $(awk "BEGIN{printf \"%.1f\", $t_up-$t_export}") s"
else
  curl -sf "$ENDPOINT/models" > /dev/null || die "no server answers at $ENDPOINT/models"
fi

# ---- 3. predict: warm-up, then one measured pass per concurrency level -------------------------------------------
"$PYTHON" - "$OUT" "$ENDPOINT" "$SCORE_PY" "$WARMUP" $CONCURRENCY <<'PY'
import base64, importlib.util, json, statistics as st, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
out, endpoint, score_py, warmup = sys.argv[1], sys.argv[2].rstrip("/"), sys.argv[3], sys.argv[4] == "1"
levels = [int(x) for x in sys.argv[5:]]
spec = importlib.util.spec_from_file_location("score", score_py)
score = importlib.util.module_from_spec(spec); spec.loader.exec_module(score)
recs = [json.loads(l) for l in open(f"{out}/data/data.jsonl", encoding="utf-8")]
imgs = {r["id"]: base64.b64encode(open(f"{out}/data/images/{r['id']}.png", "rb").read()).decode() for r in recs}

def one(r):
    body = {"model": "glm-ocr", "temperature": 0.0, "max_tokens": score.MAX_TOKENS[r["task_type"]],
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{imgs[r['id']]}"}},
                {"type": "text", "text": r["prompt"]}]}]}
    req = urllib.request.Request(f"{endpoint}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json", "Authorization": "Bearer x"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            o = json.load(resp)
        p, toks, err = o["choices"][0]["message"]["content"] or "", o["usage"]["completion_tokens"], None
    except Exception as e:
        p, toks, err = "", 0, f"{type(e).__name__}: {e}"
    return {"id": r["id"], "pred": p, "sec": round(time.time() - t, 4), "out_tokens": toks, "error": err}

def run(c):
    t0 = time.time()
    with ThreadPoolExecutor(c) as ex:
        res = list(ex.map(one, recs))
    wall = time.time() - t0
    s = sorted(x["sec"] for x in res)
    toks = sum(x["out_tokens"] for x in res)
    return res, {"concurrency": c, "crops": len(recs), "wall_s": round(wall, 2), "crops_per_s": round(len(recs) / wall, 2),
                 "lat_mean_s": round(st.mean(s), 3), "lat_p50_s": round(s[len(s) // 2], 3),
                 "lat_p90_s": round(s[int(0.9 * len(s))], 3), "lat_max_s": round(s[-1], 3),
                 "errors": sum(1 for x in res if x["error"]), "out_tokens": toks, "out_tok_per_s": round(toks / wall, 1)}

summary = {"warmup": None, "levels": []}
if warmup:
    for t in ("text", "table", "formula"):
        r = next((x for x in recs if x["task_type"] == t), None)
        if r:
            one(r)
    _, w = run(levels[0])
    summary["warmup"] = w
    print(f"[run] warm-up  c={w['concurrency']}: {w['wall_s']} s, {w['crops_per_s']} crops/s (discarded)", file=sys.stderr)
for c in levels:
    res, row = run(c)
    with open(f"{out}/preds_c{c}.jsonl", "w", encoding="utf-8") as f:
        for x in res:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    summary["levels"].append(row)
    print(f"[run] measured c={c}: {row['wall_s']} s, {row['crops_per_s']} crops/s, p50 {row['lat_p50_s']} s, "
          f"{row['out_tok_per_s']} tok/s, errors {row['errors']}", file=sys.stderr)
json.dump(summary, open(f"{out}/sweep.json", "w"), indent=1)
PY
t_pred=$(now)
stop_server

# ---- 4. score ----------------------------------------------------------------------------------------------------
t_score0=$(now)
if [ "$SCORE_LEVELS" != none ]; then
  export PATH CDM_CJK_FONT=${CDM_CJK_FONT:-mj} CDM_SAVE_VIS=0
  if ! command -v magick > /dev/null; then
    # PyMuPDF stand-in for the two ImageMagick calls CDM makes, rendered on opaque white:
    #   magick -density D -quality Q in.pdf[0] out.png   and   magick -density D -quality Q in.pdf -append out.png
    mkdir -p "$OUT/.bin"
    { echo "#!$(command -v "$PYTHON")"; cat <<'PY'; } > "$OUT/.bin/magick"
import re, sys
import pymupdf
from PIL import Image
a = sys.argv[1:]
dpi = int(a[a.index("-density") + 1]) if "-density" in a else 72
files = [x for i, x in enumerate(a) if not x.startswith("-") and (i == 0 or a[i - 1] not in ("-density", "-quality"))]
src, out = files[0], files[-1]
m = re.fullmatch(r"(.*)\[(\d+)\]", src)
doc = pymupdf.open(m.group(1) if m else src)
ims = []
for p in ([int(m.group(2))] if m else range(doc.page_count)):
    pix = doc[p].get_pixmap(dpi=dpi, alpha=False)
    ims.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
canvas = Image.new("RGB", (max(i.width for i in ims), sum(i.height for i in ims)), "white")
y = 0
for i in ims:
    canvas.paste(i, (0, y)); y += i.height
canvas.save(out)
PY
    chmod +x "$OUT/.bin/magick"
    PATH=$OUT/.bin:$PATH
  fi
  log "CDM self-check"
  (cd "$OUT" && "$PYTHON" - "$OMNIDOCBENCH" <<'PY') || die "CDM self-check failed: formulas would all score 0 (TeX Live too old, a missing package, or magick)"
import sys, tempfile, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, sys.argv[1])
from src.metrics.cdm_metric import CDM
root = tempfile.mkdtemp(prefix="cdm_selfcheck_", dir=".")
for f in (r"\frac{a}{b}+\sqrt{c}", r"x=\text{가나}+1"):
    f1 = CDM(output_root=root).evaluate(f, f, "selfcheck")["F1_score"]
    print(f"[run] CDM({f!r}, itself) = {f1}", file=sys.stderr)
    if f1 < 0.999:
        sys.exit(1)
PY
  for c in $SCORE_LEVELS; do
    [ -f "$OUT/preds_c$c.jsonl" ] || die "no predictions for SCORE_LEVELS level $c"
    mkdir -p "$OUT/c$c"
    for s in $SPLITS; do
      log "scoring c=$c $s"
      "$PYTHON" "$SCORE_PY" --data "$OUT/data" --omnidocbench "$OMNIDOCBENCH" --score_workers "$SCORE_WORKERS" \
        --preds "$OUT/preds_c$c.jsonl" --split "$s" --out "$OUT/c$c/$s" > "$OUT/c$c/$s.log" 2>&1 \
        || { tail -20 "$OUT/c$c/$s.log" >&2; die "scoring c=$c $s failed"; }
    done
  done
fi
t_end=$(now)

# ---- 5. report ---------------------------------------------------------------------------------------------------
"$PYTHON" - "$OUT" "${MODEL:-$ENDPOINT}" "$MODE" "$SCORE_LEVELS" "$SPLITS" \
  "$t0" "$t_export" "$t_up" "$t_pred" "$t_score0" "$t_end" <<'PY'
import json, os, re, sys
from collections import defaultdict
out, model, mode, levels, splits = sys.argv[1:6]
t0, t_export, t_up, t_pred, t_score0, t_end = map(float, sys.argv[6:12])
levels = [] if levels == "none" else [int(x) for x in levels.split()]
splits = splits.split()
recs = {r["id"]: r for r in map(json.loads, open(f"{out}/data/data.jsonl", encoding="utf-8"))}
sweep = json.load(open(f"{out}/sweep.json"))
timing = {"export_s": round(t_export - t0, 1), "server_startup_s": round(t_up - t_export, 1),
          "predict_s": round(t_pred - t_up, 1), "score_s": round(t_end - t_score0, 1), "total_s": round(t_end - t0, 1)}
json.dump(timing, open(f"{out}/timing.json", "w"), indent=1)
accept = None
if os.path.exists(f"{out}/serve.log"):
    acc = [float(x) for x in re.findall(r"accept len: ([0-9.]+)", open(f"{out}/serve.log", errors="ignore").read())]
    accept = round(sum(acc) / len(acc), 2) if acc else None
HEAD = {"text": "1-Edit_dist", "table": "TEDS", "formula": "CDM"}
acc_rows, src_rows = [], []
for c in levels:
    for s in splits:
        d = f"{out}/c{c}/{s}"
        if not os.path.exists(f"{d}/summary.json"):
            continue
        summ = json.load(open(f"{d}/summary.json"))
        bt = summ["by_task"]
        head = {t: (1 - bt[t]["Edit_dist"] if t == "text" else bt[t][HEAD[t]]) for t in HEAD if t in bt}
        overall = summ.get("Overall", round(sum(head.values()) / len(head) * 100, 2) if head else None)
        acc_rows.append({"concurrency": c, "split": s, "Overall": overall,
                         "overall_tasks": "text+table+formula" if "Overall" in summ else "+".join(head),
                         **{f"{t}_{k}": v for t in bt for k, v in bt[t].items() if k != "n"},
                         **{f"{t}_n": bt[t]["n"] for t in bt}})
        g = defaultdict(list)
        for r in map(json.loads, open(f"{d}/scores.jsonl", encoding="utf-8")):
            g[(recs[r["id"]].get("source") or "?", r["task_type"])].append(r["score"])
        src_rows += [{"concurrency": c, "split": s, "source": a, "task": t, "n": len(v),
                      "mean": round(sum(v) / len(v), 4)} for (a, t), v in sorted(g.items())]
report = {"model": model, "mode": mode, "crops": len(recs), "timing": timing, "mean_accept_len": accept,
          "speed": sweep, "accuracy": acc_rows, "by_source": src_rows}
json.dump(report, open(f"{out}/report.json", "w"), indent=1, ensure_ascii=False)

L = [f"# GravityOCR eval: {model} ({mode})", "", f"{len(recs)} crops. Timing: " +
     ", ".join(f"{k} {v}" for k, v in timing.items()) + (f". Mean accept length (serve.log): {accept}." if accept else "."), "",
     "## Speed (measured pass" + (f"; warm-up c={sweep['warmup']['concurrency']} {sweep['warmup']['wall_s']} s discarded" if sweep["warmup"] else "") + ")",
     "| c | wall s | crops/s | out tok/s | lat mean | p50 | p90 | max | errors |", "|---|---|---|---|---|---|---|---|---|"]
L += [f"| {r['concurrency']} | {r['wall_s']} | {r['crops_per_s']} | {r['out_tok_per_s']} | {r['lat_mean_s']} | {r['lat_p50_s']} | "
      f"{r['lat_p90_s']} | {r['lat_max_s']} | {r['errors']} |" for r in sweep["levels"]]
if acc_rows:
    L += ["", "## Accuracy (OmniDocBench metrics; Overall = mean of text 1-Edit_dist, table TEDS, formula CDM, x100)",
          "| c | split | Overall | text 1-ED | text BLEU | table TEDS | TEDS-S | formula CDM | formula ED | CDM errors |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    f = lambda r, k, d=4: "–" if r.get(k) is None else round(r[k], d)
    for r in acc_rows:
        ted = None if r.get("text_Edit_dist") is None else 1 - r["text_Edit_dist"]
        note = "" if r["overall_tasks"] == "text+table+formula" else f" ({r['overall_tasks']})"
        L.append(f"| {r['concurrency']} | {r['split']} | {r['Overall']}{note} | {f({'x': ted}, 'x')} | {f(r, 'text_BLEU')} | "
                 f"{f(r, 'table_TEDS')} | {f(r, 'table_TEDS_structure_only')} | {f(r, 'formula_CDM')} | "
                 f"{f(r, 'formula_Edit_dist')} | {r.get('formula_cdm_errors', '–')} |")
    L += ["", "## By source (headline metric: text 1-Edit_dist, table TEDS, formula CDM)",
          "| c | split | source | task | n | mean |", "|---|---|---|---|---|---|"]
    L += [f"| {r['concurrency']} | {r['split']} | {r['source']} | {r['task']} | {r['n']} | {r['mean']} |" for r in src_rows]
open(f"{out}/report.md", "w").write("\n".join(L) + "\n")
print("\n".join(L))
PY
log "done: $OUT/report.md"
