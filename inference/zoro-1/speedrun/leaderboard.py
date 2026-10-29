#!/usr/bin/env python3
"""Zoro-1 inference speedrun: check a run against the rules, record it, and render the leaderboard.

A run is one `eval.py gsm8k` at client concurrency 1 with the fixed sampling below; its summary.json carries
everything checked here. The record is the run's wall-clock (`wall_s`, lower is better). It counts only if accuracy
stays at or above the floor and it beats the track's current record.

  python speedrun/leaderboard.py add results/gsm8k_<stamp>/summary.json --track h100 \\
      --description "self-spec b15_g8, SGLang" --log <url> --contributors @you
  python speedrun/leaderboard.py render     # README tables + speedrun/<track>.svg from speedrun/records/*.csv
  python speedrun/leaderboard.py check      # exit 1 if the README or a chart is out of date (CI runs this)

Standard library only.
"""
import argparse
import csv
import datetime
import html
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
README = HERE.parent / "README.md"
RECORDS = HERE / "records"

TASK = "gsm8k"
ACCURACY_FLOOR = 87.0   # % on the track's GSM8K subsample (seed 42); the model card reports 87.5-91.5 across settings
SETTINGS = {"max_tokens": 8192, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0,
            "thinking": True}
TRACKS = {
    "h100": {"name": "1× NVIDIA H100", "n": 200},
    "l40s": {"name": "1× NVIDIA L40S", "n": 200},
    "mlx": {"name": "Apple Silicon (MLX)", "n": 50},
}
FIELDS = ["date", "wall_s", "accuracy_pct", "tok_per_s_per_request", "description", "log", "contributors"]


def load(track):
    path = RECORDS / f"{track}.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def check_run(summary, track, records):
    """-> list of rule violations (empty = the run is a new record)."""
    spec, errs = TRACKS[track], []
    if summary.get("task") != TASK:
        errs.append(f"task must be {TASK}, got {summary.get('task')!r}")
    if summary.get("n") != spec["n"]:
        errs.append(f"{track} runs {spec['n']} problems (eval.py --num-problems {spec['n']}), got n={summary.get('n')}")
    if summary.get("client_concurrency") != 1:
        errs.append(f"run at client concurrency 1 (eval.py --max-workers 1), got {summary.get('client_concurrency')}")
    if summary.get("errors"):
        errs.append(f"{summary['errors']} requests failed; a record needs every request answered")
    got = summary.get("settings") or {}
    for k, v in SETTINGS.items():
        if got.get(k) != v:
            errs.append(f"settings.{k} must be {v!r}, got {got.get(k)!r}")
    acc = summary.get("accuracy_pct", 0)
    if acc < ACCURACY_FLOOR:
        errs.append(f"accuracy {acc}% is below the {ACCURACY_FLOOR}% floor")
    best = min((float(r["wall_s"]) for r in records), default=None)
    if best is not None and summary.get("wall_s", float("inf")) >= best:
        errs.append(f"wall_s {summary.get('wall_s')} does not beat the current {track} record ({best} s)")
    return errs


def table(track, records):
    rows = ["| # | Wall-clock | GSM8K | tok/s per request | Description | Date | Log | Contributors |",
            "|---|---|---|---|---|---|---|---|"]
    for i, r in enumerate(records, 1):
        log = f"[log]({r['log']})" if r["log"] else "–"
        rows.append(f"| {i} | {float(r['wall_s']):.1f} s | {float(r['accuracy_pct']):.1f}% | "
                    f"{float(r['tok_per_s_per_request']):.0f} | {r['description']} | {r['date']} | {log} | "
                    f"{r['contributors']} |")
    if not records:
        rows.append("| – | no record yet: be the first | | | | | | |")
    return "\n".join(rows)


def chart(track, records, w=640, h=240, pad=48):
    """Record wall-clock over the record sequence, as a step line (SVG, light and dark readable)."""
    title = f"Zoro-1 speedrun · {TRACKS[track]['name']}"
    head = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(title)}">'
            f'<rect width="{w}" height="{h}" fill="#fff"/>'
            f'<text x="{pad}" y="24" font-family="sans-serif" font-size="14" fill="#222">{html.escape(title)}</text>')
    if not records:
        return head + (f'<text x="{w / 2}" y="{h / 2}" text-anchor="middle" font-family="sans-serif" font-size="13" '
                       f'fill="#666">no record yet</text></svg>\n')
    ys = [float(r["wall_s"]) for r in records]
    top, n = max(ys) * 1.1, len(ys)
    x = lambda i: pad + (w - 2 * pad) * (i / max(n - 1, 1))  # noqa: E731
    y = lambda v: h - pad + (pad * 2 - h) * (v / top)  # noqa: E731
    pts = []
    for i, v in enumerate(ys):
        if i:
            pts.append(f"{x(i):.1f},{y(ys[i - 1]):.1f}")
        pts.append(f"{x(i):.1f},{y(v):.1f}")
    body = [f'<line x1="{pad}" y1="{h - pad}" x2="{w - pad}" y2="{h - pad}" stroke="#999"/>',
            f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{h - pad}" stroke="#999"/>',
            f'<text x="12" y="{pad - 8}" font-family="sans-serif" font-size="11" fill="#666">wall-clock (s)</text>',
            f'<polyline points="{" ".join(pts)}" fill="none" stroke="#2563eb" stroke-width="2"/>']
    for i, v in enumerate(ys):
        body.append(f'<circle cx="{x(i):.1f}" cy="{y(v):.1f}" r="3.5" fill="#2563eb"/>')
        body.append(f'<text x="{x(i):.1f}" y="{y(v) - 8:.1f}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11" fill="#222">{v:.0f}</text>')
        body.append(f'<text x="{x(i):.1f}" y="{h - pad + 16}" text-anchor="middle" font-family="sans-serif" '
                    f'font-size="11" fill="#666">#{i + 1}</text>')
    return head + "".join(body) + "</svg>\n"


def rendered_readme(text):
    for track in TRACKS:
        block = f"<!-- speedrun:{track} -->\n{table(track, load(track))}\n<!-- /speedrun:{track} -->"
        text, k = re.subn(rf"<!-- speedrun:{track} -->.*?<!-- /speedrun:{track} -->", lambda _m: block, text,
                          flags=re.S)
        if k != 1:
            raise SystemExit(f"README.md needs exactly one <!-- speedrun:{track} --> ... <!-- /speedrun:{track} --> block")
    return text


def render():
    README.write_text(rendered_readme(README.read_text(encoding="utf-8")), encoding="utf-8")
    for track in TRACKS:
        (HERE / f"{track}.svg").write_text(chart(track, load(track)), encoding="utf-8")


def stale():
    """-> files that `render` would change."""
    out = []
    if rendered_readme(README.read_text(encoding="utf-8")) != README.read_text(encoding="utf-8"):
        out.append(str(README))
    for track in TRACKS:
        svg = HERE / f"{track}.svg"
        if not svg.exists() or svg.read_text(encoding="utf-8") != chart(track, load(track)):
            out.append(str(svg))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("add", help="check a run against the rules and append it as a record")
    a.add_argument("summary", help="summary.json written by eval.py")
    a.add_argument("--track", required=True, choices=list(TRACKS))
    a.add_argument("--description", required=True, help="what changed, in one line (e.g. 'self-spec b15_g8, SGLang')")
    a.add_argument("--log", default="", help="link to the run's summary.json + requests.jsonl (PR, gist, ...)")
    a.add_argument("--contributors", required=True, help="e.g. '@you, @them'")
    a.add_argument("--date", default=datetime.date.today().isoformat())
    sub.add_parser("render", help="rewrite the README tables and charts from speedrun/records/")
    sub.add_parser("check", help="exit 1 if the README tables or charts are out of date")
    args = ap.parse_args()

    if args.cmd == "add":
        summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
        records = load(args.track)
        errs = check_run(summary, args.track, records)
        if errs:
            sys.exit("not a record:\n  - " + "\n  - ".join(errs))
        row = {"date": args.date, "wall_s": summary["wall_s"], "accuracy_pct": summary["accuracy_pct"],
               "tok_per_s_per_request": summary["tok_per_s_per_request"], "description": args.description,
               "log": args.log, "contributors": args.contributors}
        with open(RECORDS / f"{args.track}.csv", "a", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, FIELDS).writerow(row)
        render()
        print(f"new {args.track} record: {summary['wall_s']} s at {summary['accuracy_pct']}%")
    elif args.cmd == "render":
        render()
    else:
        out = stale()
        if out:
            sys.exit("out of date (run `python speedrun/leaderboard.py render`): " + ", ".join(out))


if __name__ == "__main__":
    main()
