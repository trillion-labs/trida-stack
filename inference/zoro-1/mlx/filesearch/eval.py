#!/usr/bin/env python3
"""Zero-shot file search with Zoro-1's decision head: a small offline eval.

DecisionRAG (arXiv:2610.35868) retrieves by *navigating* a corpus's own hierarchy with a decision model:
at each level every candidate gets a yes/no question against one shared context ("is the answer likely
in candidate [n]?"), the top k are expanded, and a final multiple-choice question reranks the shortlist.
A folder tree is such a hierarchy. This script measures how well Zoro-1 does that, untrained, on a
snapshot of this repository and a set of questions whose answer file is known.

Methods (all rank files; a question is a hit when a gold file is ranked):
  bm25      keyword baseline over path + content (no model)
  flat      yes/no over every file (requests of --per-request), rescore the top 50 together, choice over the top 20
  nav       walk the tree: yes/no over a level's folders and files, expand the top --k-dirs folders, ...;
            then rescore the collected files together and choice-rerank the top 20 (the paper's procedure)
  nav+bm25  nav, with BM25's top 10 added to the final rescoring (the paper's "+ dense candidates")
  nav2      nav on a flattened tree (small folders show their files directly, so ~3 levels instead of 6),
            folders described by their main files' summaries, and a score-relative folder budget
  flatc     every file, but one multiple-choice question per request of --per-request files instead of one
            yes/no question per file (one forward per request); the top --keep-per-chunk of each request go to
            a final choice over the pool
  flatc+bm25  flatc, with BM25's top --bm25-extra files added to the final choice

Needs only the Python standard library and a running trida-mlx server with Zoro-1 and its jev head
(the Zoro-1 Chat app starts one on port 8080):

    python3 inference/zoro-1/mlx/filesearch/eval.py                                  # all methods, repo @ 2fa0507
    python3 inference/zoro-1/mlx/filesearch/eval.py --methods bm25,nav --limit 5     # a quick look
    python3 inference/zoro-1/mlx/filesearch/eval.py --root ~/notes --questions my_questions.jsonl

Questions are JSONL: {"id", "lang", "question", "gold": [paths relative to the corpus root]}.
Results: <out>/summary.json and <out>/<method>.jsonl (per-question ranks, scores, requests, latency).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import re
import statistics
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_REV = "2fa0507"          # feat/zoro-chat-app @ PR #69: the snapshot questions.jsonl was written against
SKIP_DIRS = {".git", "__pycache__", "node_modules", ".venv", ".build", "build", "dist"}
MAX_FILE_BYTES = 400_000
TASK = "Find where in this file tree the answer to the question is."


# ---- corpus ----------------------------------------------------------------------------------------

def load_git_snapshot(rev: str) -> dict:
    """{path: text or None (binary)} from `git archive <rev>` of the repository this script lives in."""
    top = subprocess.run(["git", "-C", str(HERE), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
                         check=True).stdout.strip()
    data = subprocess.run(["git", "-C", top, "archive", "--format=tar", rev], capture_output=True, check=True).stdout
    files = {}
    with tarfile.open(fileobj=io.BytesIO(data)) as tar:
        for m in tar.getmembers():
            if m.isfile() and not any(p in SKIP_DIRS for p in m.name.split("/")):
                files[m.name] = _text(tar.extractfile(m).read(MAX_FILE_BYTES))
    return files


def load_folder(root: Path) -> dict:
    files = {}
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_file() and not any(part in SKIP_DIRS or part.startswith(".") and part != ".github"
                                   for part in rel.parts[:-1]):
            with open(p, "rb") as f:
                files[rel.as_posix()] = _text(f.read(MAX_FILE_BYTES))
    return files


def _text(raw: bytes):
    if b"\0" in raw[:4096]:
        return None
    try:
        return raw.decode("utf-8")
    except UnicodeDecodeError:
        return None


_BULKY = re.compile(r"(\.lock|lock\.txt|\.ipynb|\.json|\.jsonl|\.csv|\.svg|\.pt)$")


class Tree:
    """Folders and files of the corpus, with one-line descriptions the model reads."""

    def __init__(self, files: dict):
        self.files = files
        self.children = defaultdict(list)          # folder ("" = root) -> child paths (folders end with "/")
        for path in files:
            parts = path.split("/")
            for i in range(len(parts)):
                parent = "/".join(parts[:i]) + ("/" if i else "")
                child = "/".join(parts[: i + 1]) + ("/" if i < len(parts) - 1 else "")
                if child not in self.children[parent]:
                    self.children[parent].append(child)
        for v in self.children.values():
            v.sort(key=lambda c: (not c.endswith("/"), c.lower()))

    def files_under(self, folder: str) -> list:
        return [p for p in self.files if p.startswith(folder)]

    def level(self, folder: str, size: int) -> list:
        """nav2: the folder's children, with the smallest subfolders replaced by their own children while the
        level stays within `size` candidates, so small folders cost no navigation step."""
        cands = list(self.children[folder])
        while True:
            dirs = sorted((c for c in cands if c.endswith("/")), key=lambda d: len(self.children[d]))
            d = next((d for d in dirs if len(cands) - 1 + len(self.children[d]) <= size), None)
            if d is None:
                return sorted(cands, key=lambda c: (not c.endswith("/"), c.lower()))
            cands.remove(d)
            cands += self.children[d]

    def describe_rich(self, node: str, n_files: int = 8) -> str:
        """nav2: a folder described by what its largest files say about themselves."""
        if not node.endswith("/"):
            return self.describe(node)
        under = self.files_under(node)
        readme = next((p for p in under if p[len(node):].lower().startswith("readme")), None)
        about = f" {summary(readme, self.files[readme], 160)}" if readme and self.files.get(readme) else ""
        main = sorted((p for p in under if self.files.get(p) and p != readme and not _BULKY.search(p)),
                      key=lambda p: len(self.files[p]), reverse=True)[:n_files]
        parts = "; ".join(f"{p[len(node):]}: {summary(p, self.files[p], 70)}" for p in main)
        return f"folder {node} ({len(under)} files).{about} Main files: {parts}"

    def describe(self, node: str) -> str:
        if node.endswith("/"):
            kids = [c[len(node):] for c in self.children[node]]
            n = len(self.files_under(node))
            readme = next((node + k for k in kids if k.lower().startswith("readme")), None)
            about = f" — {summary(readme, self.files[readme])}" if readme and self.files.get(readme) else ""
            shown = ", ".join(kids[:14]) + (", …" if len(kids) > 14 else "")
            return f"folder {node} ({n} files: {shown}){about}"
        text = self.files.get(node)
        return f"file {node} — {summary(node, text) if text else '(binary)'}"


_NOISE = re.compile(r"^(#!|import |from |@|copyright|spdx|license|// swift-tools|<\?xml|\{|\}|\[|\]|---$|```)", re.I)


def summary(path: str, text: str, limit: int = 220) -> str:
    """The first meaningful lines of a file (docstring, heading, leading comments, or first lines)."""
    out = []
    for line in text.splitlines()[:60]:
        s = line.strip().strip('"\'').lstrip("#/*-!; ").strip()
        if not s or _NOISE.match(line.strip()) or _NOISE.match(s):
            continue
        out.append(s)
        if sum(len(x) for x in out) > limit:
            break
    s = " ".join(out)
    return (s[:limit] + "…") if len(s) > limit else s or "(empty)"


# ---- BM25 ------------------------------------------------------------------------------------------

def tokens(text: str) -> list:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z0-9]+|[가-힣]+", text.lower())


class BM25:
    def __init__(self, docs: dict, k1=1.2, b=0.75):
        self.tf = {p: Counter(tokens(p.replace("/", " ") + " " + (t or ""))) for p, t in docs.items()}
        self.len = {p: sum(c.values()) for p, c in self.tf.items()}
        self.avg = sum(self.len.values()) / max(len(self.len), 1)
        df = Counter(w for c in self.tf.values() for w in c)
        n = len(self.tf)
        self.idf = {w: math.log(1 + (n - d + 0.5) / (d + 0.5)) for w, d in df.items()}
        self.k1, self.b = k1, b

    def rank(self, query: str) -> list:
        q = set(tokens(query))
        def score(p):
            c, L = self.tf[p], self.len[p]
            return sum(self.idf[w] * c[w] * (self.k1 + 1) / (c[w] + self.k1 * (1 - self.b + self.b * L / self.avg))
                       for w in q if w in c)
        return sorted(self.tf, key=score, reverse=True)


# ---- the decision model ----------------------------------------------------------------------------

class Judge:
    def __init__(self, base_url: str, per_request: int, timeout: float, candidates_first: bool = False,
                 choice_labels: str = "number"):
        self.url = base_url.rstrip("/") + "/decide"
        self.per_request, self.timeout = per_request, timeout
        self.candidates_first, self.choice_labels = candidates_first, choice_labels
        self.reset()

    def reset(self):
        self.requests, self.server_ms, self.state_tokens, self.cached_tokens = 0, 0.0, 0, 0

    def _post(self, body: dict) -> dict:
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"{e.code}: {e.read().decode(errors='replace')[:400]}") from None
        self.requests += 1
        d = out.get("direct", out)
        self.server_ms += d.get("timing_ms", 0.0)
        self.state_tokens += d.get("state_tokens", 0)
        self.cached_tokens += d.get("cached_tokens", 0)
        return out

    def _state(self, query: str, items: list) -> dict:
        cands = "\n".join(f"[{i + 1}] {desc}" for i, (_, desc) in enumerate(items))
        if self.candidates_first:      # the candidate list is a query-independent prefix a server can cache
            return {"candidates": cands, "task": TASK, "question": query}
        return {"task": TASK, "question": query, "candidates": cands}

    def binary(self, query: str, items: list) -> dict:
        """{node: P(yes)} for [(node, description)], all items of one request judged in one shared context."""
        out = {}
        for i in range(0, len(items), self.per_request):
            out.update(self._binary(query, items[i:i + self.per_request]))
        return out

    def _binary(self, query: str, items: list) -> dict:
        qs = {f"c{i + 1}": {"type": "noul", "instructions":
                            f"Is the information needed to answer the question likely in candidate [{i + 1}] ({node})?"}
              for i, (node, _) in enumerate(items)}
        try:
            ans = self._post({"state": self._state(query, items), "questions": qs, "reasoning": "never"})["answers"]
        except RuntimeError as e:
            if "state is" in str(e) and len(items) > 1:          # too long for one state: split in two
                h = len(items) // 2
                return {**self._binary(query, items[:h]), **self._binary(query, items[h:])}
            raise
        return {node: ans[f"c{i + 1}"]["noul"] for i, (node, _) in enumerate(items)}

    def choice(self, query: str, items: list) -> dict:
        """{node: P(chosen)} over [(node, description)] as one multiple-choice question."""
        q = {"best": {"type": "choice", "instructions": "Which candidate most likely contains the information needed "
                                                        "to answer the question?",
                      # "number": options are just the candidates' numbers (their descriptions are in the
                      # state), so the question row stays short; "path": each option also names the file
                      "criteria": {str(i + 1): (node if self.choice_labels == "path" else "")
                                   for i, (node, _) in enumerate(items)}}}
        try:
            ans = self._post({"state": self._state(query, items), "questions": q, "reasoning": "never"})["answers"]
        except RuntimeError as e:
            if "state is" in str(e) and len(items) > 2:
                return self.choice(query, items[: len(items) * 2 // 3])
            raise
        probs = ans["best"]["probabilities"]
        return {node: probs[str(i + 1)] for i, (node, _) in enumerate(items)}


# ---- methods ---------------------------------------------------------------------------------------

def finish(judge: Judge, tree: Tree, query: str, scored: dict, rescore: int, choose: int, extra=()) -> tuple:
    """The paper's last stages: rescore the top `rescore` files in one shared context, then reorder the top
    `choose` with a multiple-choice question. Returns (ranking, final scores)."""
    pool = sorted(scored, key=scored.get, reverse=True)[:rescore]
    pool += [p for p in extra if p not in pool]
    final = judge.binary(query, [(p, tree.describe(p)) for p in pool]) if len(pool) > 1 else dict(scored)
    ranked = sorted(final, key=final.get, reverse=True)
    if choose > 1 and len(ranked) > 1:
        top = ranked[:choose]
        pc = judge.choice(query, [(p, tree.describe(p)) for p in top])
        ranked = sorted(top, key=pc.get, reverse=True) + ranked[choose:]
    return ranked, final


def run_flat(judge, tree, query, args, bm25=None):
    files = list(tree.files)
    scored = judge.binary(query, [(p, tree.describe(p)) for p in files])
    ranked, _ = finish(judge, tree, query, scored, args.rescore, args.choose)
    return ranked, {"pool": len(files)}


def run_nav(judge, tree, query, args, bm25=None):
    frontier, pool, trace = [""], {}, []
    for depth in range(args.max_depth):
        kids = [c for f in frontier for c in tree.children[f]]
        if not kids:
            break
        s = judge.binary(query, [(c, tree.describe(c)) for c in kids])
        dirs = sorted((c for c in kids if c.endswith("/")), key=s.get, reverse=True)
        for c in kids:
            if not c.endswith("/"):
                pool[c] = s[c]
        frontier = dirs[: args.k_dirs]
        trace.append({"depth": depth, "candidates": len(kids), "kept": frontier,
                      "top": sorted(((round(s[c], 3), c) for c in kids), reverse=True)[:5]})
    for f in frontier:                       # depth limit reached: their files still count
        pool.update({p: 0.0 for p in tree.files_under(f) if p not in pool})
    extra = bm25.rank(query)[: args.bm25_extra] if bm25 is not None else ()
    ranked, _ = finish(judge, tree, query, pool, args.rescore, args.choose, extra)
    return ranked, {"pool": len(pool), "pooled": sorted(set(pool) | set(extra)), "trace": trace}


def run_flatc(judge, tree, query, args, bm25=None):
    files, pool = list(tree.files), {}
    for i in range(0, len(files), args.per_request):
        chunk = files[i:i + args.per_request]
        pc = judge.choice(query, [(p, tree.describe(p)) for p in chunk])
        for p in sorted(pc, key=pc.get, reverse=True)[: args.keep_per_chunk]:
            pool[p] = pc[p]
    top = sorted(pool, key=pool.get, reverse=True)[: max(args.choose, 2)]
    extra = [p for p in bm25.rank(query)[: args.bm25_extra] if p not in top] if bm25 is not None else []
    top += extra
    final = judge.choice(query, [(p, tree.describe(p)) for p in top])
    ranked = sorted(final, key=final.get, reverse=True) + [p for p in sorted(pool, key=pool.get, reverse=True)
                                                           if p not in final]
    return ranked, {"pool": len(pool), "pooled": sorted(set(pool) | set(extra)), "bm25_added": extra}


def run_nav2(judge, tree, query, args, bm25=None):
    frontier, pool, trace = [""], {}, []
    for depth in range(args.max_depth):
        kids = [c for f in frontier for c in tree.level(f, args.level_size)]
        if not kids:
            break
        s = judge.binary(query, [(c, tree.describe_rich(c)) for c in kids])
        dirs = sorted((c for c in kids if c.endswith("/")), key=s.get, reverse=True)
        for c in kids:
            if not c.endswith("/"):
                pool[c] = s[c]
        best = s[dirs[0]] if dirs else 0.0
        frontier = [d for i, d in enumerate(dirs) if i < args.k_dirs2 or s[d] >= args.rel * best][: args.k_dirs2 + 2]
        trace.append({"depth": depth, "candidates": len(kids), "kept": frontier,
                      "top": sorted(((round(s[c], 3), c) for c in kids), reverse=True)[:5]})
    for f in frontier:
        pool.update({p: 0.0 for p in tree.files_under(f) if p not in pool})
    ranked, _ = finish(judge, tree, query, pool, args.rescore, args.choose)
    return ranked, {"pool": len(pool), "pooled": sorted(pool), "trace": trace}


# ---- eval ------------------------------------------------------------------------------------------

def metrics(rows: list) -> dict:
    def rank(r):
        return min((r["ranking"].index(g) + 1 for g in r["gold"] if g in r["ranking"]), default=None)
    ranks = [rank(r) for r in rows]
    m = {"n": len(rows),
         "hit@1": sum(1 for x in ranks if x == 1) / len(rows),
         "hit@5": sum(1 for x in ranks if x and x <= 5) / len(rows),
         "hit@10": sum(1 for x in ranks if x and x <= 10) / len(rows),
         "mrr@10": sum(1 / x for x in ranks if x and x <= 10) / len(rows)}
    if any("pooled" in r for r in rows):
        m["gold_in_pool"] = sum(1 for r in rows if set(r["gold"]) & set(r.get("pooled", []))) / len(rows)
    lat = [r["seconds"] for r in rows if r.get("seconds") is not None]
    if lat:
        m["median_s"] = statistics.median(lat)
        m["requests_per_q"] = statistics.mean(r["requests"] for r in rows)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--root", type=Path, help="a folder to search (default: this repo at --rev)")
    ap.add_argument("--rev", default=DEFAULT_REV, help="git revision of this repo to snapshot when --root is not set")
    ap.add_argument("--questions", type=Path, default=HERE / "questions.jsonl")
    ap.add_argument("--methods", default="bm25,flat,flatc,nav,nav2")
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions")
    ap.add_argument("--k-dirs", type=int, default=4, help="folders kept per level")
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--per-request", type=int, default=40, help="candidates per yes/no request (shared context)")
    ap.add_argument("--rescore", type=int, default=50, help="files rescored together before the choice step")
    ap.add_argument("--choose", type=int, default=20, help="files reordered by the choice question (0 = off)")
    ap.add_argument("--bm25-extra", type=int, default=10)
    ap.add_argument("--level-size", type=int, default=60, help="nav2: candidates per level after flattening")
    ap.add_argument("--k-dirs2", type=int, default=6, help="nav2: folders kept per level (plus any within --rel of the best)")
    ap.add_argument("--rel", type=float, default=0.8, help="nav2: also keep folders scoring >= rel x the best folder")
    ap.add_argument("--candidates-first", action="store_true",
                    help="put the candidate list before the task and question in the state (cacheable prefix)")
    ap.add_argument("--choice-labels", choices=["number", "path"], default="number",
                    help="choice options as the candidates' numbers only, or numbers with paths (longer rows)")
    ap.add_argument("--keep-per-chunk", type=int, default=5, help="flatc: files kept from each request")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--out", type=Path, default=Path("filesearch_results") / time.strftime("%Y%m%d-%H%M%S"))
    args = ap.parse_args()

    files = load_folder(args.root.expanduser()) if args.root else load_git_snapshot(args.rev)
    tree = Tree(files)
    qs = [json.loads(line) for line in args.questions.read_text().splitlines() if line.strip()]
    qs = qs[: args.limit] if args.limit else qs
    missing = [(q["id"], g) for q in qs for g in q["gold"] if g not in files]
    if missing:
        sys.exit(f"gold files not in the corpus: {missing[:5]} — wrong --rev/--root?")
    print(f"corpus: {len(files)} files, {sum(1 for c in tree.children if c)} folders; {len(qs)} questions")

    bm25 = BM25(files)
    judge = Judge(args.base_url, args.per_request, args.timeout, args.candidates_first, args.choice_labels)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    if any(m != "bm25" for m in methods):
        try:
            judge.binary("ping", [("README.md", "file README.md")])
        except Exception as e:  # noqa: BLE001
            sys.exit(f"the decision endpoint {judge.url} is not answering ({e}); start the server with Zoro-1 "
                     "and its jev_head.pt, or pass --methods bm25")
    args.out.mkdir(parents=True, exist_ok=True)
    summary_out = {"corpus": str(args.root or f"git:{args.rev}"), "files": len(files), "args": {
        k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "methods": {}}

    for method in methods:
        rows = []
        with open(args.out / f"{method}.jsonl", "w") as log:
            for q in qs:
                judge.reset()
                t0 = time.perf_counter()
                if method == "bm25":
                    ranking, info, secs = bm25.rank(q["question"]), {}, None
                else:
                    fn = {"flat": run_flat, "flatc": run_flatc, "flatc+bm25": run_flatc, "nav2": run_nav2}.get(
                        method, run_nav)
                    ranking, info = fn(judge, tree, q["question"], args, bm25 if method.endswith("+bm25") else None)
                    secs = time.perf_counter() - t0
                row = {**q, "ranking": ranking[:20], "seconds": secs, "requests": judge.requests,
                       "server_ms": round(judge.server_ms, 1), "state_tokens": judge.state_tokens,
                       "cached_tokens": judge.cached_tokens, **info}
                hit = next((i + 1 for i, p in enumerate(ranking) if p in q["gold"]), None)
                cached = f"{judge.cached_tokens / judge.state_tokens:4.0%} cached  " if judge.state_tokens else ""
                print(f"  {method:9s} {q['id']} rank={hit or '-':>3}  {secs or 0:5.1f}s  {judge.requests:2d} req  {cached}"
                      f"top1={ranking[0] if ranking else None}")
                rows.append(row)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
        m = {"all": metrics(rows)}
        for lang in sorted({r.get("lang", "?") for r in rows}):
            m[lang] = metrics([r for r in rows if r.get("lang", "?") == lang])
        summary_out["methods"][method] = m

    (args.out / "summary.json").write_text(json.dumps(summary_out, indent=1, ensure_ascii=False))
    cols = ["hit@1", "hit@5", "hit@10", "mrr@10", "gold_in_pool", "median_s", "requests_per_q"]
    print(f"\n{'method':10s} {'lang':4s} " + " ".join(f"{c:>14s}" for c in cols))
    for method, by in summary_out["methods"].items():
        for lang, m in by.items():
            print(f"{method:10s} {lang:4s} " + " ".join(f"{m.get(c, ''):>14}" for c in cols))
    print(f"\nwrote {args.out}/summary.json and per-question logs")


if __name__ == "__main__":
    main()
