"""A minimal local agent loop against the on-device server (stdlib only).

    python -m trida_mlx.server --model ./Trida2.0-4B-mlx-q8 &      # terminal 1
    python -m trida_mlx.agent --workdir ~/some/project                # terminal 2
    python -m trida_mlx.agent --workdir . --allow-shell "how many python files are here?"

Tools: list_dir, read_file, search_files, calculator, now, and (opt-in) run_shell /
write_file — both ask for confirmation unless --yes. Any OpenAI-compatible client works
the same way; this file only exists to exercise tool calling end to end on device.
"""
from __future__ import annotations

import argparse
import ast
import datetime as _dt
import json
import operator
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

DIM, BOLD, CYAN, RESET = "\033[2m", "\033[1m", "\033[36m", "\033[0m"


def _fn(name, desc, props, required=()):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": props, "required": list(required)}}}


class Tools:
    def __init__(self, workdir: Path, allow_shell: bool, allow_write: bool, yes: bool):
        self.root, self.allow_shell, self.allow_write, self.yes = workdir.resolve(), allow_shell, allow_write, yes

    def specs(self):
        s = [
            _fn("list_dir", "List files in a directory (relative to the workspace).", {"path": {"type": "string"}}),
            _fn("read_file", "Read a text file (relative to the workspace). Returns at most max_chars.",
                {"path": {"type": "string"}, "max_chars": {"type": "integer"}}, ["path"]),
            _fn("search_files", "Find lines matching a substring in files under a directory.",
                {"query": {"type": "string"}, "path": {"type": "string"}}, ["query"]),
            _fn("calculator", "Evaluate an arithmetic expression, e.g. '17*24+3'.",
                {"expression": {"type": "string"}}, ["expression"]),
            _fn("now", "Current local date and time.", {}),
        ]
        if self.allow_shell:
            s.append(_fn("run_shell", "Run a shell command in the workspace; returns stdout/stderr.",
                         {"command": {"type": "string"}}, ["command"]))
        if self.allow_write:
            s.append(_fn("write_file", "Write a text file (relative to the workspace).",
                         {"path": {"type": "string"}, "content": {"type": "string"}}, ["path", "content"]))
        return s

    def _p(self, rel):
        p = (self.root / (rel or ".")).resolve()
        if self.root not in (p, *p.parents):
            raise ValueError("path escapes the workspace")
        return p

    def _confirm(self, what):
        if self.yes:
            return True
        return input(f"{BOLD}allow {what}? [y/N] {RESET}").strip().lower() == "y"

    def call(self, name, args):
        try:
            if name == "list_dir":
                p = self._p(args.get("path"))
                return "\n".join(sorted((e.name + ("/" if e.is_dir() else "")) for e in p.iterdir())[:500])
            if name == "read_file":
                return self._p(args["path"]).read_text(errors="replace")[: int(args.get("max_chars") or 8000)]
            if name == "search_files":
                q, hits = args["query"], []
                for f in self._p(args.get("path")).rglob("*"):
                    if f.is_file() and f.stat().st_size < 1_000_000 and ".git" not in f.parts:
                        try:
                            for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                                if q in line:
                                    hits.append(f"{f.relative_to(self.root)}:{i}: {line.strip()[:200]}")
                        except Exception:
                            pass
                    if len(hits) >= 50:
                        break
                return "\n".join(hits) or "(no matches)"
            if name == "calculator":
                return str(_safe_eval(args["expression"]))
            if name == "now":
                return _dt.datetime.now().isoformat(timespec="seconds")
            if name == "run_shell" and self.allow_shell:
                if not self._confirm(f"shell `{args['command']}`"):
                    return "user denied"
                r = subprocess.run(args["command"], shell=True, cwd=self.root, capture_output=True, text=True,
                                   timeout=60)
                return (r.stdout + ("\n[stderr]\n" + r.stderr if r.stderr else ""))[-8000:] or f"(exit {r.returncode})"
            if name == "write_file" and self.allow_write:
                p = self._p(args["path"])
                if not self._confirm(f"write {p}"):
                    return "user denied"
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(args["content"])
                return f"wrote {len(args['content'])} chars to {args['path']}"
            return f"unknown tool {name}"
        except Exception as e:  # noqa: BLE001
            return f"error: {type(e).__name__}: {e}"


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
        ast.Pow: operator.pow, ast.Mod: operator.mod, ast.FloorDiv: operator.floordiv, ast.USub: operator.neg}


def _safe_eval(expr):
    def ev(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        raise ValueError("unsupported expression")
    return ev(ast.parse(expr, mode="eval").body)


def stream_chat(base, body):
    req = urllib.request.Request(f"{base}/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    msg = {"role": "assistant", "content": "", "reasoning_content": "", "tool_calls": []}
    usage, in_reason = None, False
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            for ch in obj.get("choices", []):
                d = ch.get("delta", {})
                if d.get("reasoning_content"):
                    if not in_reason:
                        sys.stdout.write(DIM); in_reason = True
                    sys.stdout.write(d["reasoning_content"]); msg["reasoning_content"] += d["reasoning_content"]
                if d.get("content"):
                    if in_reason:
                        sys.stdout.write(RESET + "\n"); in_reason = False
                    sys.stdout.write(d["content"]); msg["content"] += d["content"]
                if d.get("tool_calls"):
                    msg["tool_calls"].extend(d["tool_calls"])
                sys.stdout.flush()
    if in_reason:
        sys.stdout.write(RESET)
    print()
    return msg, usage


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("task", nargs="*", help="one-shot task; omit for an interactive session")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--workdir", default=".")
    ap.add_argument("--allow-shell", action="store_true")
    ap.add_argument("--allow-write", action="store_true")
    ap.add_argument("--yes", action="store_true", help="don't ask before shell/write tools")
    ap.add_argument("--max-steps", type=int, default=12)
    ap.add_argument("--no-think", action="store_true")
    ap.add_argument("--temperature", type=float, default=None)
    a = ap.parse_args(argv)
    tools = Tools(Path(a.workdir).expanduser(), a.allow_shell, a.allow_write, a.yes)
    messages = [{"role": "system", "content":
                 "You are a helpful on-device assistant. Use the provided tools when they help; the workspace "
                 f"root is {tools.root}. Paths are relative to it. Be concise."}]

    def run(task):
        messages.append({"role": "user", "content": task})
        for step in range(a.max_steps):
            body = {"messages": messages, "tools": tools.specs(), "stream": True,
                    "chat_template_kwargs": {"enable_thinking": not a.no_think}}
            if a.temperature is not None:
                body["temperature"] = a.temperature
            msg, usage = stream_chat(a.base_url, body)
            if usage and usage.get("trida"):
                t = usage["trida"]
                print(f"{DIM}[{t['mode']}: {t['new_tokens']} tok, {t['decode_tok_s']} tok/s, "
                      f"{t['tokens_per_forward']} tok/fwd, prompt {t['prompt_tokens']} "
                      f"(cached {t['reused_tokens']}), prefill {t['prefill_s']}s]{RESET}")
            entry = {"role": "assistant", "content": msg["content"]}
            if msg["tool_calls"]:
                entry["tool_calls"] = [{"id": tc["id"], "type": "function", "function": tc["function"]}
                                       for tc in msg["tool_calls"]]
            messages.append(entry)
            if not msg["tool_calls"]:
                return
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except Exception:
                    args = {}
                print(f"{CYAN}→ {fn['name']}({json.dumps(args, ensure_ascii=False)[:200]}){RESET}")
                result = tools.call(fn["name"], args)
                print(f"{DIM}{result[:600]}{'…' if len(result) > 600 else ''}{RESET}")
                messages.append({"role": "tool", "tool_call_id": tc["id"], "name": fn["name"], "content": result})
        print("(max steps reached)")

    if a.task:
        run(" ".join(a.task))
        return
    print(f"{BOLD}trida on-device agent{RESET} — workspace {tools.root}. Ctrl-D to quit.")
    while True:
        try:
            task = input(f"{BOLD}> {RESET}").strip()
        except EOFError:
            break
        if task:
            run(task)


if __name__ == "__main__":
    main()
