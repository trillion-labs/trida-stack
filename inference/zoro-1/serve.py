#!/usr/bin/env python3
"""nano-inference for Zoro-1 — serve trillionlabs/Zoro-1, one checkpoint for generation and decisions.

Zoro-1 is a two-stream block-diffusion agent model (Qwen3.5-4B) with a typed-decision (Jev) head. Modes over the
SAME weights:

  causal      native autoregressive (the reference for self-spec)
  self-spec   self-speculative: diffusion draft + autoregressive verify, block 15 / gen 8 (b15_g8, the setting the
              model card reports). DEFAULT; lossless vs causal at matched sampling settings
  decide      typed decisions, reason-then-decide and OpenAI-compatible chat from one in-process engine
              (serve_decide.py, causal decoding on stock SGLang >= 0.5.22)

Zoro-1 does not support pure iterative diffusion decoding, so there is no diffusion mode.

Usage:
  python serve.py --port 30000                       # trillionlabs/Zoro-1, self-spec
  python serve.py --mode causal --port 30000
  python serve.py --mode decide --port 30001
  python serve.py /path/to/local/zoro-1 --port 30000 # a local copy, or $TRIDA_MODEL

The model is the optional positional argument; when omitted it falls back to $TRIDA_MODEL (for decide, first
$TRIDA_DECIDE_MODEL), then to `trillionlabs/Zoro-1`. It is passed straight to the backend, so a Hugging Face id is
auto-downloaded. Zoro-1 is gated: request access on its model page, then `hf auth login` or set HF_TOKEN.

SGLang loads Zoro-1 through the two-stream architecture, so for causal / self-spec the checkpoint is shaped once
(sglang/tools/shape_for_sglang.py): an HF id is downloaded, then a shim with a rewritten config.json and symlinked
weights is written to $TRIDA_SHAPED_DIR (default ~/.cache/trida/sglang) and served. --no-shape serves the path as is.

Dependency seam: causal and self-spec wrap the diffusion-serving SGLang backend (sglang/README.md). Point
SGLANG_PYTHON at the interpreter that has it installed (defaults to current python). decide runs on stock SGLang
instead: DECIDE_PYTHON names that interpreter (defaults to current python).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

CONFIGS = Path(__file__).parent / "configs"

# Default checkpoint: positional arg > $TRIDA_MODEL > this id.
DEFAULT_MODEL = "trillionlabs/Zoro-1"

# mode -> (backend algorithm id, config filename or None, extra server flags)
# The algorithm ids are the SGLang backend's --dllm-algorithm identifiers (API
# contract with the serving backend), not user-facing names.
MODES = {
    "causal":     (None, None, []),
    "self-spec":  ("HybridDiffusionSelfSpec", "zoro1_self_spec_b15_g8.yaml", []),
}


# Served by serve_decide.py rather than sglang.launch_server; not a decode algorithm, so not in MODES.
DECIDE_MODE = "decide"


def _shape_tool():
    spec = importlib.util.spec_from_file_location("shape_for_sglang",
                                                  Path(__file__).parent / "sglang" / "tools" / "shape_for_sglang.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def for_sglang(model, cache=None):
    """The checkpoint directory the SGLang backend should load: `model` itself when it is already shaped, else a
    shim of it under `cache` (made once, reused after). An HF id is downloaded first (the HF cache is reused)."""
    raw = Path(model)
    if not raw.is_dir():
        from huggingface_hub import snapshot_download
        raw = Path(snapshot_download(model))
    raw = raw.resolve()
    tool = _shape_tool()
    if tool.is_shaped(json.loads((raw / "config.json").read_text())):
        return str(raw)
    cache = Path(cache or os.environ.get("TRIDA_SHAPED_DIR", Path.home() / ".cache" / "trida" / "sglang"))
    # an HF snapshot dir is named by its revision; keep the repo name too so the cache reads well
    if Path(model).is_dir():
        name = f"{raw.name}-{hashlib.sha1(str(raw).encode()).hexdigest()[:8]}"
    else:
        name = f"{model.replace('/', '--')}-{raw.name[:12]}"
    out = cache / name
    if out.is_dir() and tool.is_shaped(json.loads((out / "config.json").read_text())):
        return str(out)
    if out.exists():
        import shutil
        shutil.rmtree(out)
    tool.shape(raw, out)
    print(f"[serve] shaped {raw} -> {out}", flush=True)
    return str(out)


def build_decide_command(args):
    python = os.environ.get("DECIDE_PYTHON", sys.executable)
    return [python, str(Path(__file__).parent / "serve_decide.py"), args.model, "--host", args.host,
            "--port", str(args.port), "--mem-fraction", str(args.mem_fraction)] + args.passthrough


def build_command(args):
    if args.mode == DECIDE_MODE:
        return build_decide_command(args)
    algo, cfg_name, extra = MODES[args.mode]
    python = os.environ.get("SGLANG_PYTHON", sys.executable)
    cmd = [
        python, "-m", "sglang.launch_server",
        "--model-path", args.model,
        "--trust-remote-code",
        "--tp-size", str(args.tp),
        "--dtype", args.dtype,
        "--attention-backend", "flashinfer",
        "--mem-fraction-static", str(args.mem_fraction),
        "--max-running-requests", str(args.max_running),
        "--cuda-graph-bs", *args.cuda_graph_bs.split(),
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reasoning_parser:
        # server-side splits <think>…</think> into reasoning_content vs content
        cmd += ["--reasoning-parser", args.reasoning_parser]
    if algo:
        cfg = args.config or str(CONFIGS / cfg_name)
        cmd += ["--dllm-algorithm", algo, "--dllm-algorithm-config", cfg]
    cmd += extra + args.passthrough
    return cmd


def main():
    ap = argparse.ArgumentParser(
        description="Serve Zoro-1: generation (causal / self-spec) or typed decisions (decide).")
    ap.add_argument("model", nargs="?", default=None,
                    help="HF repo id (auto-downloaded) or local checkpoint path "
                         f"(default: $TRIDA_MODEL, else {DEFAULT_MODEL}; --mode decide: $TRIDA_DECIDE_MODEL)")
    ap.add_argument("--mode", choices=list(MODES) + [DECIDE_MODE], default="self-spec",
                    help="decode mode (default self-spec, b15_g8); decide serves typed decisions and chat "
                         "(serve_decide.py)")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (1 = one full model per GPU)")
    ap.add_argument("--mem-fraction", type=float, default=0.8)
    ap.add_argument("--max-running", type=int, default=32)
    ap.add_argument("--cuda-graph-bs", default="1 2 4 8")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--config", default=None, help="override the mode's dllm config YAML")
    ap.add_argument("--reasoning-parser", default=None,
                    help="server-side thinking parser (e.g. qwen3, deepseek-r1): splits "
                         "<think>...</think> into message.reasoning_content vs message.content")
    ap.add_argument("--log", default=None,
                    help="also write the server output to this file; feed it to `eval.py --server-log` "
                         "for tokens-per-forward")
    ap.add_argument("--no-stats", dest="stats", action="store_false",
                    help="self-spec: don't log the periodic tok/fwd + acceptance stats line")
    ap.add_argument("--no-shape", dest="shape", action="store_false",
                    help="causal / self-spec: serve the checkpoint as given, without shaping it for SGLang")
    ap.add_argument("--dry-run", action="store_true", help="print the launch command and exit (no download, no shaping)")
    ap.add_argument("passthrough", nargs="*",
                    help="extra flags forwarded verbatim to sglang.launch_server (serve_decide.py for decide)")
    args = ap.parse_args()
    if args.model is None and args.mode == DECIDE_MODE:
        args.model = os.environ.get("TRIDA_DECIDE_MODEL")   # Zoro-1 carries jev_head.pt, so the default works too
    if args.model is None:
        args.model = os.environ.get("TRIDA_MODEL", DEFAULT_MODEL)

    if args.mode in MODES and args.shape and not args.dry_run:
        args.model = for_sglang(args.model)
    cmd = build_command(args)
    print("[serve]", " ".join(shlex.quote(c) for c in cmd), flush=True)
    if args.dry_run:
        return
    if args.mode == "self-spec" and args.stats:
        # backend logs "[HybridDiffusionSelfSpec] ... tok/fwd=X, accept=Y%" every N forwards
        os.environ.setdefault("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TIMING", "1")
        os.environ.setdefault("SGLANG_HYBRID_DIFFUSION_SELF_SPEC_TIMING_INTERVAL", "100")
    if not args.log:
        os.execvp(cmd[0], cmd)
    # tee: stream to the terminal AND the log file (eval.py --server-log reads the file).
    # Without --log we exec, so the server inherits our pid and signals reach it directly. Here we
    # are a parent, so SIGTERM/SIGINT must be forwarded or `kill`/SLURM leaves the server holding
    # the GPU (an orphaned server blocks the next job with an out-of-memory error at startup).
    with open(args.log, "ab") as fh, subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT) as proc:
        def _forward(signum, _frame):
            proc.send_signal(signum)
        for _sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(_sig, _forward)
        try:
            for line in proc.stdout:
                sys.stdout.buffer.write(line); sys.stdout.flush()
                fh.write(line); fh.flush()
        except KeyboardInterrupt:
            proc.terminate()
        sys.exit(proc.wait())


if __name__ == "__main__":
    main()
