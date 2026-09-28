#!/usr/bin/env python3
"""nano-inference — serve a diffusion LLM.

Three decode modes over the SAME checkpoint (one set of weights):

  causal      native autoregressive
  diffusion   parallel block-diffusion (confidence-based denoising)
              runtime block_size 3 == block B=4 (a carried seed + 2 masks)
  self-spec   self-speculative: diffusion draft + autoregressive verify (block 7 / gen 4)
              DEFAULT — the recommended mode for the reference model (lossless vs causal)

Usage:
  python serve.py --port 30000                                    # $TRIDA_MODEL, self-spec
  python serve.py trillionlabs/Trida2.0-4B --mode causal --port 30000   # HF id or local path

The model is the optional positional argument; when omitted it falls back to
the TRIDA_MODEL env var, then to the default `trillionlabs/Trida2.0-4B`. It is
passed straight to the backend's --model-path, so a Hugging Face repo id is
auto-downloaded; a local checkpoint directory works too. Private HF repos (the default is private during preview) need `huggingface-cli login` or HF_TOKEN in the environment first.

Dependency seam: this wraps the diffusion-serving SGLang backend. Point
SGLANG_PYTHON at the interpreter that has it installed (defaults to current python).
"""
import argparse
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

CONFIGS = Path(__file__).parent / "configs"

# Default checkpoint: positional arg > $TRIDA_MODEL > this id.
DEFAULT_MODEL = "trillionlabs/Trida2.0-4B"

# mode -> (backend algorithm id, config filename or None, extra server flags)
# The algorithm ids are the SGLang backend's --dllm-algorithm identifiers (API
# contract with the serving backend), not user-facing names.
MODES = {
    # two-stream (this stack)
    "causal":            (None, None, []),
    "diffusion":         ("LowConfidenceShiftHybridDiffusion", "trida_diffusion_b4.yaml", ["--skip-server-warmup"]),
    "self-spec":         ("HybridDiffusionSelfSpec", "trida_self_spec_b7_g4.yaml", []),
    # other block-diffusion families the backend serves
    "sdar":              ("LowConfidence", "sdar_b4_dynamic.yaml", ["--disable-radix-cache"]),
    "llada2-0":          ("LowConfidence", "llada2_0_b32.yaml", ["--disable-radix-cache"]),
    "llada2-1-speed":    ("JointThreshold", "llada2_1_b32_speed.yaml", ["--disable-radix-cache"]),
    "llada2-1-quality":  ("JointThreshold", "llada2_1_b32_quality.yaml", ["--disable-radix-cache"]),
}


def build_command(args):
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
        description="Serve a diffusion LLM (causal / diffusion / self-spec).")
    ap.add_argument("model", nargs="?", default=None,
                    help="HF repo id (auto-downloaded) or local checkpoint path "
                         f"(default: $TRIDA_MODEL, else {DEFAULT_MODEL})")
    ap.add_argument("--mode", choices=list(MODES), default="self-spec",
                    help="decode mode (default self-spec: the recommended mode for Trida2.0-4B)")
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
    ap.add_argument("--dry-run", action="store_true", help="print the launch command and exit")
    ap.add_argument("passthrough", nargs="*",
                    help="extra flags forwarded verbatim to sglang.launch_server")
    args = ap.parse_args()
    if args.model is None:
        args.model = os.environ.get("TRIDA_MODEL", DEFAULT_MODEL)

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
