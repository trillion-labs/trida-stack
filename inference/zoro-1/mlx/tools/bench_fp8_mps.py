# /// script
# requires-python = ">=3.11"
# dependencies = ["torch>=2.15", "mlx>=0.29", "numpy"]
# ///
"""FP8 on Apple Silicon: PyTorch MPS (2.15+) vs MLX, on Zoro-1's linear layers.

Answers issue #58: would serving Zoro-1 with PyTorch on MPS in FP8 beat our MLX q8 path?

Decoding a 4B model on a Mac is bound by memory bandwidth: every step reads every weight once,
and almost all of those bytes are the linear layers. So this times one pass over *all* of the
text model's linear layers (every decoder layer's projections and MLP, plus the LM head), at the
real shapes read from the model's config.json, with random weights:

  M = 1   one token per step (plain AR decoding)
  M = 16  one self-spec block (draft verify), the shape our server actually runs

Formats:
  mlx  bf16          the reference
  mlx  q8  (g64)     what the app ships (affine 8-bit, group 64)
  mlx  mxfp8         MLX's own FP8 format (if this MLX has it)
  torch bf16         F.linear on MPS
  torch fp8 rowwise  float8_e4m3fn weights + activations, per-row scales, _scaled_mm -> bf16
  torch fp8 tensor   same, one scale per tensor

It also reports each format's output error against an fp32 matmul of the same weights, and the
weight bytes read per pass. It does not time attention / GatedDeltaNet / the self-spec loop:
those are the same work whatever the weight format, and differ by framework, not by FP8.

    uv run --script inference/zoro-1/mlx/tools/bench_fp8_mps.py \
        --config ~/models/Zoro-1-mlx-q8/config.json

Needs macOS 15+ and torch >= 2.15 for the FP8 rows; formats that fail are reported, not fatal.
`--dry-run` prints the shapes and bytes only (works anywhere).
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import statistics
import subprocess
import time
from pathlib import Path

DEFAULT_CONFIG = "~/models/Zoro-1-mlx-q8/config.json"


# ----------------------------------------------------------------------------------- shapes
def linear_shapes(cfg: dict) -> list[tuple[str, int, int]]:
    """(name, in_features, out_features) for every linear layer of the text model, in order."""
    t = cfg.get("text_config", cfg)
    h, inter, n = t["hidden_size"], t["intermediate_size"], t["num_hidden_layers"]
    if "layer_types" in t:
        linear = [lt == "linear_attention" for lt in t["layer_types"]]
    else:
        every = t.get("full_attention_interval", 4)
        linear = [(i + 1) % every != 0 for i in range(n)]
    out = []
    for i in range(n):
        if linear[i]:  # GatedDeltaNet
            kd = t["linear_key_head_dim"] * t["linear_num_key_heads"]
            vd = t["linear_value_head_dim"] * t["linear_num_value_heads"]
            nv = t["linear_num_value_heads"]
            out += [(f"L{i}.in_proj_qkv", h, 2 * kd + vd), (f"L{i}.in_proj_z", h, vd),
                    (f"L{i}.in_proj_b", h, nv), (f"L{i}.in_proj_a", h, nv), (f"L{i}.out_proj", vd, h)]
        else:  # gated full attention: q_proj also produces the output gate
            hd, nh, nkv = t["head_dim"], t["num_attention_heads"], t["num_key_value_heads"]
            gate = 2 if t.get("attn_output_gate", True) else 1
            out += [(f"L{i}.q_proj", h, nh * hd * gate), (f"L{i}.k_proj", h, nkv * hd),
                    (f"L{i}.v_proj", h, nkv * hd), (f"L{i}.o_proj", nh * hd, h)]
        out += [(f"L{i}.gate_proj", h, inter), (f"L{i}.up_proj", h, inter), (f"L{i}.down_proj", inter, h)]
    out.append(("lm_head", h, t["vocab_size"]))
    return out


def weight_bytes(shapes, fmt: str) -> int:
    """Bytes read per pass. Tiny projections (b/a: out < 64) stay bf16 in every quantized format."""
    total = 0
    for _, i, o in shapes:
        if fmt == "bf16" or o < 64 or i % 64:
            total += 2 * i * o
        elif fmt == "q8":       # 1 byte + (bf16 scale + bf16 bias) per 64
            total += i * o + 4 * (i // 64) * o
        elif fmt == "mxfp8":    # 1 byte + one 8-bit exponent per 32
            total += i * o + (i // 32) * o
        elif fmt in ("fp8row", "fp8tensor"):  # 1 byte + fp32 scale per row (or per tensor)
            total += i * o + (4 * o if fmt == "fp8row" else 4)
    return total


# ----------------------------------------------------------------------------------- timing
def summarize(times: list[float]) -> float:
    return statistics.median(times) * 1e3  # ms


def rel_err(y, ref) -> float:
    import numpy as np
    y, ref = np.asarray(y, dtype=np.float32), np.asarray(ref, dtype=np.float32)
    return float(np.linalg.norm(y - ref) / (np.linalg.norm(ref) + 1e-12))


def bench_mlx(shapes, M, fmt, warmup, iters, seed):
    import mlx.core as mx
    mx.random.seed(seed)
    xs = {i: mx.random.normal((M, i)).astype(mx.bfloat16) for i in {s[1] for s in shapes}}
    layers = []
    for _, i, o in shapes:
        w = (mx.random.normal((o, i)) * (i ** -0.5)).astype(mx.bfloat16)
        if fmt == "bf16" or o < 64 or i % 64:
            layers.append(("dense", i, w))
        elif fmt == "q8":
            wq, s, b = mx.quantize(w, group_size=64, bits=8)
            layers.append(("q", i, (wq, s, b, dict(group_size=64, bits=8))))
        elif fmt == "mxfp8":
            wq, s, *rest = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
            b = rest[0] if rest else None
            layers.append(("q", i, (wq, s, b, dict(group_size=32, bits=8, mode="mxfp8"))))
        mx.eval(layers[-1][2])
        del w

    def run():
        outs = []
        for kind, i, p in layers:
            if kind == "dense":
                outs.append(xs[i] @ p.T)
            else:
                wq, s, b, kw = p
                outs.append(mx.quantized_matmul(xs[i], wq, s, b, transpose=True, **kw))
        mx.eval(outs)
        return outs

    for _ in range(warmup):
        run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)

    # accuracy on the LM head's neighbour: the first full-size MLP projection
    k = next(j for j, (n, i, o) in enumerate(shapes) if n.endswith("gate_proj"))
    _, i, o = shapes[k]
    mx.random.seed(seed + 1)
    w = (mx.random.normal((o, i)) * (i ** -0.5)).astype(mx.bfloat16)
    x = xs[i]
    ref = (x.astype(mx.float32) @ w.astype(mx.float32).T)
    if fmt == "bf16":
        y = x @ w.T
    elif fmt == "q8":
        wq, s, b = mx.quantize(w, group_size=64, bits=8)
        y = mx.quantized_matmul(x, wq, s, b, transpose=True, group_size=64, bits=8)
    else:
        wq, s, *rest = mx.quantize(w, group_size=32, bits=8, mode="mxfp8")
        y = mx.quantized_matmul(x, wq, s, rest[0] if rest else None, transpose=True,
                                group_size=32, bits=8, mode="mxfp8")
    err = rel_err(y.astype(mx.float32), ref)
    del layers, xs
    gc.collect()
    mx.clear_cache()
    return summarize(times), err


def bench_torch(shapes, M, fmt, warmup, iters, seed, device):
    import torch
    import torch.nn.functional as F
    torch.manual_seed(seed)
    dev = torch.device(device)
    f8 = torch.float8_e4m3fn
    F8MAX = torch.finfo(f8).max

    def sync():
        if dev.type == "mps":
            torch.mps.synchronize()

    def q_rows(t):  # per-row scale: t ~= q * s, s shape (rows, 1)
        s = t.float().abs().amax(dim=1, keepdim=True).clamp(min=1e-12) / F8MAX
        return (t.float() / s).to(f8), s

    def q_tensor(t):
        s = t.float().abs().amax().clamp(min=1e-12) / F8MAX
        return (t.float() / s).to(f8), s

    scaled_mm = getattr(torch, "_scaled_mm", None) or getattr(F, "scaled_mm")

    xs = {i: torch.randn(M, i, device=dev, dtype=torch.bfloat16) for i in {s[1] for s in shapes}}
    layers = []
    for _, i, o in shapes:
        w = (torch.randn(o, i, device=dev) * (i ** -0.5)).to(torch.bfloat16)
        if fmt == "bf16" or o < 64 or i % 64:
            layers.append(("dense", i, w))
        else:
            wq, ws = q_rows(w) if fmt == "fp8row" else q_tensor(w)
            # _scaled_mm wants mat2 column-major: pass W^T as a transposed view of row-major W
            layers.append(("fp8", i, (wq.t(), ws.reshape(1, -1) if fmt == "fp8row" else ws)))
        del w

    def mm(x, wqT, ws):
        xq, xs_ = q_rows(x) if fmt == "fp8row" else q_tensor(x)
        return scaled_mm(xq, wqT, scale_a=xs_, scale_b=ws, out_dtype=torch.bfloat16)

    def run():
        outs = []
        for kind, i, p in layers:
            outs.append(F.linear(xs[i], p) if kind == "dense" else mm(xs[i], *p))
        sync()
        return outs

    for _ in range(warmup):
        run()
    times = []
    for _ in range(iters):
        t0 = time.perf_counter()
        run()
        times.append(time.perf_counter() - t0)

    k = next(j for j, (n, i, o) in enumerate(shapes) if n.endswith("gate_proj"))
    _, i, o = shapes[k]
    torch.manual_seed(seed + 1)
    w = (torch.randn(o, i, device=dev) * (i ** -0.5)).to(torch.bfloat16)
    x = xs[i]
    ref = x.float() @ w.float().t()
    if fmt == "bf16":
        y = F.linear(x, w)
    else:
        wq, ws = q_rows(w) if fmt == "fp8row" else q_tensor(w)
        y = mm(x, wq.t(), ws.reshape(1, -1) if fmt == "fp8row" else ws)
    err = rel_err(y.float().cpu().numpy(), ref.cpu().numpy())
    del layers, xs
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    return summarize(times), err


# ----------------------------------------------------------------------------------- main
def machine() -> str:
    bits = [platform.platform()]
    try:
        bits.append(subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=5).stdout.strip())
        mem = int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True,
                                 timeout=5).stdout.strip())
        bits.append(f"{mem / 2**30:.0f} GB")
    except Exception:
        pass
    return " | ".join(b for b in bits if b)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG, help="the model's config.json (any Zoro-1 / Zoro dir)")
    ap.add_argument("--m", default="1,7,15", help="rows per matmul: 1 = AR decode, 7 = self-spec canvas at N=4 (the app), 15 = N=8")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default="", help="comma list of rows to run, e.g. mlx-q8,torch-fp8row")
    ap.add_argument("--torch-device", default="mps", help="mps (default); cpu for a smoke test")
    ap.add_argument("--json", default="", help="also write the results here")
    ap.add_argument("--dry-run", action="store_true", help="print shapes and bytes, run nothing")
    a = ap.parse_args()

    cfg = json.loads(Path(os.path.expanduser(a.config)).read_text())
    shapes = linear_shapes(cfg)
    rows = [("mlx", "bf16"), ("mlx", "q8"), ("mlx", "mxfp8"),
            ("torch", "bf16"), ("torch", "fp8row"), ("torch", "fp8tensor")]
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        rows = [r for r in rows if f"{r[0]}-{r[1]}" in want]

    print(f"machine: {machine()}")
    print(f"config:  {a.config}  ->  {len(shapes)} linear layers, "
          f"{sum(i * o for _, i, o in shapes) / 1e9:.2f} B weights")
    versions = []
    for mod in ("torch", "mlx.core"):
        try:
            m = __import__(mod, fromlist=["_"])
            versions.append(f"{mod.split('.')[0]} {m.__version__}")
        except Exception as e:
            versions.append(f"{mod.split('.')[0]} unavailable ({type(e).__name__})")
    print("versions: " + ", ".join(versions))
    if a.dry_run:
        for fw, fmt in rows:
            print(f"  {fw:5s} {fmt:9s} {weight_bytes(shapes, 'bf16' if fmt == 'bf16' else fmt) / 2**30:6.2f} GiB")
        return

    results = []
    Ms = [int(x) for x in a.m.split(",")]
    print(f"\n{'format':18s} {'M':>3s} {'ms/pass':>9s} {'GB/s':>7s} {'GiB':>6s} {'rel.err':>8s}  note")
    for M in Ms:
        base = None
        for fw, fmt in rows:
            name = f"{fw}-{fmt}"
            nbytes = weight_bytes(shapes, fmt)
            try:
                if fw == "mlx":
                    ms, err = bench_mlx(shapes, M, fmt, a.warmup, a.iters, a.seed)
                else:
                    ms, err = bench_torch(shapes, M, fmt, a.warmup, a.iters, a.seed, a.torch_device)
            except Exception as e:  # report and keep going: an unsupported format is a result too
                msg = f"{type(e).__name__}: {str(e).splitlines()[0][:110] if str(e) else ''}"
                print(f"{name:18s} {M:3d} {'-':>9s} {'-':>7s} {nbytes / 2**30:6.2f} {'-':>8s}  FAILED {msg}")
                results.append(dict(format=name, M=M, error=msg))
                continue
            if name == "mlx-q8":
                base = ms
            note = f"{base / ms:.2f}x vs mlx-q8" if base and name != "mlx-q8" else ""
            print(f"{name:18s} {M:3d} {ms:9.2f} {nbytes / ms / 1e6:7.0f} {nbytes / 2**30:6.2f} {err:8.4f}  {note}")
            results.append(dict(format=name, M=M, ms=ms, gbps=nbytes / ms / 1e6, gib=nbytes / 2**30, rel_err=err))
        print()
    print("ms/pass = one pass over every linear layer = the weight-reading part of one decode step.\n"
          "Lower is better. A full step adds attention/GDN, the same for every format.")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(machine=machine(), versions=versions, config=a.config,
                                                results=results), indent=2))
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()
