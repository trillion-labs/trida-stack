"""Where does a self-spec step spend its time? (run on the Mac)

    uv run trida-mlx-profile --model ~/models/Trida2.0-4B-mlx-q8

Times, at the same KV length: the AR step (1 token) vs the self-spec canvas step
(2N-1 tokens), each split into LM head / MLPs / full attention / gated-delta, plus the
Python graph-build time and the commit, and A/Bs the step optimizations (fused canvas GDN
kernel, LM-head row skipping on cold starts). Medians over --iters runs, GPU-synchronized.
"""
from __future__ import annotations

import argparse
import statistics
import time

import mlx.core as mx

from .engine import Engine
from .model import GDNState

LONG_PROMPT = [{"role": "user", "content": "Summarize the history of computing in detail. " * 40}]


def timeit(fn, iters, warmup=3):
    for _ in range(warmup):
        mx.eval(fn())
    ts = []
    for _ in range(iters):
        t = time.perf_counter()
        mx.eval(fn())
        ts.append(time.perf_counter() - t)
    return statistics.median(ts) * 1e3


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trillionlabs/Trida2.0-4B")
    ap.add_argument("--gen-block", type=int, default=4)
    ap.add_argument("--iters", type=int, default=20)
    a = ap.parse_args(argv)
    eng = Engine(a.model, gen_block=a.gen_block, prompt_cache=False)
    rt, n, mask = eng.rt, a.gen_block, eng.mask_id
    blk = 2 * n - 1
    prompt = eng.encode(eng.render(LONG_PROMPT, enable_thinking=False))
    cache = rt.make_cache()
    rt.prefill(cache, prompt)
    mx.eval([l.ssm if isinstance(l, GDNState) else l.keys for l in cache.layers])
    P = cache.length
    D = rt.args.hidden_size
    layers = rt.backbone.layers
    print(f"prefix length {P}, canvas {blk} tokens, device {mx.default_device()}")

    rows = []

    def row(name, ms1, ms7, note=""):
        rows.append((name, ms1, ms7, note))

    # ---- whole steps -------------------------------------------------------
    def ar_step():
        logits = rt.ar_step(cache, 42)
        tok = mx.argmax(logits)
        mx.eval(tok)
        return tok

    def trim_ar():
        # undo the appended token so every run sees the same prefix
        for i, l in enumerate(cache.layers):
            if not isinstance(l, GDNState):
                l.trim(l.offset - P)
        cache.tokens = cache.tokens[:P]

    snap_states = [l for l in cache.layers]

    def restore():
        trim_ar()
        for i, s in enumerate(snap_states):
            if isinstance(s, GDNState):
                cache.layers[i] = s

    def run_ar():
        restore()
        return ar_step()

    def run_canvas(rows_=None, adv=n):
        restore()
        L = rt.canvas(cache, [11, 12, 13, 14][:n] + [mask] * (n - 1), n, rows=rows_)
        am = mx.argmax(L, axis=-1)
        mx.eval(am)
        rt.commit(cache, adv)
        return [l.ssm if isinstance(l, GDNState) else l.keys for l in cache.layers]

    t_ar = timeit(run_ar, a.iters)
    rt.fused_gdn = False
    t_cv_unfused = timeit(run_canvas, a.iters)
    rt.fused_gdn = True
    t_cv = timeit(run_canvas, a.iters)
    t_cold = timeit(lambda: run_canvas(rows_=n, adv=1), a.iters)
    t_cold_all = timeit(lambda: run_canvas(rows_=None, adv=1), a.iters)
    t_rej = timeit(lambda: run_canvas(adv=2), a.iters)

    # graph-build (host) time only
    def build_only():
        restore()
        t = time.perf_counter()
        L = rt.canvas(cache, [11, 12, 13, 14][:n] + [mask] * (n - 1), n)
        dt = time.perf_counter() - t
        mx.eval(L)
        cache.pending = None
        return dt
    for _ in range(3):
        build_only()
    t_build = statistics.median(build_only() for _ in range(a.iters)) * 1e3
    restore()

    # ---- components ----------------------------------------------------------
    def comp(L):
        x = mx.random.normal((1, L, D)).astype(mx.bfloat16)
        mx.eval(x)
        t_head = timeit(lambda: rt._lm_head(x), a.iters)
        t_mlp = timeit(lambda: [l.mlp(l.post_attention_layernorm(x)) for l in layers], a.iters)

        def attn():
            outs = []
            for i, l in enumerate(layers):
                if not l.is_linear:
                    kv = cache.layers[i]
                    m = rt._canvas_mask(kv.offset, L, n) if L > 1 else None
                    outs.append(rt._attention(l.self_attn, x, kv, m))
                    kv.trim(L)
            return outs
        t_attn = timeit(attn, a.iters)

        def gdn():
            return [rt._gdn(l.linear_attn, x, cache.layers[i], n if L > 1 else None)[0]
                    for i, l in enumerate(layers) if l.is_linear]
        t_gdn = timeit(gdn, a.iters)
        return t_head, t_mlp, t_attn, t_gdn

    c1 = comp(1)
    c7 = comp(blk)
    rt.fused_gdn = False
    c7u = comp(blk)
    rt.fused_gdn = True

    print(f"\n{'':34s} {'AR (1 tok)':>11s} {'canvas (%d)' % blk:>11s}   ratio")
    print(f"{'whole step (fwd+argmax+commit)':34s} {t_ar:9.2f}ms {t_cv:9.2f}ms   {t_cv / t_ar:4.2f}x")
    n_lin = sum(l.is_linear for l in layers)
    for name, x1, x7 in [("  LM head", c1[0], c7[0]), (f"  MLPs ({len(layers)})", c1[1], c7[1]),
                         (f"  full attention ({len(layers) - n_lin})", c1[2], c7[2]),
                         (f"  gated-delta ({n_lin})", c1[3], c7[3])]:
        print(f"{name:34s} {x1:9.2f}ms {x7:9.2f}ms   {x7 / x1:4.2f}x")
    s1, s7 = sum(c1), sum(c7)
    print(f"{'  sum of components':34s} {s1:9.2f}ms {s7:9.2f}ms   (rest = embed, norms, sync, commit)")
    print(f"\ncanvas variants:")
    print(f"  verify step, fused GDN            {t_cv:7.2f} ms")
    print(f"  verify step, unfused GDN          {t_cv_unfused:7.2f} ms   (GDN component {c7u[3]:.2f} ms vs fused {c7[3]:.2f} ms)")
    print(f"  cold step, LM head rows 0..{n - 1}      {t_cold:7.2f} ms")
    print(f"  cold step, LM head all rows       {t_cold_all:7.2f} ms")
    print(f"  verify + reject (commit adv=2)    {t_rej:7.2f} ms   (commit recompute cost {t_rej - t_cv:+.2f} ms)")
    print(f"  host graph build (canvas)         {t_build:7.2f} ms")
    print(f"\nbreak-even: self-spec beats AR when tokens/forward > {t_cv / t_ar:.2f}")
    print(f"peak memory {mx.get_peak_memory() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
