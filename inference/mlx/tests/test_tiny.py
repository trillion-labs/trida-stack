"""Correctness tests on a tiny random Qwen3.5-hybrid model (CPU is fine; no weights needed).

    python -m pytest inference/mlx/tests -q      # or: python inference/mlx/tests/test_tiny.py

What is pinned:
  * chunked prefill == one-shot prefill
  * canvas clean rows (0..N-1) reproduce the AR logits exactly (the lossless property)
  * canvas MASK rows read the *block-end* GDN state (causal_mode=2 semantics)
  * commit(adv) leaves the cache identical to having AR-decoded those adv tokens
  * greedy self-spec output == greedy AR output, token for token, over long generations
  * a snapshot/restore round trip (prompt cache) reproduces a fresh prefill
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from trida_mlx.decode import DecodeStats, SamplingParams, ar_generate, selfspec_generate
from trida_mlx.model import GDNState, TridaRuntime, build_model
from mlx_lm.models.gated_delta import compute_g

MASK = 299
TINY = dict(
    model_type="qwen3_5_text", hidden_size=64, intermediate_size=128, num_hidden_layers=8,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6, vocab_size=300,
    linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=32, linear_value_head_dim=16,
    linear_conv_kernel_dim=4, tie_word_embeddings=True, full_attention_interval=4,
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25,
                     "mrope_section": [1, 1, 0]},
)


def tiny_runtime(seed=0, sharp=4.0):
    mx.random.seed(seed)
    model = build_model(TINY)
    # sharpen the output distribution so greedy decoding has structure (and fewer near-ties)
    emb = model.language_model.model.embed_tokens
    emb.weight = emb.weight * sharp
    mx.eval(model.parameters())
    return TridaRuntime(model, prefill_chunk=512)


def rand_prompt(n, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 290, size=n).tolist()


def close(a, b, tol=1e-4):
    return float(mx.max(mx.abs(a - b)).item()) < tol


def test_chunked_prefill():
    rt = tiny_runtime()
    p = rand_prompt(37)
    c1 = rt.make_cache(); l1 = rt.prefill(c1, p)
    rt2 = tiny_runtime(); rt2.prefill_chunk = 5
    c2 = rt2.make_cache(); l2 = rt2.prefill(c2, p)
    assert close(l1, l2)


def _ar_logits(rt, prompt, cont):
    c = rt.make_cache()
    out = [rt.prefill(c, prompt)]
    for t in cont:
        out.append(rt.ar_step(c, t))
    return out, c


def test_canvas_clean_rows_are_ar_logits():
    rt = tiny_runtime(); n = 4
    prompt = rand_prompt(20)
    cont = rand_prompt(n, seed=7)  # p, s0, s1, s2
    ar, _ = _ar_logits(rt, prompt, cont)
    c = rt.make_cache(); rt.prefill(c, prompt)
    L = rt.canvas(c, cont + [MASK] * (n - 1), n)
    for i in range(n):  # row i == AR logits after cont[:i+1]
        assert close(L[i], ar[i + 1]), f"row {i}"


def test_commit_matches_ar_state():
    n = 4
    for adv in range(1, n + 1):
        rt = tiny_runtime()
        prompt = rand_prompt(15)
        cont = rand_prompt(n, seed=3)
        c = rt.make_cache(); rt.prefill(c, prompt)
        rt.canvas(c, cont + [MASK] * (n - 1), n)
        rt.commit(c, adv)
        nxt = 42
        got = rt.ar_step(c, nxt)
        ar, _ = _ar_logits(rt, prompt, cont[:adv] + [nxt])
        assert close(got, ar[-1]), f"adv={adv}"
        assert c.tokens == prompt + cont[:adv] + [nxt]


def test_mask_rows_read_block_end_state():
    """Independent re-derivation of one GDN layer's canvas output with a per-token loop."""
    rt = tiny_runtime(); n = 4
    layer = next(l for l in rt.backbone.layers if l.is_linear)
    lin = layer.linear_attn
    x = mx.random.normal((1, 2 * n - 1, TINY["hidden_size"]))
    st = GDNState(conv=mx.random.normal((1, 3, lin.conv_dim)),
                  ssm=mx.random.normal((1, lin.num_v_heads, lin.head_v_dim, lin.head_k_dim)) * 0.1)
    out, _, _ = rt._gdn(lin, x, st, n)
    # manual: conv -> q,k,v -> step recurrence, record states
    qkv = lin.in_proj_qkv(x); z = lin.in_proj_z(x).reshape(1, 2 * n - 1, lin.num_v_heads, lin.head_v_dim)
    ci = mx.concatenate([st.conv, qkv], axis=1)
    co = nn.silu(lin.conv1d(ci))
    q, k, v = mx.split(co, [lin.key_dim, 2 * lin.key_dim], -1)
    T = 2 * n - 1
    q = q.reshape(1, T, lin.num_k_heads, -1); k = k.reshape(1, T, lin.num_k_heads, -1)
    v = v.reshape(1, T, lin.num_v_heads, -1)
    s = k.shape[-1] ** -0.5
    q = s * s * mx.fast.rms_norm(q, None, 1e-6); k = s * mx.fast.rms_norm(k, None, 1e-6)
    rep = lin.num_v_heads // lin.num_k_heads
    q = mx.repeat(q, rep, 2); k = mx.repeat(k, rep, 2)
    beta = mx.sigmoid(lin.in_proj_b(x)); g = compute_g(lin.A_log, lin.in_proj_a(x), lin.dt_bias)
    S = st.ssm; ys = []
    for t in range(T):
        S = S * g[:, t][..., None, None]
        kv = (S * k[:, t][:, :, None, :]).sum(-1)
        S = S + k[:, t][:, :, None, :] * ((v[:, t] - kv) * beta[:, t][..., None])[..., None]
        ys.append((S * q[:, t][:, :, None, :]).sum(-1))
    S_end = S
    for t in range(n, T):
        ys[t] = (S_end * q[:, t][:, :, None, :]).sum(-1)
    y = mx.stack(ys, axis=1)
    ref = lin.out_proj(lin.norm(y, z).reshape(1, T, -1))
    assert close(out, ref, 1e-4)


def test_fused_gdn_matches_unfused():
    """Metal-only: the fused canvas GDN kernel == two mlx-lm kernel calls + block-end readout."""
    from trida_mlx.kernels import fused_available
    if not fused_available():
        print("    (skipped: no Metal)")
        return
    rt = tiny_runtime(); n = 4
    lin = next(l for l in rt.backbone.layers if l.is_linear).linear_attn
    x = mx.random.normal((1, 2 * n - 1, TINY["hidden_size"]))
    st = GDNState(conv=mx.random.normal((1, 3, lin.conv_dim)),
                  ssm=mx.random.normal((1, lin.num_v_heads, lin.head_v_dim, lin.head_k_dim)) * 0.1)
    rt.fused_gdn = True
    o1, _, r1 = rt._gdn(lin, x, st, n)
    rt.fused_gdn = False
    o2, _, r2 = rt._gdn(lin, x, st, n)
    assert close(o1, o2, 1e-5) and close(r1.s_clean, r2.s_clean, 1e-6)
    # clean rows must be bit-identical to the unfused (= AR) kernel
    assert bool(mx.array_equal(o1[:, :n], o2[:, :n]).item())


def _collect(gen):
    out = []
    for chunk in gen:
        out.extend(chunk)
    return out


def test_greedy_selfspec_is_lossless():
    sp = SamplingParams(temperature=0.0)
    for seed in range(3):
        rt = tiny_runtime(seed=seed)
        prompt = rand_prompt(25, seed=seed + 10)
        ar = _collect(ar_generate(rt, rt.make_cache(), prompt, max_new_tokens=120, eos_ids=set(), sp=sp))
        from trida_mlx.decode import DecodeStats
        st = DecodeStats()
        ss = _collect(selfspec_generate(rt, rt.make_cache(), prompt, max_new_tokens=120, eos_ids=set(),
                                        sp=sp, mask_id=MASK, n=4, stats=st))
        assert ss == ar, f"seed {seed}: first diff at {next(i for i,(a,b) in enumerate(zip(ss,ar)) if a!=b)}"
        assert st.forwards <= len(ar)


def test_greedy_lossless_with_oracle_drafts():
    """Random models rarely accept drafts, so also drive the accept path: patch the canvas
    so MASK rows 'predict' the true AR continuation and check output + acceptance."""
    sp = SamplingParams(temperature=0.0)
    rt = tiny_runtime(seed=5)
    prompt = rand_prompt(18, seed=4)
    ar = _collect(ar_generate(rt, rt.make_cache(), prompt, max_new_tokens=60, eos_ids=set(), sp=sp))
    full = prompt + ar
    orig = rt.canvas

    # replace the MASK draft rows with a one-hot on the true AR continuation
    def patched(cache, toks, n, rows=None):
        base = cache.length
        L = orig(cache, toks, n, rows=rows)
        rows = []
        V = L.shape[-1]
        for i in range(L.shape[0]):
            want = base + i + 1
            if i >= 1 and want < len(full) and (toks[i] == MASK):
                rows.append(mx.where(mx.arange(V) == full[want], 100.0, 0.0))
            else:
                rows.append(L[i])
        return mx.stack(rows)

    rt.canvas = patched
    from trida_mlx.decode import DecodeStats
    st = DecodeStats()
    ss = _collect(selfspec_generate(rt, rt.make_cache(), prompt, max_new_tokens=60, eos_ids=set(),
                                    sp=sp, mask_id=MASK, n=4, stats=st))
    assert ss == ar
    assert st.accept_hist[-1] > 0 and st.tokens_per_forward > 2.5, st.as_dict()


def test_snapshot_restore():
    rt = tiny_runtime()
    p = rand_prompt(30)
    c = rt.make_cache(); rt.prefill(c, p[:20])
    snap = c.snapshot()
    rt.prefill(c, p[20:])
    for t in rand_prompt(9, seed=9):
        rt.ar_step(c, t)
    c.restore(snap)
    l1 = rt.prefill(c, p[20:])
    l2 = rt.prefill(rt.make_cache(), p)
    assert close(l1, l2)


def test_sampling_runs():
    sp = SamplingParams(temperature=1.0, top_k=20, top_p=0.9, seed=0)
    rt = tiny_runtime()
    out = _collect(selfspec_generate(rt, rt.make_cache(), rand_prompt(12), max_new_tokens=40,
                                     eos_ids=set(), sp=sp, mask_id=MASK, n=4))
    assert len(out) == 40


def test_truncated_probs_matches_reference():
    from trida_mlx.decode import truncated_probs

    def ref(x, t, k, p):  # numpy port of the SGLang reference _sampling_probs
        x = x / t
        kth = np.sort(x, axis=-1)[:, -k][:, None]
        x = np.where(x < kth, -np.inf, x)
        order = np.argsort(-x, axis=-1, kind="stable")
        sx = np.take_along_axis(x, order, -1)
        e = np.exp(sx - sx.max(-1, keepdims=True)); pr = e / e.sum(-1, keepdims=True)
        sx = np.where(np.cumsum(pr, -1) - pr >= p, -np.inf, sx)
        out = np.full_like(x, -np.inf); np.put_along_axis(out, order, sx, -1)
        e = np.exp(out - out.max(-1, keepdims=True)); return e / e.sum(-1, keepdims=True)

    rng = np.random.default_rng(0)
    x = rng.normal(size=(7, 1000)).astype(np.float32) * 3
    for t, k, p in [(1.0, 50, 0.95), (0.7, 20, 0.8), (1.0, 5, 1.0)]:
        got = np.array(truncated_probs(mx.array(x), SamplingParams(temperature=t, top_k=k, top_p=p)))
        assert np.allclose(got, ref(x, t, k, p), atol=1e-5), (t, k, p)


def test_speculative_sampling_preserves_distribution():
    """Sampled self-spec must reproduce the AR sampling distribution. Random models almost
    never accept drafts, so the MASK rows are patched to reuse a neighbouring row's logits:
    any draft q is valid for speculative sampling, and this one gets accepted often."""
    from collections import Counter
    rt = tiny_runtime(seed=2, sharp=1.0)
    prompt = rand_prompt(10, seed=2)
    sp = SamplingParams(temperature=1.0, top_k=4, top_p=0.9)
    orig = rt.canvas

    def patched(cache, toks, n, rows=None):
        L = orig(cache, toks, n, rows=rows)
        return mx.stack([L[i] if (i == 0 or toks[i] != MASK) else L[i - 1] for i in range(L.shape[0])])

    def hist(fn, reps, seed0, **kw):
        c, st_all = Counter(), DecodeStats()
        st_all.accept_hist = [0] * 4
        for i in range(reps):
            mx.random.seed(seed0 + i)
            st = DecodeStats()
            out = _collect(fn(rt, rt.make_cache(), prompt, max_new_tokens=4, eos_ids=set(), sp=sp, stats=st, **kw))
            c[out[3]] += 1
            if st.accept_hist:
                st_all.accept_hist = [x + y for x, y in zip(st_all.accept_hist, st.accept_hist)]
        return c, st_all

    R = 600
    a, _ = hist(ar_generate, R, 5000)
    a2, _ = hist(ar_generate, R, 90000)
    rt.canvas = patched
    b, st = hist(selfspec_generate, R, 7000, mask_id=MASK, n=4)
    tv = lambda x, y: 0.5 * sum(abs(x[k] - y[k]) for k in set(x) | set(y)) / R
    assert sum(st.accept_hist[1:]) > R // 4, st.accept_hist  # the accept path really ran
    assert tv(a, b) < max(0.08, 2.0 * tv(a, a2)), (tv(a, b), tv(a, a2))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            try:
                fn(); print(f"  ok   {name}")
            except Exception as e:  # noqa: BLE001
                fails += 1; print(f"  FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if fails else 0)
