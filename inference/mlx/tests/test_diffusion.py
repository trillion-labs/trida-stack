"""Block diffusion decoding (``diffusion_generate``) on the tiny random Qwen3.5-hybrid model.

What is pinned:
  * ``discard`` leaves the cache exactly as it was before the canvas
  * B = 2 is exact AR (its one masked position is read from the clean seed row)
  * B = 3: the first two tokens are exact AR picks, and the cache after decoding equals a fresh prefill
  * the threshold sets the number of denoise rounds (0 -> one round per block, > 1 -> B-1 rounds)
  * EOS and max_new_tokens end decoding, MASK is never emitted, sampling is seeded
  * the sampler matches the reference ``LowConfidenceShiftHybridDiffusion._sample_tokens``: its
    distribution (top-p drops the token that crosses top_p) and its confidence (plain softmax)
  * the server samples diffusion with the reference top-k unless the request sets one
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from test_tiny import MASK, _collect, close, rand_prompt, tiny_runtime  # noqa: E402
from trida_mlx.decode import (DIFFUSION_TOP_K, DecodeStats, SamplingParams, ar_generate,  # noqa: E402
                              denoise_pick, diffusion_generate, diffusion_probs, truncated_probs)

GREEDY = SamplingParams(temperature=0.0)


def _diffusion(rt, prompt, n, *, block=3, threshold=0.95, sp=GREEDY, eos=(), stats=None, cache=None):
    cache = cache if cache is not None else rt.make_cache()
    return _collect(diffusion_generate(rt, cache, prompt, max_new_tokens=n, eos_ids=set(eos), sp=sp,
                                       mask_id=MASK, block=block, threshold=threshold, stats=stats))


def test_discard_restores_cache():
    rt = tiny_runtime()
    p = rand_prompt(21)
    a, b = rt.make_cache(), rt.make_cache()
    rt.prefill(a, p)
    rt.prefill(b, p)
    for canvas in ([7, MASK, MASK], [7, 12, MASK], [3, MASK, MASK, MASK, MASK]):
        rt.canvas(b, canvas, 1)
        rt.discard(b)
    assert b.tokens == a.tokens and b.pos == a.pos and b.pending is None
    # the state reads back the same: the next AR step and a later canvas give the same logits
    assert close(rt.ar_step(a, 5), rt.ar_step(b, 5))
    assert close(rt.canvas(a, [9, MASK, MASK], 1), rt.canvas(b, [9, MASK, MASK], 1))


def test_block_two_is_exact_ar():
    for seed in range(3):
        rt = tiny_runtime(seed=seed)
        prompt = rand_prompt(25, seed=seed + 10)
        ar = _collect(ar_generate(rt, rt.make_cache(), prompt, max_new_tokens=80, eos_ids=set(), sp=GREEDY))
        st = DecodeStats()
        df = _diffusion(rt, prompt, 80, block=2, stats=st)
        assert df == ar, f"seed {seed}: first diff at {next(i for i, (x, y) in enumerate(zip(df, ar)) if x != y)}"
        assert st.rounds_hist == [0, 40]  # 40 blocks of 2, one round each


def test_block_three_head_is_ar_and_cache_is_consistent():
    rt = tiny_runtime(seed=1)
    prompt = rand_prompt(19, seed=3)
    ar = _collect(ar_generate(rt, rt.make_cache(), prompt, max_new_tokens=2, eos_ids=set(), sp=GREEDY))
    cache = rt.make_cache()
    out = _diffusion(rt, prompt, 31, cache=cache)  # 31 = 10 full blocks + 1: the last block is cut short
    assert len(out) == 31 and out[:2] == ar
    assert MASK not in out
    # every committed block is in the cache, as if the tokens had been prefilled; the cut-off last
    # block was never committed
    assert cache.tokens == prompt + out[:30]
    fresh = rt.make_cache()
    rt.prefill(fresh, prompt + out[:30])
    assert close(rt.ar_step(cache, 4), rt.ar_step(fresh, 4))


def test_threshold_sets_rounds():
    rt = tiny_runtime(seed=2)
    prompt = rand_prompt(15, seed=6)
    eager, strict = DecodeStats(), DecodeStats()
    _diffusion(rt, prompt, 40, block=5, threshold=0.0, stats=eager)
    _diffusion(rt, prompt, 40, block=5, threshold=1.01, stats=strict)
    assert eager.rounds_hist == [0, 8, 0, 0, 0]
    assert strict.rounds_hist == [0, 0, 0, 0, 8]
    # forwards: one per round plus a commit per block except the last
    assert eager.forwards == 8 + 7 and strict.forwards == 8 * 4 + 7


def test_eos_and_length_stop():
    rt = tiny_runtime(seed=1)
    prompt = rand_prompt(19, seed=3)
    head = _diffusion(rt, prompt, 9)
    st = DecodeStats()
    out = _diffusion(rt, prompt, 50, eos=[head[4]], stats=st)
    assert out == head[: head.index(head[4])] and st.finish_reason == "stop"
    st = DecodeStats()
    assert len(_diffusion(rt, prompt, 7, stats=st)) == 7 and st.finish_reason == "length"


def test_sampling_is_seeded():
    sp = SamplingParams(temperature=1.0, top_k=20, top_p=0.9, seed=0)
    rt = tiny_runtime()
    prompt = rand_prompt(12)
    a = _diffusion(rt, prompt, 40, sp=sp)
    b = _diffusion(rt, prompt, 40, sp=sp)
    assert len(a) == 40 and a == b and MASK not in a


def test_mask_is_never_predicted():
    """Make MASK the model's top token everywhere: diffusion must still fill every position."""
    rt = tiny_runtime(seed=3)
    head = rt._lm_head

    def mask_first(h):
        logits = head(h)
        return logits + (mx.arange(logits.shape[-1]) == MASK) * 1e4

    rt._lm_head = mask_first
    out = _diffusion(rt, rand_prompt(10, seed=2), 24)
    assert len(out) == 24 and MASK not in out


def _reference_probs(logits, temperature, top_k, top_p):
    """NumPy port of the reference sampler's distribution (probs_final in _sample_tokens)."""
    x = logits.astype(np.float64) / temperature
    if top_k > 0:
        kth = np.sort(x, axis=-1)[..., -min(top_k, x.shape[-1])][..., None]
        x = np.where(x >= kth, x, -np.inf)
    if top_p < 1.0:
        order = np.argsort(-x, axis=-1)
        sv = np.take_along_axis(x, order, axis=-1)
        sp_ = np.exp(sv - sv[..., :1])
        sp_ /= sp_.sum(-1, keepdims=True)
        keep = np.cumsum(sp_, axis=-1) <= top_p
        keep[..., 0] = True
        sv = np.where(keep, sv, -np.inf)
        x = np.full_like(x, -np.inf)
        np.put_along_axis(x, order, sv, axis=-1)
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


def test_diffusion_probs_match_reference():
    rng = np.random.default_rng(0)
    logits = (rng.standard_normal((6, 300)) * 3).astype(np.float32)
    for t, k, p in [(1.0, 20, 0.95), (0.7, 50, 0.9), (1.3, 0, 0.8), (1.0, 20, 1.0), (1.0, 0, 1.0)]:
        got = np.array(diffusion_probs(mx.array(logits), SamplingParams(temperature=t, top_k=k, top_p=p)))
        assert np.allclose(got, _reference_probs(logits, t, k, p), atol=1e-5), (t, k, p)


def test_top_p_boundary_differs_from_self_spec():
    """probabilities 0.5 / 0.3 / 0.15 / 0.05 at top_p 0.9: the diffusion reference keeps 2 tokens (0.95 crosses
    0.9), the self-spec reference keeps the crossing token too (3)."""
    logits = mx.log(mx.array([[0.5, 0.3, 0.15, 0.05]]))
    sp = SamplingParams(temperature=1.0, top_k=0, top_p=0.9)
    assert int((diffusion_probs(logits, sp) > 0).sum().item()) == 2
    assert int((truncated_probs(logits, sp) > 0).sum().item()) == 3


def test_confidence_is_plain_softmax():
    """A banned top token is never picked, and the runner-up's confidence is its own probability, not
    renormalized over the tokens left after the ban."""
    logits = mx.log(mx.array([[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]]))
    pred, conf = denoise_pick(logits, SamplingParams(temperature=0.0, banned=(0,)))
    assert pred == [1, 2]
    assert np.allclose(conf, [0.3, 0.7], atol=1e-6)
    pred, conf = denoise_pick(logits, SamplingParams(temperature=1.0, top_k=2, top_p=1.0, banned=(0,), seed=0))
    assert pred[0] in (1, 2) and np.isclose(conf[0], [0.6, 0.3, 0.1][pred[0]], atol=1e-6)


def test_server_diffusion_top_k_default():
    from trida_mlx import server

    saved = dict(server.DEFAULTS)
    try:
        server.DEFAULTS.update(temperature=1.0, top_k=50, top_p=0.95, mode="self-spec")
        assert server._sampling({}).top_k == 50
        assert server._sampling({"trida_mode": "diffusion"}).top_k == DIFFUSION_TOP_K
        assert server._sampling({"trida_mode": "diffusion", "top_k": 7}).top_k == 7
        server.DEFAULTS.update(mode="diffusion")
        assert server._sampling({}).top_k == DIFFUSION_TOP_K
        assert server._sampling({"trida_mode": "causal"}).top_k == 50
    finally:
        server.DEFAULTS.clear()
        server.DEFAULTS.update(saved)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
