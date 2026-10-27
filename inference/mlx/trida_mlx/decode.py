"""Decoders: autoregressive baseline and self-speculative (AR-Trust) decoding.

Self-spec follows the SGLang reference ``HybridDiffusionSelfSpec`` (variant
``bd_bidir_shift``, ``block_size = 2N-1``, ``gen_block_size = N``):

    cold start : [t0, MASK x (2N-2)]                 -> clean = row 0, specs = rows 1..N-1
    verify     : [pending, spec_0..spec_{N-2}, MASK x (N-1)]
                 spec_i is verified against row i (exact AR logits);
                 all accepted -> clean = row N-1, new specs = rows N..2N-2
                 reject at i  -> emit spec_0..spec_{i-1} + correction; next step is a
                                 cold start from the correction token.

Greedy (temperature 0) is lossless: the output equals ``ar_generate`` token for token.
With temperature > 0 the ``strict_truncated`` policy is used: drafts are sampled from the
same top-k/top-p truncated distribution the verifier uses, accepted with prob
``min(1, p/q)`` and corrected from ``norm(max(0, p - q))`` — standard speculative
sampling, so the output *distribution* matches AR sampling.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator, Optional

import mlx.core as mx

from .model import SeqCache, TridaRuntime


# ----------------------------------------------------------------------------- sampling
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    seed: Optional[int] = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0


def truncated_probs(logits: mx.array, sp: SamplingParams) -> mx.array:
    """Top-k / top-p truncated sampling distribution, float32 [..., V].

    Same distribution as the reference ``_sampling_probs`` (top-k, then top-p keeping the
    token that crosses ``top_p``), but with top-k set the nucleus is computed on the k
    survivors only, so there is no full-vocabulary sort (V = 248k)."""
    x = logits.astype(mx.float32)
    if sp.temperature != 1.0:
        x = x / sp.temperature
    V = x.shape[-1]
    if 0 < sp.top_k < V:
        top = mx.sort(mx.topk(x, sp.top_k, axis=-1), axis=-1)[..., ::-1]  # [.., k] descending
        kth = top[..., -1:]
        x = mx.where(x < kth, -mx.inf, x)
        if sp.top_p < 1.0:
            ps = mx.softmax(top, axis=-1)
            cum = mx.cumsum(ps, axis=-1)
            keep = (cum - ps) < sp.top_p  # a prefix of the sorted survivors
            n_keep = mx.maximum(keep.sum(axis=-1, keepdims=True), 1)
            thresh = mx.take_along_axis(top, n_keep - 1, axis=-1)
            x = mx.where(x < thresh, -mx.inf, x)
    elif sp.top_p < 1.0:
        order = mx.argsort(-x, axis=-1)
        sx = mx.take_along_axis(x, order, axis=-1)
        sp_ = mx.softmax(sx, axis=-1)
        cum = mx.cumsum(sp_, axis=-1)
        sx = mx.where((cum - sp_) >= sp.top_p, -mx.inf, sx)
        x = mx.take_along_axis(sx, mx.argsort(order, axis=-1), axis=-1)
    return mx.softmax(x, axis=-1)


def sample_from_probs(probs: mx.array) -> mx.array:
    return mx.random.categorical(mx.log(probs), axis=-1)


@dataclass
class DecodeStats:
    mode: str = ""
    prompt_tokens: int = 0
    reused_tokens: int = 0
    new_tokens: int = 0
    forwards: int = 0
    cold_forwards: int = 0
    accept_hist: list = field(default_factory=list)
    prefill_s: float = 0.0
    decode_s: float = 0.0
    finish_reason: str = "length"

    @property
    def tokens_per_forward(self) -> float:
        return self.new_tokens / self.forwards if self.forwards else 0.0

    @property
    def decode_tps(self) -> float:
        return self.new_tokens / self.decode_s if self.decode_s else 0.0

    def as_dict(self) -> dict:
        return {
            "mode": self.mode, "prompt_tokens": self.prompt_tokens, "reused_tokens": self.reused_tokens,
            "new_tokens": self.new_tokens, "forwards": self.forwards, "cold_forwards": self.cold_forwards,
            "tokens_per_forward": round(self.tokens_per_forward, 3), "accept_hist": self.accept_hist,
            "prefill_s": round(self.prefill_s, 4), "decode_s": round(self.decode_s, 4),
            "decode_tok_s": round(self.decode_tps, 2), "finish_reason": self.finish_reason,
        }


class _Emitter:
    """Applies max_new_tokens / EOS to emitted chunks and records stats."""

    def __init__(self, stats: DecodeStats, max_new: int, eos: set):
        self.stats, self.max_new, self.eos, self.done = stats, max_new, eos, False

    def __call__(self, toks: list) -> list:
        out = []
        for t in toks:
            if t in self.eos:
                self.done = True
                self.stats.finish_reason = "stop"
                break
            out.append(t)
            if self.stats.new_tokens + len(out) >= self.max_new:
                self.done = True
                self.stats.finish_reason = "length"
                break
        self.stats.new_tokens += len(out)
        return out


def _prefill(rt: TridaRuntime, cache: SeqCache, prompt: list, stats: DecodeStats):
    t = time.perf_counter()
    todo = prompt[cache.length:]
    stats.prompt_tokens, stats.reused_tokens = len(prompt), cache.length
    if not todo:  # whole prompt cached: re-feed the last token to get its logits
        raise ValueError("prompt fully cached; callers must leave at least one token to prefill")
    logits = rt.prefill(cache, todo)
    mx.eval(logits)
    stats.prefill_s = time.perf_counter() - t
    return logits


# ----------------------------------------------------------------------------- AR
def ar_generate(rt: TridaRuntime, cache: SeqCache, prompt: list, *, max_new_tokens: int,
                eos_ids: set, sp: SamplingParams, stats: Optional[DecodeStats] = None) -> Iterator[list]:
    stats = stats or DecodeStats()
    stats.mode = "causal"
    if sp.seed is not None:
        mx.random.seed(sp.seed)
    logits = _prefill(rt, cache, prompt, stats)
    emit = _Emitter(stats, max_new_tokens, eos_ids)
    t = time.perf_counter()
    while True:
        tok = mx.argmax(logits) if sp.greedy else sample_from_probs(truncated_probs(logits, sp))
        tok = int(tok.item())
        out = emit([tok])
        if out:
            yield out
        if emit.done:
            break
        logits = rt.ar_step(cache, tok)
        stats.forwards += 1
    stats.decode_s = time.perf_counter() - t


# ----------------------------------------------------------------------------- self-spec
def selfspec_generate(rt: TridaRuntime, cache: SeqCache, prompt: list, *, max_new_tokens: int,
                      eos_ids: set, sp: SamplingParams, mask_id: int, n: int = 4,
                      stats: Optional[DecodeStats] = None) -> Iterator[list]:
    stats = stats or DecodeStats()
    stats.mode = f"self-spec(N={n})"
    stats.accept_hist = [0] * n  # accepted specs per verify round: 0..N-1
    if sp.seed is not None:
        mx.random.seed(sp.seed)
    blk, K = 2 * n - 1, n - 1
    logits = _prefill(rt, cache, prompt, stats)
    emit = _Emitter(stats, max_new_tokens, eos_ids)
    t_start = time.perf_counter()

    if sp.greedy:
        t0 = int(mx.argmax(logits).item())
    else:
        t0 = int(sample_from_probs(truncated_probs(logits, sp)).item())
    t0_emitted = False
    pending, specs, spec_q = None, None, None  # spec_q: [K, V] draft distributions

    while not emit.done:
        if specs is not None:  # ---- verify round
            canvas = [pending] + specs + [mask_id] * (n - 1)
            L = rt.canvas(cache, canvas, n)
            stats.forwards += 1
            if sp.greedy:
                am = mx.argmax(L, axis=-1)
                vals = am.tolist()
                acc = 0
                while acc < K and vals[acc] == specs[acc]:
                    acc += 1
                corr = vals[acc] if acc < K else None
                clean, new_specs = vals[n - 1], vals[n:blk]
                new_q = None
            else:
                P = truncated_probs(L, sp)  # [blk, V]
                sv = mx.array(specs, dtype=mx.int32)
                p = mx.take_along_axis(P[:K], sv[:, None], axis=-1)[:, 0]
                q = mx.take_along_axis(spec_q, sv[:, None], axis=-1)[:, 0]
                ratio = mx.where(q > 0, p / mx.maximum(q, 1e-30), 0.0)
                u = mx.random.uniform(shape=(K,))
                accepted = (ratio >= 1.0) | (u < ratio)
                resid = mx.maximum(P[:K] - spec_q, 0.0)
                rs = resid.sum(axis=-1, keepdims=True)
                resid = mx.where(rs > 0, resid / mx.maximum(rs, 1e-30), P[:K])
                corr_all = sample_from_probs(resid)
                samp = sample_from_probs(P[n - 1 : blk])  # clean + new specs
                packed = mx.concatenate([accepted.astype(mx.int32), corr_all.astype(mx.int32),
                                         samp.astype(mx.int32)])
                vals = packed.tolist()
                acc_l, corr_l, samp_l = vals[:K], vals[K : 2 * K], vals[2 * K :]
                acc = 0
                while acc < K and acc_l[acc]:
                    acc += 1
                corr = corr_l[acc] if acc < K else None
                clean, new_specs = samp_l[0], samp_l[1:]
                new_q = P[n:blk]
            if corr is not None:  # rejected at spec `acc`
                rt.commit(cache, 1 + acc)
                stats.accept_hist[acc] += 1
                out = emit(specs[:acc] + [corr])
                t0, t0_emitted = corr, True
                pending, specs, spec_q = None, None, None
            else:  # all K accepted: N tokens this forward
                rt.commit(cache, n)
                stats.accept_hist[K] += 1
                out = emit(specs + [clean])
                pending, specs, spec_q = clean, new_specs, new_q
        else:  # ---- cold start
            canvas = [t0] + [mask_id] * (blk - 1)
            L = rt.canvas(cache, canvas, n, rows=n if rt.skip_rows else None)
            stats.forwards += 1
            stats.cold_forwards += 1
            if sp.greedy:
                vals = mx.argmax(L[:n], axis=-1).tolist()
                new_q = None
            else:
                P = truncated_probs(L[:n], sp)
                vals = sample_from_probs(P).tolist()
                new_q = P[1:n]
            clean, new_specs = vals[0], vals[1:n]
            rt.commit(cache, 1)
            out = emit(([] if t0_emitted else [t0]) + [clean])
            t0_emitted = True
            pending, specs, spec_q = clean, new_specs, new_q
        if out:
            yield out
    stats.decode_s = time.perf_counter() - t_start
