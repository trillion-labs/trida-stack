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

import collections

import mlx.core as mx
import numpy as np

from .model import SeqCache, TridaRuntime


# ----------------------------------------------------------------------------- sampling
@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    seed: Optional[int] = None
    # token ids that are never produced (e.g. <tool_call> for tool_choice "none")
    banned: tuple = ()
    # ((token id, min raw probability), ...): the token is only allowed where its probability under the raw
    # logits (temperature 1, no top-k/top-p) is at least that value. Temperature < 1 and top-k sharpen a
    # single-token option like <tool_call> against answers spread over many first tokens; the gate keeps
    # such low-confidence picks out (server --tool-call-gate).
    gate: tuple = ()
    # when a list: (raw probability, gate) of every position where a gated token was blocked although the
    # model gave it >= GATE_NOTE_MIN (exact AR rows only; for the server log)
    gate_log: Optional[list] = None

    @property
    def greedy(self) -> bool:
        return self.temperature <= 0


_BAN_BIAS: dict = {}
GATE_NOTE_MIN = 0.05


def _gate_note(sp: SamplingParams, rows: mx.array) -> None:
    """Records near-miss blocks of gated tokens in ``rows`` (exact AR logits, [V] or [r, V])."""
    if sp.gate_log is None or not sp.gate:
        return
    x = rows.astype(mx.float32)
    if x.ndim == 1:
        x = x[None]
    lse = mx.logsumexp(x, axis=-1)
    for tid, thr in sp.gate:
        if not 0 <= tid < x.shape[-1] or thr <= 0:
            continue
        for p in mx.exp(x[:, tid] - lse).tolist():
            if GATE_NOTE_MIN <= p < thr:
                sp.gate_log.append((round(p, 4), thr))


def apply_ban(x: mx.array, sp: SamplingParams) -> mx.array:
    """``x`` [..., V] with the banned tokens at -inf (greedy and sampled alike, drafts and verify
    alike, so self-spec stays exact w.r.t. the banned AR distribution)."""
    if not sp.banned and not sp.gate:
        return x
    V = x.shape[-1]
    x = x.astype(mx.float32)
    if sp.banned:
        key = (V, tuple(sorted(sp.banned)))
        bias = _BAN_BIAS.get(key)
        if bias is None:
            b = np.zeros(V, np.float32)
            b[[i for i in key[1] if 0 <= i < V]] = -np.inf
            bias = _BAN_BIAS[key] = mx.array(b)
        x = x + bias
    if sp.gate:
        lse = mx.logsumexp(x, axis=-1, keepdims=True)
        for tid, thr in sp.gate:
            if not 0 <= tid < V or thr <= 0:
                continue
            p = mx.exp(x[..., tid : tid + 1] - lse)          # raw probability, [..., 1]
            onehot = mx.arange(V) == tid                      # [V]
            x = mx.where(mx.logical_and(p < thr, onehot), -mx.inf, x)
    return x


def truncated_probs(logits: mx.array, sp: SamplingParams) -> mx.array:
    """Top-k / top-p truncated sampling distribution, float32 [..., V].

    Same distribution as the reference ``_sampling_probs`` (top-k, then top-p keeping the
    token that crosses ``top_p``), but with top-k set the nucleus is computed on the k
    survivors only, so there is no full-vocabulary sort (V = 248k)."""
    x = apply_ban(logits.astype(mx.float32), sp)
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
    gate_blocked: list = field(default_factory=list)  # raw probabilities of blocked gated tokens (>= 0.05)
    loop_stopped_thinking: bool = False  # the loop guard closed the thinking and the model answered after it
    prefill_s: float = 0.0
    decode_s: float = 0.0
    finish_reason: str = "length"
    stop_token: Optional[int] = None  # the EOS / stop id that ended the generation, if one did

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
            **({"gate_blocked": self.gate_blocked} if self.gate_blocked else {}),
            **({"loop_stopped_thinking": True} if self.loop_stopped_thinking else {}),
        }


LOOP_MIN_REPEATS = 6     # a block repeated this many times in a row ...
LOOP_MIN_SPAN = 96       # ... covering at least this many tokens ends the generation
LOOP_MAX_PERIOD = 256


def looping(hist: list) -> int:
    """Length of the repeating block if ``hist`` ends in one block of at most LOOP_MAX_PERIOD tokens
    repeated LOOP_MIN_REPEATS times back to back (spanning >= LOOP_MIN_SPAN tokens), else 0. Catches
    degenerate loops such as the same tool call emitted until max_tokens."""
    n = len(hist)
    for p in range(1, min(LOOP_MAX_PERIOD, n // LOOP_MIN_REPEATS) + 1):
        span = max(p * LOOP_MIN_REPEATS, LOOP_MIN_SPAN)
        if span > n:
            break
        tail = hist[n - span:]
        if tail[p:] == tail[:-p]:
            return p
    return 0


LOOP_NGRAM = 16          # a reply whose last LOOP_WINDOW tokens are at least LOOP_REPEAT_FRAC made of
LOOP_WINDOW = 1536       # 16-token phrases it already wrote inside that window is going in circles
LOOP_REPEAT_FRAC = 0.5   # (e.g. re-deriving the same steps of a hard problem with "wait, let me reconsider")


class NgramLoop:
    """Sliding-window detector for loops that are not exact repeats of one block: counts how many of the
    last LOOP_WINDOW positions end a LOOP_NGRAM-token phrase that already occurred in the window."""

    def __init__(self, n: int = LOOP_NGRAM, window: int = LOOP_WINDOW, frac: float = LOOP_REPEAT_FRAC):
        self.n, self.window, self.frac = n, window, frac
        self.counts: dict = {}
        self.win: collections.deque = collections.deque()
        self.rep = 0

    def add(self, hist: list) -> bool:
        """Feed the history after one new token; True when the window is full and mostly repeats."""
        if len(hist) < self.n:
            return False
        key = hash(tuple(hist[-self.n:]))
        seen = self.counts.get(key, 0) > 0
        self.counts[key] = self.counts.get(key, 0) + 1
        self.win.append((key, seen))
        self.rep += seen
        if len(self.win) > self.window:
            k, was = self.win.popleft()
            self.rep -= was
            c = self.counts[k] - 1
            if c:
                self.counts[k] = c
            else:
                del self.counts[k]
        return len(self.win) == self.window and self.rep >= self.frac * self.window


class _Emitter:
    """Applies max_new_tokens / EOS / the loop guard to emitted chunks and records stats."""

    loop_guard = False  # the server turns it on (see server.py --no-loop-guard); off for verify/bench/tests

    def __init__(self, stats: DecodeStats, max_new: int, eos: set):
        self.stats, self.max_new, self.eos, self.done = stats, max_new, eos, False
        self.hist: list = []
        self.ngram = NgramLoop()

    def __call__(self, toks: list) -> list:
        out = []
        for t in toks:
            if t in self.eos:
                self.done = True
                self.stats.finish_reason = "stop"
                self.stats.stop_token = t
                break
            out.append(t)
            self.hist.append(t)
            if self.stats.new_tokens + len(out) >= self.max_new:
                self.done = True
                self.stats.finish_reason = "length"
                break
            if self.loop_guard and ((len(self.hist) % 8 == 0 and looping(self.hist)) or self.ngram.add(self.hist)):
                self.done = True
                self.stats.finish_reason = "repetition"
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
        tok = mx.argmax(apply_ban(logits, sp)) if sp.greedy else sample_from_probs(truncated_probs(logits, sp))
        tok = int(tok.item())
        _gate_note(sp, logits)
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
        t0 = int(mx.argmax(apply_ban(logits, sp)).item())
    else:
        t0 = int(sample_from_probs(truncated_probs(logits, sp)).item())
    _gate_note(sp, logits)
    t0_emitted = False
    pending, specs, spec_q = None, None, None  # spec_q: [K, V] draft distributions

    while not emit.done:
        if specs is not None:  # ---- verify round
            canvas = [pending] + specs + [mask_id] * (n - 1)
            L = rt.canvas(cache, canvas, n)
            stats.forwards += 1
            if sp.greedy:
                am = mx.argmax(apply_ban(L, sp), axis=-1)
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
            _gate_note(sp, L[: acc + 1] if corr is not None else L[:n])  # rows that saw real context
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
                vals = mx.argmax(apply_ban(L[:n], sp), axis=-1).tolist()
                new_q = None
            else:
                P = truncated_probs(L[:n], sp)
                vals = sample_from_probs(P).tolist()
                new_q = P[1:n]
            clean, new_specs = vals[0], vals[1:n]
            _gate_note(sp, L[0])
            rt.commit(cache, 1)
            out = emit(([] if t0_emitted else [t0]) + [clean])
            t0_emitted = True
            pending, specs, spec_q = clean, new_specs, new_q
        if out:
            yield out
    stats.decode_s = time.perf_counter() - t_start
