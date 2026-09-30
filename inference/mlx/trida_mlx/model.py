"""Trida2.0-4B (Qwen3.5 hybrid: gated-delta linear attention + full attention) on MLX.

This module owns the *forward passes* the self-speculative decoder needs, built on the
weights/modules of ``mlx_lm.models.qwen3_5`` (so quantized checkpoints, the Metal
gated-delta kernel and RoPE all come from mlx-lm unchanged):

* ``prefill``      causal prompt prefill (chunked), persists KV + GDN state.
* ``ar_step``      one causal token (the AR baseline, ``--mode causal``).
* ``canvas``       one self-spec forward over a ``2N-1`` token canvas with the
                   ``bd_bidir_shift`` semantics of the SGLang reference
                   (``HybridDiffusionSelfSpec``):

    - full attention: rows ``0..N-1`` are token-causal, rows ``N..2N-2`` (the MASK
      slots) attend to the whole canvas; the prefix is visible to every row.
    - gated-delta (``causal_mode=2, num_clean=N``): rows ``0..N-1`` read the recurrent
      state *at their own position*; MASK rows read the state at the *block end*.
    - nothing is persisted by the forward: ``commit(adv)`` later keeps the first
      ``adv`` canvas tokens (KV trim + GDN state/conv window of the accepted prefix).

Because rows ``0..N-1`` are exactly the causal computation, their logits are the AR
logits, which is what makes greedy self-spec lossless w.r.t. ``ar_step``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3_5
from mlx_lm.models.cache import KVCache
from mlx_lm.models.gated_delta import compute_g, gated_delta_kernel, gated_delta_ops

from .kernels import canvas_gdn_fused, fused_available

DEFAULT_MODEL = "trillionlabs/Trida2.0-4B"
DEFAULT_MASK_ID = 248077


# ----------------------------------------------------------------------------- loading
def resolve_model_path(path_or_id: str) -> Path:
    p = Path(path_or_id).expanduser()
    if p.is_dir():
        return p
    if path_or_id.startswith(("/", "~", ".")) or path_or_id.count("/") > 1:
        raise FileNotFoundError(f"local model directory not found: {p}")
    from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            path_or_id,
            allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.tiktoken"],
        )
    )


def _text_config(cfg: dict) -> dict:
    text = dict(cfg.get("text_config", cfg))
    text.pop("architectures", None)
    return text


def build_model(text_cfg: dict) -> qwen3_5.Model:
    args = qwen3_5.ModelArgs(model_type="qwen3_5", text_config=dict(text_cfg))
    return qwen3_5.Model(args)


def load_model(path_or_id: str = DEFAULT_MODEL, *, dtype=mx.bfloat16):
    """Load a raw HF Trida checkpoint or one written by ``convert.py`` (optionally quantized).

    Returns ``(model, config_dict, path)``.
    """
    path = resolve_model_path(path_or_id)
    cfg = json.loads((path / "config.json").read_text())
    model = build_model(_text_config(cfg))
    weights = {}
    for f in sorted(path.glob("*.safetensors")):
        weights.update(mx.load(str(f)))
    if not weights:
        raise FileNotFoundError(f"no *.safetensors in {path}")
    quant = cfg.get("quantization")
    if quant:
        nn.quantize(
            model,
            group_size=quant["group_size"],
            bits=quant["bits"],
            class_predicate=lambda p, m: f"{p}.scales" in weights,
        )
    weights = model.sanitize(weights)
    if model.language_model.args.tie_word_embeddings:
        # mlx-lm's qwen3_5 sanitize pops "lm_head.weight" before it is prefixed, so a raw
        # text-only checkpoint that also ships the (tied) head leaves it behind. With tied
        # embeddings HF/vLLM ignore it and use embed_tokens; do the same.
        head = weights.pop("language_model.lm_head.weight", None)
        for sfx in ("scales", "biases"):
            weights.pop(f"language_model.lm_head.{sfx}", None)
        emb = weights.get("language_model.model.embed_tokens.weight")
        if head is not None and emb is not None and head.shape == emb.shape and not quant:
            if not mx.array_equal(head.astype(mx.float32), emb.astype(mx.float32)).item():
                print("[trida-mlx] warning: config ties embeddings but lm_head.weight differs from "
                      "embed_tokens; using embed_tokens (same as HF/vLLM)")
    if not quant:
        weights = {
            k: (v if k.endswith("A_log") or not mx.issubdtype(v.dtype, mx.floating) else v.astype(dtype))
            for k, v in weights.items()
        }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return model, cfg, path


def mask_id_of(path: Path, cfg: dict, tokenizer=None) -> int:
    bd = path / "block_diffusion.json"
    if bd.is_file():
        return int(json.loads(bd.read_text())["mask_id"])
    if "mask_id" in cfg:
        return int(cfg["mask_id"])
    if tokenizer is not None:
        tid = tokenizer.convert_tokens_to_ids("<|mask|>")
        if isinstance(tid, int) and tid >= 0 and tid != tokenizer.unk_token_id:
            return tid
    return DEFAULT_MASK_ID


# ----------------------------------------------------------------------------- cache
@dataclass
class GDNState:
    conv: mx.array  # [1, K-1, conv_dim] last pre-conv rows
    ssm: mx.array  # [1, Hv, Dv, Dk] float32 recurrent state


@dataclass
class _CanvasGDN:
    """What a canvas forward leaves behind per GDN layer so ``commit`` can persist
    the state after any accepted prefix without re-running the model."""

    prev: GDNState
    conv_input: mx.array  # [1, K-1+T, conv_dim] = prev conv rows ++ canvas pre-conv rows
    q: mx.array
    k: mx.array
    v: mx.array
    g: mx.array
    beta: mx.array  # clean rows only (first n_clean)
    s_clean: mx.array  # state after all n_clean clean rows


@dataclass
class SeqCache:
    """Per-sequence decode state: KV caches (attention layers), GDN states (linear
    layers) and the list of committed token ids (what the state encodes)."""

    layers: list
    tokens: list = field(default_factory=list)
    pending: Optional[list] = None  # _CanvasGDN per layer after a canvas forward
    canvas_len: int = 0

    @property
    def length(self) -> int:
        return len(self.tokens)

    # A snapshot is cheap: MLX arrays are immutable, so GDN states are kept by
    # reference; attention KV only needs its length (it is append-only).
    def snapshot(self):
        assert self.pending is None
        return (
            list(self.tokens),
            [l if isinstance(l, GDNState) else l.offset for l in self.layers],
        )

    def restore(self, snap) -> None:
        tokens, states = snap
        assert self.pending is None
        for i, s in enumerate(states):
            if isinstance(s, GDNState):
                self.layers[i] = s
            else:
                kv = self.layers[i]
                if s > kv.offset:
                    raise ValueError("cannot restore a KV snapshot longer than the cache")
                kv.trim(kv.offset - s)
        self.tokens = list(tokens)

    def nbytes(self) -> int:
        n = 0
        for l in self.layers:
            if isinstance(l, GDNState):
                n += l.conv.nbytes + l.ssm.nbytes
            else:
                n += l.nbytes
        return n


# ----------------------------------------------------------------------------- runtime
def _gdr(q, k, v, g, beta, state):
    """Gated-delta recurrence: Metal kernel on GPU, reference ops elsewhere."""
    if mx.default_device() == mx.gpu and mx.metal.is_available():
        return gated_delta_kernel(q, k, v, g, beta, state)
    return gated_delta_ops(q, k, v, g, beta, state)


class TridaRuntime:
    def __init__(self, model: qwen3_5.Model, prefill_chunk: int = 512, fused_gdn: bool = True):
        self.model = model
        # one Metal launch per GDN layer for the canvas (GPU only; ops fallback elsewhere)
        self.fused_gdn = fused_gdn
        # cold start: LM head on rows 0..N-1 only. Off by default: measured slower on M3 Pro
        # (46.2 -> 48.3 ms) since the head is bandwidth-bound and rows are nearly free.
        self.skip_rows = False
        self.lm = model.language_model  # TextModel
        self.backbone = self.lm.model  # Qwen3_5TextModel
        self.args = self.lm.args
        self.prefill_chunk = prefill_chunk
        self.progress_cb = None  # optional fn(tokens_done_this_call) during long prefills
        self._mask_cache: dict = {}

    # -- cache ------------------------------------------------------------------
    def make_cache(self) -> SeqCache:
        layers = []
        for layer in self.backbone.layers:
            if layer.is_linear:
                lin = layer.linear_attn
                dtype = self.backbone.norm.weight.dtype
                layers.append(
                    GDNState(
                        conv=mx.zeros((1, lin.conv_kernel_size - 1, lin.conv_dim), dtype=dtype),
                        ssm=mx.zeros((1, lin.num_v_heads, lin.head_v_dim, lin.head_k_dim), dtype=mx.float32),
                    )
                )
            else:
                layers.append(KVCache())
        return SeqCache(layers=layers)

    # -- layers -----------------------------------------------------------------
    @staticmethod
    def _attention(attn, x, kv: KVCache, mask):
        B, L, _ = x.shape
        qo = attn.q_proj(x)
        queries, gate = mx.split(qo.reshape(B, L, attn.num_attention_heads, -1), 2, axis=-1)
        gate = gate.reshape(B, L, -1)
        keys, values = attn.k_proj(x), attn.v_proj(x)
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        keys = attn.k_norm(keys.reshape(B, L, attn.num_key_value_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, attn.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        queries = attn.rope(queries, offset=kv.offset)
        keys = attn.rope(keys, offset=kv.offset)
        keys, values = kv.update_and_fetch(keys, values)
        out = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
        out = out.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return attn.o_proj(out * mx.sigmoid(gate))

    def _gdn(self, lin, x, st: GDNState, n_clean: Optional[int]):
        """Returns (out, new_state_or_None, canvas_record_or_None)."""
        B, S, _ = x.shape
        qkv = lin.in_proj_qkv(x)
        z = lin.in_proj_z(x).reshape(B, S, lin.num_v_heads, lin.head_v_dim)
        b = lin.in_proj_b(x)
        a = lin.in_proj_a(x)
        conv_input = mx.concatenate([st.conv.astype(qkv.dtype), qkv], axis=1)
        conv_out = nn.silu(lin.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [lin.key_dim, 2 * lin.key_dim], -1),
                [lin.num_k_heads, lin.num_k_heads, lin.num_v_heads],
                [lin.head_k_dim, lin.head_k_dim, lin.head_v_dim],
            )
        ]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        beta = mx.sigmoid(b)
        g = compute_g(lin.A_log, a, lin.dt_bias)

        if n_clean is None:  # plain causal (prefill / AR step)
            y, s = _gdr(q, k, v, g, beta, st.ssm)
            n_keep = lin.conv_kernel_size - 1
            new = GDNState(conv=conv_input[:, -n_keep:], ssm=s)
            rec = None
        else:  # canvas: clean rows token-causal, MASK rows read the block-end state
            n = n_clean
            if self.fused_gdn and fused_available():
                y, s_clean = canvas_gdn_fused(q, k, v, g, beta, st.ssm, n)
            else:
                y1, s_clean = _gdr(q[:, :n], k[:, :n], v[:, :n], g[:, :n], beta[:, :n], st.ssm)
                _, s_end = _gdr(q[:, n:], k[:, n:], v[:, n:], g[:, n:], beta[:, n:], s_clean)
                qm = q[:, n:]
                rep = lin.num_v_heads // lin.num_k_heads
                if rep > 1:
                    qm = mx.repeat(qm, rep, axis=2)
                y2 = mx.einsum("bhvk,bthk->bthv", s_end, qm.astype(mx.float32)).astype(y1.dtype)
                y = mx.concatenate([y1, y2], axis=1)
            new = None
            rec = _CanvasGDN(
                prev=st, conv_input=conv_input,
                q=q[:, :n], k=k[:, :n], v=v[:, :n], g=g[:, :n], beta=beta[:, :n],
                s_clean=s_clean,
            )
        out = lin.norm(y, z)
        return lin.out_proj(out.reshape(B, S, -1)), new, rec

    def _lm_head(self, h):
        if self.args.tie_word_embeddings:
            return self.backbone.embed_tokens.as_linear(h)
        return self.lm.lm_head(h)

    def _forward(self, ids: mx.array, cache: SeqCache, attn_mask, n_clean, last_only: bool,
                 rows: Optional[int] = None):
        h = self.backbone.embed_tokens(ids)
        recs = [] if n_clean is not None else None
        for i, layer in enumerate(self.backbone.layers):
            x = layer.input_layernorm(h)
            c = cache.layers[i]
            if layer.is_linear:
                r, new, rec = self._gdn(layer.linear_attn, x, c, n_clean)
                if new is not None:
                    cache.layers[i] = new
                if recs is not None:
                    recs.append(rec)
            else:
                r = self._attention(layer.self_attn, x, c, attn_mask)
                if recs is not None:
                    recs.append(None)
            h = h + r
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.backbone.norm(h)
        if last_only:
            h = h[:, -1:]
        elif rows is not None:
            h = h[:, :rows]
        return self._lm_head(h), recs

    # -- public forwards --------------------------------------------------------
    def prefill(self, cache: SeqCache, tokens: list) -> mx.array:
        """Causal prefill of ``tokens`` after whatever ``cache`` already holds.
        Returns the float32 logits [V] of the last token."""
        assert cache.pending is None and tokens
        logits = None
        for s in range(0, len(tokens), self.prefill_chunk):
            chunk = tokens[s : s + self.prefill_chunk]
            ids = mx.array([chunk], dtype=mx.int32)
            mask = "causal" if len(chunk) > 1 else None
            logits, _ = self._forward(ids, cache, mask, None, last_only=True)
            cache.tokens.extend(chunk)
            if s + self.prefill_chunk < len(tokens):
                mx.eval([l.ssm if isinstance(l, GDNState) else l.keys for l in cache.layers])
                if self.progress_cb is not None:
                    self.progress_cb(len(chunk))
        return logits[0, -1].astype(mx.float32)

    def ar_step(self, cache: SeqCache, token: int) -> mx.array:
        logits, _ = self._forward(mx.array([[token]], dtype=mx.int32), cache, None, None, last_only=True)
        cache.tokens.append(token)
        return logits[0, -1].astype(mx.float32)

    def _canvas_mask(self, prefix_len: int, blk: int, n_clean: int) -> mx.array:
        key = (blk, n_clean)
        inner = self._mask_cache.get(key)
        if inner is None:
            r = mx.arange(blk)[:, None]
            c = mx.arange(blk)[None, :]
            inner = (c <= r) | (r >= n_clean)  # clean rows causal, MASK rows see the whole canvas
            self._mask_cache[key] = inner
        pre = mx.ones((blk, prefix_len), dtype=mx.bool_)
        return mx.concatenate([pre, inner], axis=1)

    def canvas(self, cache: SeqCache, canvas_tokens: list, n_clean: int, rows: Optional[int] = None) -> mx.array:
        """One self-spec forward. Returns float32 logits [rows or blk, V] (shift readout: row i
        predicts the token at canvas position i+1). ``rows`` limits the LM head to the first
        rows (a cold start only reads rows 0..N-1). Call ``commit`` afterwards."""
        assert cache.pending is None
        blk = len(canvas_tokens)
        mask = self._canvas_mask(cache.length, blk, n_clean)
        ids = mx.array([canvas_tokens], dtype=mx.int32)
        logits, recs = self._forward(ids, cache, mask, n_clean, last_only=False, rows=rows)
        cache.pending = recs
        cache.canvas_len = blk
        self._canvas_tokens = list(canvas_tokens)
        return logits[0].astype(mx.float32)

    def commit(self, cache: SeqCache, adv: int) -> None:
        """Keep the first ``adv`` (1..n_clean) canvas tokens in the cache."""
        recs, blk = cache.pending, cache.canvas_len
        assert recs is not None and 1 <= adv
        for i, rec in enumerate(recs):
            if rec is None:
                cache.layers[i].trim(blk - adv)
                continue
            n = rec.q.shape[1]
            assert adv <= n
            n_keep = rec.conv_input.shape[1] - blk
            conv = rec.conv_input[:, adv : adv + n_keep]
            if adv == n:
                ssm = rec.s_clean
            else:
                _, ssm = _gdr(rec.q[:, :adv], rec.k[:, :adv], rec.v[:, :adv],
                              rec.g[:, :adv], rec.beta[:, :adv], rec.prev.ssm)
            cache.layers[i] = GDNState(conv=conv, ssm=ssm)
        cache.tokens.extend(self._canvas_tokens[:adv])
        cache.pending = None
        cache.canvas_len = 0
