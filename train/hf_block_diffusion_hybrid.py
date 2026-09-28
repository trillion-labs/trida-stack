"""Block-diffusion for HYBRID linear-attention decoders — Qwen/Qwen3.5.

Qwen3.5 (``transformers.models.qwen3_5``) is a HYBRID model: 3 of every 4
layers are ``"linear_attention"`` (gated-delta recurrence, ``Qwen3_5GatedDeltaNet``) and only every
4th is ``"full_attention"`` (see ``config.layer_types``). A recurrent linear scan CANNOT consume an
arbitrary attention mask — it is a left-to-right state machine — so the dense flex-mask path does not
apply to the linear layers.

This module keeps the dense path 100% untouched (we SUBCLASS ``HFBlockDiffusion``) and adds a
linear-attention-aware forward:

  * ``full_attention`` layers  — get the exact block-diffusion mask the dense trainer builds
    (prompt-causal ``S``; ``x_t`` bidirectional within its block + attends earlier clean ``x_0``
    blocks; ``x_0`` block/token-causal). This is where the bidirectional denoising signal flows.
  * ``linear_attention`` layers — the combined ``[ S | x_t | x_0 ]`` sequence is split into the two
    independent conditioning streams region-A ``[S | x_t]`` (length ``L``) and region-B ``[x_0]``
    (length ``rpad``); the recurrent layer is applied to each separately (a pointwise MLP/RMSNorm is
    unaffected by the split, only the token-mixer is — so this isolates ``x_0`` from ``x_t`` exactly
    as the full-attention mask does). Two modes, ``--linear_block_mode``:
      - ``causal`` (first cut, CPU-verifiable): region-A scanned as one causal gated-delta sequence
        (``x_t`` conditioned on the prompt + earlier — still noised — response tokens); region-B
        scanned causally on its own. Intra-block bidirectionality comes ONLY from the full layers.
      - ``bidirectional`` (research-level, best-effort): region-A uses a block-local-bidirectional
        gated-delta scan (``_block_bidirectional_gdr``): the recurrent state is carried causally
        across diffusion blocks, but WITHIN each block a forward and a backward sub-scan (both from
        the block's causal entry state) are averaged, giving intra-block bidirectionality in the
        linear layers too. Region-B stays causal. This reference is pure-torch, O(T) python-looped,
        and UNVERIFIED against any fused kernel — it needs GPU + FLA-kernel work to be usable at
        scale.

CPU note: FlexAttention has no CPU backward, and the FLA / causal-conv1d kernels are CUDA-only.
For CPU shape/gradient tests set ``attn_impl="eager"`` (full layers consume a dense additive mask
instead of a flex ``BlockMask``) and rely on transformers' built-in torch fallbacks for the linear
layers (``torch_chunk_gated_delta_rule`` / ``F.conv1d``). Production uses ``attn_impl=
"flex_attention"`` + FLA kernels on GPU.

Scope: the single-turn ``forward`` path (contiguous response span), the packed (sequence-packing)
path in ``causal`` linear mode (``forward_packed_hybrid``), and pure-AR (``--ar_only``) both unpacked
(inherited native causal forward) and packed (``forward_ar``, document-isolated). The multi-turn path
and bidirectional/merged-views packing raise ``NotImplementedError``.
"""

import functools
import os
import sys
import warnings

import torch
import torch.nn.functional as F

from transformers import AutoConfig

from train.hf_block_diffusion import (
    HFBlockDiffusion,
    bucketed_clean_len,
    _PAD,
    _SHARED,
    _XT,
    _X0,
)


# ----------------------------------------------------------------------------------------
# hybrid detection
# ----------------------------------------------------------------------------------------
def _layer_types_of(config):
    """Return the decoder ``layer_types`` list, unwrapping a nested ``text_config`` (the Qwen3.5
    multimodal config nests the text tower). ``None`` if the model does not declare layer types."""
    lt = getattr(config, "layer_types", None)
    if lt is None and hasattr(config, "text_config"):
        lt = getattr(config.text_config, "layer_types", None)
    return lt


def is_hybrid_config(config) -> bool:
    """A model is *hybrid* (needs this trainer) iff it declares ``layer_types`` with at least one
    non-``full_attention`` layer. An all-``full_attention`` model is a plain dense decoder and must
    use the dense ``HFBlockDiffusion`` path unchanged."""
    lt = _layer_types_of(config)
    return bool(lt) and any(t != "full_attention" for t in lt)


def is_hybrid_model_id(model_id: str, trust_remote_code: bool = False) -> bool:
    cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=trust_remote_code)
    return is_hybrid_config(cfg)


# ----------------------------------------------------------------------------------------
# packed (sequence-packing) helpers — document isolation for the linear layers (CAUSAL mode)
# ----------------------------------------------------------------------------------------
# A packed row concatenates several documents. The linear (gated-delta) layers are recurrent state
# machines, so they need an explicit RESET at each document boundary or state (and the causal conv's
# receptive field) leaks across documents. The fused FLA/causal-conv1d kernels do this natively when
# handed ``cu_seqlens`` / ``seq_idx``; the torch reference kernels used on CPU IGNORE those kwargs,
# so we bridge the gap with two per-instance wrappers installed on every linear layer (below). The
# wrappers are INERT when no packing metadata is present (single-turn path) — they delegate straight
# to the original kernel and are bit-identical to the unpacked behavior.


def _seq_idx_from_seg(seg2d: torch.Tensor) -> torch.Tensor:
    """Per-token integer document index the causal-conv wants (``seq_idx``, shape ``[B, T]``): a
    0-based counter that increments at every change of ``seg2d`` along a row (document boundaries AND
    pad boundaries). Equal, contiguous values => one document; ``causal_conv1d_fn`` resets its rolling
    conv window wherever adjacent ``seq_idx`` differ."""
    change = torch.zeros_like(seg2d, dtype=torch.long)
    change[:, 1:] = (seg2d[:, 1:] != seg2d[:, :-1]).long()
    return change.cumsum(dim=1).to(torch.int32)


def _cu_seqlens_from_seg(seg2d: torch.Tensor) -> torch.Tensor:
    """Cumulative sequence lengths (``cu_seqlens``, 1-D ``[num_seg + 1]`` int32) in the FLA varlen
    convention: offsets into the FLATTENED ``[B*T]`` batch marking the start of every contiguous
    same-document run, plus a final ``B*T``. Each row start is forced to be a boundary so no document
    ever spans two rows. This partitions ``[0, B*T]`` contiguously with no gaps."""
    B, T = seg2d.shape
    change = torch.zeros(B, T, dtype=torch.bool, device=seg2d.device)
    change[:, 0] = True                                   # every row starts a new document
    change[:, 1:] = seg2d[:, 1:] != seg2d[:, :-1]
    starts = change.reshape(-1).nonzero().flatten()
    end = torch.tensor([B * T], device=seg2d.device, dtype=starts.dtype)
    return torch.cat([starts, end]).to(torch.int32)


def _positions_from_seg(seg2d: torch.Tensor) -> torch.Tensor:
    """Per-document position ids: within each contiguous same-doc run, positions restart at 0
    (``[0,1,2, 0,1,2,3, ...]``). Packed docs must use reset positions so RoPE matches the unpacked
    forward — otherwise a doc at a later absolute offset would get different rotary phases."""
    B, L = seg2d.shape
    ar = torch.arange(L, device=seg2d.device)[None, :].expand(B, -1)
    is_start = torch.ones(B, L, dtype=torch.bool, device=seg2d.device)
    is_start[:, 1:] = seg2d[:, 1:] != seg2d[:, :-1]
    start_pos = torch.cummax(torch.where(is_start, ar, torch.zeros_like(ar)), dim=1).values
    return ar - start_pos


def _is_fused_gdr(la) -> bool:
    """True iff this linear layer's ``chunk_gated_delta_rule`` is FLA's FUSED kernel (which honors
    ``cu_seqlens`` natively, in ONE launch) rather than transformers' ``torch_chunk_gated_delta_rule``
    reference (which swallows ``cu_seqlens`` in ``**kwargs`` and IGNORES it).

    Derived from the actual bound attribute, not from the availability probes: the probes are
    ``lru_cache``d and the module-level binding happens at ``modeling_qwen3_5`` import time, so the two
    can disagree (the CPU tests deliberately force the probes to ``False`` before that import). Getting
    this wrong in the "probe says fused, kernel is actually torch" direction would silently drop
    document isolation, so read the binding itself."""
    mod = sys.modules.get(type(la).__module__)
    fused = getattr(mod, "chunk_gated_delta_rule", None)
    return fused is not None and la.chunk_gated_delta_rule is fused


def _make_varlen_gdr(orig, fused: bool = False):
    """Wrap a linear layer's ``chunk_gated_delta_rule`` so packed rows get a per-document recurrent
    state reset. When ``cu_seqlens`` is absent -> delegate to ``orig`` unchanged (unpacked path).
    When present, the ``[B, T, H, D]`` inputs are flattened to a single ``[1, B*T, ...]`` varlen
    sequence (``_cu_seqlens_from_seg`` indexes that flattened layout, and forces a boundary at every
    row start so no document spans two rows) and then:

      * ``fused=True`` (real FLA kernel): ONE fused varlen launch with ``cu_seqlens`` — the kernel
        resets the gated-delta state at every boundary itself. The flatten is NOT optional: FLA
        raises ``"The batch size is expected to be 1 ... when using cu_seqlens"`` for ``B > 1``, and
        the packed BD path always runs ``BB = 2B >= 2`` (and micro-batch > 1 does the same for AR),
        so passing the model's native ``[B, T, ...]`` straight through would crash.
      * ``fused=False`` (transformers' torch reference, i.e. CPU): ``orig`` ignores ``cu_seqlens``, so
        we emulate the reset by calling it once per document slice with a FRESH state. O(num_docs)
        python-level launches — correct but slow, so it is only installed when there is no fused
        kernel to launch.

    ``cu_seqlens_cpu`` is load-bearing for performance: FLA's ``get_max_num_splits`` calls python
    ``max()`` over ``cu_seqlens``, which is one ``.item()`` device sync per document per layer per
    direction when the tensor lives on GPU; handing it a CPU copy keeps that on the host.
    """
    def scan(query, key, value, g=None, beta=None, **kw):
        cu = kw.get("cu_seqlens", None)
        if cu is None:
            return orig(query, key, value, g=g, beta=beta, **kw)
        B, T = query.shape[:2]
        Hv, Dv = value.shape[2], value.shape[3]

        def flat(x):
            return x.reshape(1, B * T, *x.shape[2:])

        qf, kf, vf, gf, bf = flat(query), flat(key), flat(value), flat(g), flat(beta)
        if fused:
            # Memoised on the cu_seqlens tensor itself: the trainer builds one per region per forward
            # and hands the SAME tensor to every linear layer, so only the first layer pays the copy.
            cpu = kw.get("cu_seqlens_cpu", None)
            if cpu is None:
                cpu = getattr(cu, "_trida_cu_cpu", None)
            if cpu is None and cu.device.type != "cpu":
                cpu = cu.detach().to("cpu")
                try:
                    cu._trida_cu_cpu = cpu
                except AttributeError:                # e.g. a traced/fake tensor — just skip the cache
                    pass
            if cpu is not None:
                kw["cu_seqlens_cpu"] = cpu
            out, state = orig(qf, kf, vf, g=gf, beta=bf, **kw)
            return out.reshape(B, T, Hv, Dv), state
        uql = kw.get("use_qk_l2norm_in_kernel", False)
        cul = cu.tolist()
        outs = []
        for i in range(len(cul) - 1):
            s, e = int(cul[i]), int(cul[i + 1])
            if e <= s:
                continue
            o, _ = orig(qf[:, s:e], kf[:, s:e], vf[:, s:e], g=gf[:, s:e], beta=bf[:, s:e],
                        initial_state=None, output_final_state=False, use_qk_l2norm_in_kernel=uql)
            outs.append(o)
        out = torch.cat(outs, dim=1)                      # [1, B*T, Hv, Dv], segments in order
        return out.reshape(B, T, Hv, Dv), None
    return scan


def _two_stream_shared_refine(q, k, v, g, beta, *, L, V, cu, block_size, causal_mode_clean,
                              causal_mode_noisy):
    """Shared-clean two-stream recurrence for ``[x0 ; xt_1 ; … ; xt_V]`` (``q``… are ``[B, (1+V)L, H, D]``,
    q/k already l2-normed). Replicates ``_block_train_chunk_refine`` EXACTLY but computes the clean
    pass (``o_clean``) and the refined clean chunk-states ``h_all`` ONCE and runs only the noisy pass per
    view — the two-call wrapper instead re-ran clean + h_all for every view and materialised
    ``cat([x0, xt_v])`` copies. ``h_all`` stays differentiable, so the noisy losses' gradients reach
    the clean projections through the states from every view (autograd sums them), just as in
    ``chunk_refine`` where each view recomputed an identical ``h_all``. Packed (``cu`` given): every
    stream is flattened to ``[1, B*L]`` (kernel varlen contract, batch==1). Returns
    ``[B, (1+V)L, H, Dv]``."""
    from train.block_gated_delta_rule import chunk as _ck
    B0 = q.shape[0]
    assert L % block_size == 0, f"shared-refine needs block-aligned L (L={L}, block={block_size})"
    packed = cu is not None

    def stream(x, i):                                   # stream i of the layer input, kernel layout
        s = x[:, i * L:(i + 1) * L]
        return s.reshape(1, B0 * L, *s.shape[2:]) if packed else s

    q_c, k_c, v_c, g_c, b_c = (stream(t, 0) for t in (q, k, v, g, beta))
    Bk, Tc, H, K = q_c.shape
    Dv = v_c.shape[-1]
    scale = K ** -0.5
    chunk_size = _ck.compute_aligned_chunk_size(block_size)
    N = Tc // block_size
    # Clean pass that also returns the chunk-boundary states (h_chunks, differentiable). Feeding
    # them to chunk_refine_gda_state skips recomputing those states inside h_all; unlike the vendored
    # `chunk_refine_reuse`, h_chunks is NOT detached, so gradients stay exact.
    o_clean, _, h_chunks = _ck.ChunkBlockCausalGDAWithHFunction.apply(
        q_c, k_c, v_c, g_c, b_c, scale, block_size, None, False,
        causal_mode_clean, cu, False, chunk_size)
    h_all = _ck.chunk_refine_gda_state(k=k_c, v=v_c, g=g_c, beta=b_c, h0=None, h_chunks=h_chunks,
                                       block_size=block_size, chunk_size=chunk_size, cu_seqlens=cu)
    outs = [o_clean]
    for vi in range(1, V + 1):
        q_n, k_n, v_n, g_n, b_n = (stream(t, vi) for t in (q, k, v, g, beta))
        outs.append(_ck._noisy_batched_forward(
            q_n, k_n, v_n, g_n, b_n, h_all, Bk, N, block_size, H, K, Dv, scale, False,
            causal_mode_noisy=causal_mode_noisy, chunk_size=chunk_size, cu_seqlens=cu))
    if packed:
        outs = [o.reshape(B0, L, H, Dv) for o in outs]
    return torch.cat(outs, dim=1)


def _make_two_stream_gdr(orig, block_size, causal_mode_clean, causal_mode_noisy, nviews_getter=None):
    """Wrap a linear layer's ``chunk_gated_delta_rule`` with FLARE's two-stream block-diffusion GDN.

    The HF ``Qwen3_5GatedDeltaNet.forward`` is called ONCE on the combined ``[x0 ; xt]`` sequence
    (clean first half, noisy second half), so ``query/key/value`` arrive here as ``[B, 2*half, H, D]``
    — exactly the layout FLARE's ``chunk_block_causal_gated_delta_rule(block_train=True)`` expects
    (it splits at ``half = T//2`` internally). We l2-normalise q/k ourselves (the fla-style two-stream
    path forbids in-kernel l2norm, ``chunk.py`` ``NotImplementedError``) and dispatch the fused
    kernel. ``causal_mode_noisy=0`` == true bidirectional-within-block xt (the real method);
    ``causal_mode_noisy=1`` == causal xt (the old approximation). Returns ``(out [B,2*half,Hv,Dv],
    state)``; ``orig`` is kept only as the fallback the caller can restore for a non-block forward."""
    from train.block_gated_delta_rule import chunk_block_causal_gated_delta_rule as _gdr

    def _l2norm(x, dim=-1, eps=1e-6):   # FLARE's model.py:_l2norm (exact)
        return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)

    def _call(q, k, v, g, beta, cu):
        return _gdr(q=q, k=k, v=v, g=g, beta=beta, scale=None, block_size=block_size,
                    initial_state=None, output_final_state=False,
                    block_train=True, block_train_method="auto",
                    causal_mode_clean=causal_mode_clean, causal_mode_noisy=causal_mode_noisy,
                    cu_seqlens=cu, use_qk_l2norm_in_kernel=False)[0]

    def _pair(q, k, v, g, beta, cu):
        """One ``[x0 ; xt]`` pair, ``[BB, 2L, ...]`` -> ``[BB, 2L, Hv, Dv]`` (unpacked or packed)."""
        if cu is None:                       # unpacked: direct [BB, 2L] block_train
            return _call(q, k, v, g, beta, None)
        # packed: FLA block_train + cu_seqlens needs batch==1. Flatten the [x0 ; xt] halves of each
        # of the BB rows to one row: [BB, 2L] -> ([1, BB*L] clean ++ [1, BB*L] noisy). L is already a
        # bd_size multiple (collator bucket), so no per-row padding/remap is needed. `cu` is the flat
        # [BB*L] doc-offset table for the clean half; the kernel applies it to both halves.
        BB, twoL = q.shape[0], q.shape[1]
        L = twoL // 2

        def fc(x):  # split at L, flatten each half to [1, BB*L, ...], concat -> [1, 2*BB*L, ...]
            xc, xn = x[:, :L], x[:, L:]
            return torch.cat([xc.reshape(1, BB * L, *xc.shape[2:]),
                              xn.reshape(1, BB * L, *xn.shape[2:])], dim=1)

        out = _call(fc(q), fc(k), fc(v), fc(g), fc(beta), cu)   # [1, 2*BB*L, Hv, Dv]
        Hv, Dv = out.shape[2], out.shape[3]
        oc = out[:, :BB * L].reshape(BB, L, Hv, Dv)
        on = out[:, BB * L:].reshape(BB, L, Hv, Dv)
        return torch.cat([oc, on], dim=1)

    def scan(query, key, value, g=None, beta=None, **kw):
        q = _l2norm(query, dim=-1, eps=1e-6)
        k = _l2norm(key, dim=-1, eps=1e-6)
        cu = kw.get("cu_seqlens", None)
        V = nviews_getter() if nviews_getter is not None else 1
        if V == 1:                           # plain [x0 ; xt]
            return _pair(q, k, value, g, beta, cu), None
        # Shared clean stream: the layer sees [x0 ; xt_1 ; … ; xt_V] (length (1+V)·L). The kernel's
        # contract is one [clean ; noisy] pair of equal halves, and the noisy stream only READS the
        # clean chunk states (clean never sees noisy), so run one pair per view against the same x0
        # and keep the clean output from the first pair. The recurrence still recomputes the clean
        # states per view; everything outside the recurrence (projections, MLP, attention, norms)
        # is computed once for x0 by construction of the layout.
        # Shared clean stream, layer input [x0 ; xt_1 ; … ; xt_V]: one clean pass + one set of clean
        # chunk-states + one noisy pass per view, no cats (see _two_stream_shared_refine). Exact.
        L = q.shape[1] // (1 + V)
        return _two_stream_shared_refine(
            q, k, value, g, beta, L=L, V=V, cu=cu, block_size=block_size,
            causal_mode_clean=causal_mode_clean, causal_mode_noisy=causal_mode_noisy), None
    return scan


def _make_two_stream_conv(block_size, cu_getter=None, nviews_getter=None):
    """Wrap a linear layer's ``causal_conv1d_fn`` with FLARE's two-stream block-diffusion ShortConv.

    The HF layer calls ``self.causal_conv1d_fn(x=[B, conv_dim, T], weight=[conv_dim, width], bias,
    activation, seq_idx)`` once on the combined ``[x0 ; xt]`` ``mixed_qkv``. FLARE's ``block_train_conv``
    (Triton two-stream / fla-batched — no ``causal_conv1d`` dependency) takes ``x=[B, T, conv_dim]`` and
    internally splits at ``T//2`` so conv taps inside a noisy block read the noisy stream and taps
    before it read the clean stream (no noisy->clean leakage across the x0|xt boundary). We transpose
    in/out to bridge the channels-first vs channels-last convention. ``cu_getter`` returns the current
    per-forward ``cu_seqlens`` (packed) or ``None`` (unpacked)."""
    from train.block_gated_delta_rule.convolution import block_train_conv

    # Pin 'fla_batched': the 'twostream' method's backward Triton kernel has an illegal-memory-access
    # bug on packed multi-document inputs. Both methods compute the identical two-stream conv;
    # QWEN35_BLOCK_CONV_METHOD overrides.
    conv_method = "fla_batched"

    def _pair(xt, weight, bias, act, cu):
        """One ``[x0 ; xt]`` pair ``[BB, 2L, C]`` -> ``[BB, 2L, C]`` (unpacked or packed)."""
        if cu is None:                                            # unpacked
            return block_train_conv(xt, weight, bias, block_size=block_size,
                                    activation=act, cu_seqlens=None, method=conv_method)
        # packed: mirror the recurrence flatten — [BB, 2L, C] -> [1, 2*BB*L, C] ([x0_flat ++ xt_flat])
        BB, twoL, C = xt.shape
        L = twoL // 2
        xc, xn = xt[:, :L], xt[:, L:]
        flat = torch.cat([xc.reshape(1, BB * L, C), xn.reshape(1, BB * L, C)], dim=1)
        out = block_train_conv(flat, weight, bias, block_size=block_size,
                               activation=act, cu_seqlens=cu, method=conv_method)   # [1, 2*BB*L, C]
        oc = out[:, :BB * L].reshape(BB, L, C)
        on = out[:, BB * L:].reshape(BB, L, C)
        return torch.cat([oc, on], dim=1)                         # [BB, 2L, C]

    def conv_fn(x, weight, bias=None, activation=None, seq_idx=None, **_):
        xt = x.transpose(1, 2).contiguous()                       # [B, T=(1+V)L, conv_dim]
        cu = cu_getter() if cu_getter is not None else None
        act = activation or "silu"
        V = nviews_getter() if nviews_getter is not None else 1
        if V == 1:
            return _pair(xt, weight, bias, act, cu).transpose(1, 2).contiguous()
        # shared clean stream (see _make_two_stream_gdr): one [x0 ; xt_v] pair per view; the conv is
        # depthwise/width-4 so the repeated clean half is negligible.
        L = xt.shape[1] // (1 + V)
        outs = []
        for vi in range(V):
            sl = slice((1 + vi) * L, (2 + vi) * L)
            o = _pair(torch.cat([xt[:, :L], xt[:, sl]], dim=1), weight, bias, act, cu)
            if vi == 0:
                outs.append(o[:, :L])
            outs.append(o[:, L:])
        return torch.cat(outs, dim=1).transpose(1, 2).contiguous()   # [B, conv_dim, (1+V)L]
    return conv_fn


def _flare_block_diff_mask_mod(b, h, q_idx, kv_idx, *, block_size, n, causal_x0, causal_xt, doc_ids=None):
    """FLARE's ``_block_diff_mask_x0_xt`` (dllm_model.py) generalised to ``[x0 ; xt_1 ; … ; xt_V]``
    (total length ``(1+V)·n``; stream ``s = idx // n``, position ``idx % n``). x0 (stream 0) is
    block/token-causal; each noisy view xt_v is bidirectional/causal WITHIN its block and cross-attends
    EARLIER x0 blocks; DIFFERENT noisy views never see each other (they are independent samples that
    merely share the clean stream). With V=1 this is exactly the original ``[x0 ; xt]`` mask.
    ``doc_ids`` (length ``(1+V)·n``) confines attention to one packed doc."""
    q_s = q_idx // n
    kv_s = kv_idx // n
    q_is_xt = q_s >= 1
    kv_is_xt = kv_s >= 1
    q_pos = q_idx - q_s * n
    kv_pos = kv_idx - kv_s * n
    q_block = q_pos // block_size
    kv_block = kv_pos // block_size
    if causal_x0:
        x0_mask = (~q_is_xt) & (~kv_is_xt) & (q_pos >= kv_pos)
    else:
        x0_mask = (~q_is_xt) & (~kv_is_xt) & (q_block >= kv_block)
    same_view = q_s == kv_s
    if causal_xt:
        xt_mask = q_is_xt & kv_is_xt & same_view & (q_block == kv_block) & (q_pos >= kv_pos)
    else:
        xt_mask = q_is_xt & kv_is_xt & same_view & (q_block == kv_block)
    cross_mask = q_is_xt & (~kv_is_xt) & (q_block > kv_block)
    mask = x0_mask | xt_mask | cross_mask
    if doc_ids is not None:
        mask = mask & (doc_ids[b, q_idx] == doc_ids[b, kv_idx])
    return mask | (q_idx == kv_idx)   # self-diagonal so no row softmaxes over an empty key set


def _torch_causal_conv1d_varlen(x, weight, bias=None, activation=None, seq_idx=None, **_):
    """Torch ``causal_conv1d_fn`` replacement that honors ``seq_idx`` (document reset). ``x``:
    ``[B, dim, T]``; ``weight``: ``[dim, width]``; ``seq_idx``: ``[B, T]`` int or ``None``. With
    ``seq_idx=None`` it is a plain causal depthwise conv (bit-identical to ``F.silu(conv1d(x))[:T]``,
    i.e. the model's own torch fallback) so the UNPACKED path is unchanged. With ``seq_idx`` given,
    each contiguous same-``seq_idx`` run is convolved on its own with fresh (zero) left-padding, so
    the conv's receptive field never crosses a document boundary."""
    width = weight.shape[-1]
    dim = x.shape[1]
    w = weight.unsqueeze(1)                               # [dim, 1, width] depthwise

    def conv_seg(xs):
        o = F.conv1d(xs, w, bias, padding=width - 1, groups=dim)[..., : xs.shape[-1]]
        if activation in ("silu", "swish"):
            o = F.silu(o)
        elif activation is not None:
            from transformers.activations import ACT2FN
            o = ACT2FN[activation](o)
        return o

    if seq_idx is None:
        return conv_seg(x)
    B, _, T = x.shape
    rows = []
    for b in range(B):
        si = seq_idx[b]
        change = torch.ones(T, dtype=torch.bool, device=x.device)
        change[1:] = si[1:] != si[:-1]
        starts = change.nonzero().flatten().tolist() + [T]
        segs = [conv_seg(x[b:b + 1, :, starts[i]:starts[i + 1]]) for i in range(len(starts) - 1)]
        rows.append(torch.cat(segs, dim=-1))
    return torch.cat(rows, dim=0)


# ----------------------------------------------------------------------------------------
# block-local-bidirectional gated-delta reference scan (research-level, UNVERIFIED)
# ----------------------------------------------------------------------------------------
def _l2norm(x, eps: float = 1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _block_bidirectional_gdr(
    query, key, value, g, beta, *, block_ids,
    chunk_size: int = 64, initial_state=None, output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False, **_,
):
    """Block-local-bidirectional gated-delta rule (pure-torch reference).

    Drop-in for ``Qwen3_5GatedDeltaNet.chunk_gated_delta_rule``: same tensor layout — ``query/key/
    value`` are ``[B, T, H, D]``, ``g``/``beta`` are ``[B, T, H]``. ``block_ids`` (``[B, T]``, long)
    tags each token with its diffusion block (equal, contiguous ids = one block); tokens in
    different blocks are processed CAUSALLY (state carried left-to-right), tokens WITHIN a block get
    a bidirectional read-out.

    Per block ``K`` with causal entry state ``E_K`` (the recurrent state after the prompt + all
    earlier blocks):
      * forward sub-scan from ``E_K`` over the block's tokens in order  -> ``out_f`` (= the plain
        causal output, since ``E_K`` is exactly the causal state at the block start);
      * backward sub-scan from ``E_K`` over the block's tokens in reverse -> ``out_b``;
      * ``out = ½(out_f + out_b)``.
    The state handed to the NEXT block is the forward (causal) exit state, so cross-block
    conditioning stays causal. Averaging is one concrete combination choice (sum/concat+proj are
    alternatives) — this is a research reference, numerically UNVERIFIED against a fused kernel and
    memory-heavy (snapshots a per-token entry state); do not use at scale without kernel work.
    """
    dtype0 = query.dtype
    if use_qk_l2norm_in_kernel:
        query, key = _l2norm(query), _l2norm(key)
    # -> [B, H, T, D] float, matching transformers' torch reference kernels
    q, k, v, beta_, g_ = (x.transpose(1, 2).float() for x in (query, key, value, beta, g))
    Bb, H, T, Dk = k.shape
    Dv = v.shape[-1]
    q = q * (Dk ** -0.5)

    bid = block_ids  # [B, T]
    start = torch.ones(Bb, T, dtype=torch.bool, device=q.device)
    start[:, 1:] = bid[:, 1:] != bid[:, :-1]        # first token of each block
    last = torch.ones(Bb, T, dtype=torch.bool, device=q.device)
    last[:, :-1] = bid[:, 1:] != bid[:, :-1]         # last token of each block

    def step(state, qt, kt, vt, gt, bt):
        # one gated-delta step (mirrors torch_recurrent_gated_delta_rule): state [B,H,Dk,Dv].
        state = state * gt.exp()[..., None, None]
        kv_mem = (state * kt[..., None]).sum(dim=-2)              # [B,H,Dv]
        delta = (vt - kv_mem) * bt[..., None]
        state = state + kt[..., None] * delta[..., None, :]
        out = (state * qt[..., None]).sum(dim=-2)                # [B,H,Dv]
        return state, out

    zero = (torch.zeros(Bb, H, Dk, Dv, device=q.device, dtype=q.dtype)
            if initial_state is None else initial_state.float())

    # forward causal pass: causal outputs + the entry state of each token's block
    state = zero
    out_f = torch.zeros(Bb, H, T, Dv, device=q.device, dtype=q.dtype)
    block_entry = torch.zeros(Bb, H, T, Dk, Dv, device=q.device, dtype=q.dtype)
    cur_entry = state
    for t in range(T):
        st = start[:, t][:, None, None, None]
        cur_entry = torch.where(st, state, cur_entry)            # snapshot at block start
        block_entry[:, :, t] = cur_entry
        state, o = step(state, q[:, :, t], k[:, :, t], v[:, :, t], g_[:, :, t], beta_[:, :, t])
        out_f[:, :, t] = o
    final_state = state

    # backward within-block pass: reset to the block entry state at each block's last token
    out_b = torch.zeros_like(out_f)
    bwd = zero
    for t in range(T - 1, -1, -1):
        lt = last[:, t][:, None, None, None]
        bwd = torch.where(lt, block_entry[:, :, t], bwd)
        bwd, o = step(bwd, q[:, :, t], k[:, :, t], v[:, :, t], g_[:, :, t], beta_[:, :, t])
        out_b[:, :, t] = o

    out = 0.5 * (out_f + out_b)
    out = out.transpose(1, 2).contiguous().to(dtype0)            # [B, T, H, Dv]
    return out, (final_state if output_final_state else None)


# ----------------------------------------------------------------------------------------
class HFBlockDiffusionHybrid(HFBlockDiffusion):
    """Block-diffusion trainer for hybrid linear-attention decoders (Qwen3.5). See module docstring."""

    def __init__(
        self,
        *args,
        linear_block_mode: str = "causal",
        attn_impl: str = "flex_attention",
        require_kernels: bool | None = None,
        **kwargs,
    ):
        assert linear_block_mode in ("causal", "bidirectional"), linear_block_mode
        # Reuse the dense setup verbatim (model load, <|mask|> add + untied resize, flex-impl
        # injection, optional per-layer compile). Nothing dense-specific is changed.
        super().__init__(*args, **kwargs)

        self.linear_block_mode = linear_block_mode
        self.attn_impl = attn_impl
        self.layer_types = _layer_types_of(self.model.config)
        assert is_hybrid_config(self.model.config), (
            "HFBlockDiffusionHybrid expects a hybrid (linear+full) model; use HFBlockDiffusion for "
            "an all-full_attention model."
        )

        if not self.ar_only:
            # attn_impl controls only the FULL-attention layers: flex BlockMask (prod, GPU) or a
            # dense additive mask (CPU tests — FlexAttention has no CPU backward).
            self.model.config._attn_implementation = attn_impl
            self.tm.config._attn_implementation = attn_impl

        self._check_linear_kernels(require_kernels)
        if self.ar_only:
            # Pure-AR SFT ignores linear_block_mode (every layer runs plain causal), but packed AR
            # rows still need the per-document state reset in the linear layers.
            self._install_packed_scan()
        elif self.linear_block_mode == "bidirectional":
            # Real FLARE two-stream GDN (bidirectional-within-block via clean-boundary-state re-read),
            # replacing the old pure-torch `_block_bidirectional_gdr` approximation.
            self._install_two_stream_scan()
        else:
            # Sequence-packing support: make the linear layers honor document boundaries
            # (cu_seqlens / seq_idx). Inert for the single-turn path (no metadata => delegates to the
            # original kernel), so this does not change unpacked behavior.
            self._install_packed_scan()

    # ---- kernel availability ----
    def _check_linear_kernels(self, require_kernels):
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )
        fla_ok = is_flash_linear_attention_available()
        cc1d_ok = is_causal_conv1d_available()
        # `fla` is essential (the gated-delta recurrence). `causal_conv1d` is NOT required: the
        # two-stream path uses FLARE's Triton `block_train_conv`, and the causal/packed paths fall
        # back to `_torch_causal_conv1d_varlen`. So gate the hard error on `fla` only.
        have = fla_ok
        # Persisted for reporting / tests. NOTE ``_install_packed_scan`` does NOT trust this flag —
        # it reads the per-layer binding via ``_is_fused_gdr`` (see there), because the probes and the
        # bindings can disagree.
        self._fused_linear_kernels = bool(fla_ok and cc1d_ok)
        if require_kernels is None:
            # Default: demand the fused kernels for real (CUDA) training; tolerate the slow torch
            # fallback on CPU (shape/gradient tests only).
            require_kernels = torch.cuda.is_available()
        if not have:
            msg = (
                "Qwen3.5 linear-attention layers need `fla` (flash-linear-attention) + "
                "`causal_conv1d`, which are not installed. Install per "
                "https://github.com/fla-org/flash-linear-attention and "
                "https://github.com/Dao-AILab/causal-conv1d ."
            )
            if require_kernels:
                raise ImportError(msg + " (set require_kernels=False to allow the slow torch fallback)")
            warnings.warn(msg + " Falling back to transformers' torch reference kernels (SLOW; CPU "
                          "shape/gradient testing only, NOT for training).")

    # ---- bidirectional scan install ----
    def _install_bidirectional_scan(self):
        """Swap each linear layer's ``chunk_gated_delta_rule`` for a dispatcher: when the layer's
        ``_bd_block_ids`` is set (region-A call) use ``_block_bidirectional_gdr``; when ``None``
        (region-B / causal call) delegate to the original (fused or torch) causal kernel."""
        for layer in self.tm.layers:
            if getattr(layer, "layer_type", None) != "linear_attention":
                continue
            la = layer.linear_attn
            la._bd_block_ids = None
            la._orig_chunk = la.chunk_gated_delta_rule

            def make(la):
                def scan(query, key, value, g, beta, **kw):
                    if la._bd_block_ids is None:
                        return la._orig_chunk(query, key, value, g=g, beta=beta, **kw)
                    return _block_bidirectional_gdr(query, key, value, g, beta,
                                                    block_ids=la._bd_block_ids, **kw)
                return scan

            la.chunk_gated_delta_rule = make(la)

    # ---- FLARE two-stream scan install (the real bidirectional method) ----
    def _install_two_stream_scan(self):
        """Rebind each linear layer's recurrence + conv to FLARE's two-stream block-diffusion kernels.
        The decoder stack feeds ONE ``[x0 ; xt]`` sequence per layer call, so the kernels split at
        ``T//2`` internally. ``causal_mode_clean`` = token-causal x0 when an AR aux loss is on (matches
        the mask's ``causal_x0``); ``causal_mode_noisy`` = 0 (bidirectional xt) unless
        ``within_block_causal``. Also installs a per-forward ``cu_seqlens`` holder for packing."""
        self._flare_cu = None   # set per-forward by forward_flare (packed) / None (unpacked)
        # Number of noisy views sharing ONE clean stream in the layer input [x0 ; xt_1 ; … ; xt_V]
        # (set per-forward by forward_flare; 1 == plain [x0 ; xt]).
        self._flare_nviews = 1
        if not hasattr(self, "flare_share_x0"):
            self.flare_share_x0 = True
        # The vendored chunk_refine recurrence has an inner torch checkpoint; nested under our
        # per-layer checkpointing it recomputes the noisy pass an extra time for ~no memory benefit,
        # so disable it whenever outer checkpointing is on.
        if self.grad_checkpoint:
            from train.block_gated_delta_rule import _profiling as _bgdr_prof
            _bgdr_prof._DISABLE_CHECKPOINT = True
        cmc = 1 if self.ar_loss_weight > 0 else 0
        cmn = 1 if self.within_block_causal else 0
        nv = lambda s=self: getattr(s, "_flare_nviews", 1)
        n = 0
        for layer in self.tm.layers:
            if getattr(layer, "layer_type", None) != "linear_attention":
                continue
            la = layer.linear_attn
            la._orig_chunk = la.chunk_gated_delta_rule
            # torch._dynamo.disable: when the GDN layer is torch.compiled (compile_glue), dynamo
            # graph-breaks cleanly around the kernel wrappers instead of tracing into the vendored
            # kernels' Python glue (which specialises on the data-dependent `cu_seqlens`). No-op
            # otherwise.
            la.chunk_gated_delta_rule = torch._dynamo.disable(_make_two_stream_gdr(
                la._orig_chunk, block_size=self.bd_size,
                causal_mode_clean=cmc, causal_mode_noisy=cmn, nviews_getter=nv))
            la.causal_conv1d_fn = torch._dynamo.disable(_make_two_stream_conv(
                self.bd_size, cu_getter=lambda s=self: s._flare_cu, nviews_getter=nv))
            n += 1
        # Fail loud: rebinding zero layers would silently fall back to the standard causal kernel.
        assert n > 0, (
            f"_install_two_stream_scan rebound 0 linear layers of {len(self.tm.layers)} "
            f"(layer_types={self.layer_types[:6]}...); the FLARE two-stream kernel is NOT active.")

    # ---- packed (sequence-packing) scan install ----
    def _install_packed_scan(self):
        """Install the document-isolation wrappers on every linear layer (CAUSAL mode only). Both
        wrappers are inert without packing metadata (single-turn path unchanged).

        gated-delta scan — ``_make_varlen_gdr``. Always installed, but in one of TWO modes:
          * fused kernel present (GPU): flatten to the ``[1, B*T, ...]`` varlen layout FLA requires and
            make ONE fused ``cu_seqlens`` call, which resets the state at each boundary natively;
          * torch reference (CPU): emulate the reset with one call per document (slow python loop).
        The flatten is mandatory in both modes — FLA rejects ``B > 1`` together with ``cu_seqlens`` —
        so this wrapper cannot simply be dropped on GPU; only its inner loop can.

        causal conv — the fused ``causal_conv1d_fn`` already resets on ``seq_idx`` natively, so it is
        left untouched; only the absent-kernel case gets the ``seq_idx``-aware torch fallback."""
        self._packed_scan_fused = []
        for layer in self.tm.layers:
            if getattr(layer, "layer_type", None) != "linear_attention":
                continue
            la = layer.linear_attn
            fused = _is_fused_gdr(la)
            self._packed_scan_fused.append(fused)
            la.chunk_gated_delta_rule = _make_varlen_gdr(la.chunk_gated_delta_rule, fused=fused)
            if la.causal_conv1d_fn is None:                 # torch fallback (CPU / no kernels)
                la.causal_conv1d_fn = _torch_causal_conv1d_varlen

    # ---- full-attention mask (dense additive variant for CPU) ----
    def _build_dense_additive_mask(self, seg, opos, rblk, x0_token_causal=False, segid=None,
                                   within_block_causal=False):
        """Dense ``[B, 1, T, T]`` additive mask mirroring ``HFBlockDiffusion._build_block_mask``'s
        rules, for the ``eager`` attention path on CPU (0 where allowed, -inf where blocked). Used
        only when ``attn_impl != 'flex_attention'`` — O(T^2), fine for tests, not long context. When
        ``segid`` is given (packed path) an extra ``segid[q] == segid[kv]`` term isolates documents
        (mirrors the dense ``_build_packed_block_mask``); ``segid < 0`` = pad, never matches."""
        B, T = seg.shape
        dev = seg.device
        # query index -> [B,T,1], key index -> [B,1,T]; broadcast to the [B,T,T] pair grid.
        sq, skv = seg[:, :, None], seg[:, None, :]
        oq, okv = opos[:, :, None], opos[:, None, :]
        bq, bkv = rblk[:, :, None], rblk[:, None, :]

        kv_ok = skv != _PAD
        to_shared = (skv == _SHARED) & (okv <= oq)
        xt_diag = (sq == _XT) & (skv == _XT) & (bq == bkv)
        if within_block_causal:
            xt_diag = xt_diag & (okv <= oq)
        xt_offset = (sq == _XT) & (skv == _X0) & (bkv < bq)
        if x0_token_causal:
            x0_causal = (sq == _X0) & (skv == _X0) & (okv <= oq)
        else:
            x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)
        allowed = kv_ok & (to_shared | xt_diag | xt_offset | x0_causal)
        if segid is not None:
            same_seg = segid[:, :, None] == segid[:, None, :]      # document isolation
            allowed = allowed & same_seg
        self_diag = torch.eye(T, dtype=torch.bool, device=dev)[None]
        allowed = allowed | self_diag                              # never NaN an empty row
        add = torch.zeros(B, T, T, device=dev, dtype=torch.float32)
        add = add.masked_fill(~allowed, torch.finfo(torch.float32).min)
        return add[:, None]                                        # [B, 1, T, T]

    def _full_mask(self, seg, opos, rblk, x0_token_causal, segid=None):
        wbc = self.within_block_causal
        if self.attn_impl == "flex_attention":
            if segid is not None:
                return self._build_packed_block_mask(seg, opos, rblk, segid,
                                                     x0_token_causal=x0_token_causal,
                                                     within_block_causal=wbc)
            return self._build_block_mask(seg, opos, rblk, x0_token_causal=x0_token_causal,
                                          within_block_causal=wbc)
        return self._build_dense_additive_mask(seg, opos, rblk, x0_token_causal=x0_token_causal,
                                               segid=segid, within_block_causal=wbc)

    # ---- decoder stack: per-layer-type routing ----
    def _run_decoder_stack_hybrid(self, hidden, position_ids, pos_emb, full_mask, valid2d, L, blk_A,
                                  gc_active=False, cuA=None, seqidxA=None, cuB=None, seqidxB=None):
        """Full layers consume ``full_mask`` (flex BlockMask or dense additive). Linear layers split
        the sequence into region-A ``[S|x_t]`` (``:L``) and region-B ``[x_0]`` (``L:``) and are run
        on each separately so the recurrence never mixes ``x_t`` into ``x_0`` (exactly what the full
        mask enforces for softmax layers).

        Packed path: ``cuA``/``seqidxA`` (region A, from ``segidA``) and ``cuB``/``seqidxB`` (region
        B, from ``segidB``) are threaded into the respective linear-layer calls as ``cu_seq_lens_q`` /
        ``seq_idx`` so the gated-delta recurrence and the causal conv reset at document boundaries.
        ``None`` (single-turn) => no reset, unpacked behavior.

        ``gc_active`` gradient-checkpoints each layer (mirrors the dense ``_run_decoder_stack``) so
        long context (e.g. 32k) fits — only the per-layer boundary activation is retained, the rest
        recomputed in backward. The per-layer bodies below are self-contained closures: the linear
        body re-sets ``_bd_block_ids`` on every (re)compute, so checkpointing is correct in BOTH
        ``causal`` (state never set) and ``bidirectional`` (state set inside the recomputed region,
        not read stale from outside) modes."""
        import contextlib
        from torch.utils.checkpoint import checkpoint
        cos, sin = pos_emb
        bidir = self.linear_block_mode == "bidirectional"

        def run_full(layer, h):
            out = layer(h, position_embeddings=pos_emb, attention_mask=full_mask,
                        position_ids=position_ids, past_key_values=None, use_cache=False)
            return out[0] if isinstance(out, tuple) else out

        # packed metadata for each region's linear-layer call (empty => unpacked, no reset)
        kwA = dict(cu_seq_lens_q=cuA, seq_idx=seqidxA) if cuA is not None else {}
        kwB = dict(cu_seq_lens_q=cuB, seq_idx=seqidxB) if cuB is not None else {}

        def run_linear(layer, h):
            A, Bx = h[:, :L], h[:, L:]
            peA = (cos[:, :L], sin[:, :L])
            peB = (cos[:, L:], sin[:, L:])
            if bidir:
                layer.linear_attn._bd_block_ids = blk_A         # region-A: block-bidirectional
            oA = layer(A, position_embeddings=peA, attention_mask=valid2d[:, :L],
                       position_ids=position_ids[:, :L], past_key_values=None, use_cache=False, **kwA)
            if bidir:
                layer.linear_attn._bd_block_ids = None          # region-B: causal
            oB = layer(Bx, position_embeddings=peB, attention_mask=valid2d[:, L:],
                       position_ids=position_ids[:, L:], past_key_values=None, use_cache=False, **kwB)
            oA = oA[0] if isinstance(oA, tuple) else oA
            oB = oB[0] if isinstance(oB, tuple) else oB
            return torch.cat([oA, oB], dim=1)

        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if (getattr(self, "activation_offload", False) and gc_active)
                   else contextlib.nullcontext())
        with offload:
            for layer in self.tm.layers:
                fn = run_full if layer.layer_type == "full_attention" else run_linear
                if gc_active:
                    hidden = checkpoint(lambda x, l=layer, f=fn: f(l, x), hidden, use_reentrant=False)
                else:
                    hidden = fn(layer, hidden)
        return hidden

    # ------------------------------------------------------------------
    # pure-AR (--ar_only) packed forward — document-isolated causal, both layer types
    # ------------------------------------------------------------------
    def _run_decoder_stack_ar(self, hidden, position_ids, pos_emb, full_mask, valid2d,
                              cu=None, seq_idx=None, gc_active=False):
        """Plain-causal hybrid stack over ONE ``[B, T]`` region (the pure-AR layout — no noised
        views, no A/B region split, no diffusion mask). Full layers consume ``full_mask`` (a flex
        ``BlockMask`` or a dense additive mask, block-diagonal causal); linear layers get the 2-D
        padding mask plus, when packing, ``cu_seq_lens_q`` / ``seq_idx`` so the gated-delta
        recurrence and the causal conv reset at every document boundary. ``gc_active``
        gradient-checkpoints each layer (the long-context memory lever, as in
        ``_run_decoder_stack_hybrid``)."""
        import contextlib
        from torch.utils.checkpoint import checkpoint

        kw = dict(cu_seq_lens_q=cu, seq_idx=seq_idx) if cu is not None else {}

        def run_full(layer, h):
            out = layer(h, position_embeddings=pos_emb, attention_mask=full_mask,
                        position_ids=position_ids, past_key_values=None, use_cache=False)
            return out[0] if isinstance(out, tuple) else out

        def run_linear(layer, h):
            out = layer(h, position_embeddings=pos_emb, attention_mask=valid2d,
                        position_ids=position_ids, past_key_values=None, use_cache=False, **kw)
            return out[0] if isinstance(out, tuple) else out

        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if (getattr(self, "activation_offload", False) and gc_active)
                   else contextlib.nullcontext())
        with offload:
            for layer in self.tm.layers:
                fn = run_full if layer.layer_type == "full_attention" else run_linear
                if gc_active:
                    hidden = checkpoint(lambda x, l=layer, f=fn: f(l, x), hidden, use_reentrant=False)
                else:
                    hidden = fn(layer, hidden)
        return hidden

    # ================= FLARE two-stream forward ([x0 ; xt], single + multi-turn) =================
    def _make_flare_views(self, input_ids, labels, answer_pos, eps: float = 1e-3):
        """Complementary masked views over ``answer_pos`` (response tokens) with ABSOLUTE-position
        blocks — works for single-turn (contiguous) AND multi-turn (scattered assistant spans).
        Returns ``noisy [2B,L]``, ``view_labels [2B,L]`` (clean id at masked-in-this-view response
        positions else -100), ``rate [2B,L]`` (per-token block mask rate, for 1/gamma weighting).

        ``full_mask`` mode: mask EVERY response token (gamma=1) and emit a SINGLE view — the
        complement would mask nothing, so complementary masking is moot. Returns ``[B,L]`` tensors;
        every response token is supervised in the one noisy view."""
        B, L = input_ids.shape
        dev = input_ids.device
        if getattr(self, "full_mask", False):
            noisy = torch.where(answer_pos, self.mask_id, input_ids)
            vlab = labels.clone()
            vlab[~answer_pos] = -100
            rate = torch.ones(B, L, device=dev)
            return noisy, vlab, rate
        bd = self.bd_size
        nblk = (L + bd - 1) // bd
        blk = (torch.arange(L, device=dev) // bd)[None, :].expand(B, -1)
        p_block = (1 - eps) * torch.rand(B, nblk, device=dev) + eps
        p_tok = torch.gather(p_block, 1, blk)
        mask_indices = (torch.rand(B, L, device=dev) < p_tok) & answer_pos

        def view(sel):
            apply = sel & answer_pos
            noisy = torch.where(apply, self.mask_id, input_ids)
            vlab = labels.clone()
            vlab[~apply] = -100
            return noisy, vlab

        na, la = view(mask_indices)
        nb, lb = view(~mask_indices)
        return torch.cat([na, nb], 0), torch.cat([la, lb], 0), p_tok.repeat(2, 1)

    def _flare_full_mask(self, L, device, doc_ids=None, nviews: int = 1):
        """FLARE ``[x0 ; xt_1 ; … ; xt_V]`` block-diffusion mask over ``(1+V)·L`` for the
        full-attention layers (flex). ``nviews=1`` is the plain ``[x0 ; xt]`` mask."""
        from torch.nn.attention.flex_attention import create_block_mask
        S = 1 + nviews
        ext_doc = doc_ids.repeat(1, S) if doc_ids is not None else None
        mod = functools.partial(_flare_block_diff_mask_mod, block_size=self.bd_size, n=L,
                                causal_x0=(self.ar_loss_weight > 0),
                                causal_xt=self.within_block_causal, doc_ids=ext_doc)
        Bm = doc_ids.shape[0] if doc_ids is not None else None
        return create_block_mask(mod, B=Bm, H=None, Q_LEN=S * L, KV_LEN=S * L,
                                 device=device, _compile=True)

    def _run_decoder_stack_flare(self, hidden, position_ids, pos_emb, full_mask, gc_active=False, cu=None):
        """Single-call stack over the ``[x0 ; xt]`` sequence. Full-attention layers consume
        ``full_mask``; linear layers use their rebound two-stream recurrence + conv (which split the
        sequence at ``T//2`` internally). One call per layer (not the region-A/B two-call split).
        ``cu`` (packed) is forwarded to the linear layers as ``cu_seq_lens_q`` (their rebound
        recurrence/conv wrappers consume it; full layers ignore it)."""
        import contextlib
        from torch.utils.checkpoint import checkpoint
        kw = dict(cu_seq_lens_q=cu) if cu is not None else {}

        def run(layer, h):
            out = layer(h, position_embeddings=pos_emb, attention_mask=full_mask,
                        position_ids=position_ids, past_key_values=None, use_cache=False, **kw)
            return out[0] if isinstance(out, tuple) else out

        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if (getattr(self, "activation_offload", False) and gc_active)
                   else contextlib.nullcontext())
        with offload:
            for layer in self.tm.layers:
                if gc_active:
                    hidden = checkpoint(lambda x, l=layer: run(l, x), hidden, use_reentrant=False)
                else:
                    hidden = run(layer, hidden)
        return hidden

    def forward_flare(self, input_ids, labels, attention_mask, seg_id=None, resp_block=None,
                      turn_id=None, return_logits: bool = False):
        """Two-stream block-diffusion forward on ``[x0_full ; xt_full]`` (FLARE method). Handles
        single-turn AND multi-turn uniformly: ``answer_pos`` (labels != -100) marks the response
        tokens to diffuse, contiguous or scattered across turns. Loss = diff CE on the noisy (xt)
        head over masked response tokens + AR CE on the clean (x0) head over response tokens."""
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        answer_pos = (labels != -100) & valid
        assert answer_pos.any(dim=1).all(), "every sample must have at least one response token"

        # Two complementary noisy views per row (view-major: rows [0,B) = view a, [B,2B) = view b).
        noisy, view_labels, rate = self._make_flare_views(input_ids, labels, answer_pos)  # [2B, L]
        # Layout: R rows x V noisy views, layer input per row = [x0 ; xt_1 ; … ; xt_V].
        #   share_x0 (default): R=B, V=2 — one clean stream per row shared by both views.
        #   otherwise:          R=2B, V=1 — clean duplicated per view.
        # Same objective either way; `flare_share_x0=False` exists for parity tests.
        if getattr(self, "full_mask", False):
            # single fully-masked view (no complement): R=B, V=1, [x0 ; xt] per row.
            R, V = B, 1
            ids_r, ans_r, seg_r, turn_r = input_ids, answer_pos, seg_id, turn_id
            xt_tok, vl, rt = noisy, view_labels[:, None], rate[:, None]
        elif getattr(self, "flare_share_x0", True):
            R, V = B, 2
            ids_r, ans_r, seg_r, turn_r = input_ids, answer_pos, seg_id, turn_id
            xt_tok = noisy.view(2, B, L).transpose(0, 1).reshape(B, 2 * L)     # [R, V*L]
            vl = view_labels.view(2, B, L).transpose(0, 1)                       # [R, V, L]
            rt = rate.view(2, B, L).transpose(0, 1)
        else:
            R, V = 2 * B, 1
            ids_r, ans_r = input_ids.repeat(2, 1), answer_pos.repeat(2, 1)
            seg_r = seg_id.repeat(2, 1) if seg_id is not None else None
            turn_r = turn_id.repeat(2, 1) if turn_id is not None else None
            xt_tok, vl, rt = noisy, view_labels[:, None], rate[:, None]
        S = 1 + V
        x0 = self.tm.embed_tokens(ids_r)                         # [R, L, D]
        xt = self.tm.embed_tokens(xt_tok)                        # [R, V*L, D]
        combined = torch.cat([x0, xt], dim=1)                    # [R, S*L, D]
        if seg_r is not None:                                    # packed: per-doc position reset
            pos1 = _positions_from_seg(seg_r)
        else:
            pos1 = torch.arange(L, device=device)[None, :].expand(R, -1)
        position_ids = pos1.repeat(1, S)                         # every stream shares positions
        pos_emb = self.tm.rotary_emb(combined, position_ids=position_ids)

        # packing: per-doc cu_seqlens (flat over the [R, L] clean half) isolates docs in the GDN
        # recurrence + conv; doc_ids isolate them in the full-attention mask. L is a bd_size multiple
        # (collator bucket) so doc starts are block-aligned (FLARE's packed-block_train requirement).
        doc_ids = seg_r
        cu = _cu_seqlens_from_seg(seg_r) if seg_r is not None else None
        self._flare_cu = cu          # consumed by the rebound two-stream conv wrapper
        self._flare_nviews = V       # consumed by the rebound two-stream recurrence + conv wrappers
        full_mask = self._flare_full_mask(L, device, doc_ids=doc_ids, nviews=V)

        gc = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        hidden = self._run_decoder_stack_flare(combined, position_ids, pos_emb, full_mask,
                                               gc_active=gc, cu=cu)
        hidden = self.tm.norm(hidden)
        D = hidden.shape[-1]
        h_x0 = hidden[:, :L, :]                                  # [R, L, D]
        h_xt = hidden[:, L:, :].reshape(R, V, L, D)              # [R, V, L, D]

        # diffusion loss: noisy head, token-shifted, over masked response tokens (1/gamma weighted)
        shift_hxt = h_xt[:, :, :L - 1, :]
        shift_vl = vl[:, :, 1:]
        seld = shift_vl != -100
        # intra-block position of each predicted token (label at shift index t = original pos t+1),
        # on the per-doc reset positions so blocks align with the collator's block-aligned docs.
        pos_sel = None
        if self.pos_decay_gamma > 0:
            pos_in_block = (pos1 % self.bd_size)[:, 1:]                  # [R, L-1]
            pos_in_block = pos_in_block[:, None, :].expand(R, V, L - 1)  # [R, V, L-1]
            pos_sel = pos_in_block[seld]
        diff_loss = self._diff_ce(shift_hxt[seld], shift_vl[seld], rt[:, :, 1:][seld], pos_sel=pos_sel)
        ntok = int(seld.sum())

        # AR loss: clean head, token-shifted next-token over response tokens (turn-bounded), ONCE
        # per clean stream (the legacy layout has 2B clean streams, the shared layout B).
        ar_w = self.ar_loss_weight
        if ar_w > 0:
            clean_lab = torch.where(ans_r, ids_r, torch.full_like(ids_r, -100))
            if turn_r is not None:  # don't predict across a turn boundary
                same_turn = (turn_r[:, :-1] == turn_r[:, 1:]) & (turn_r[:, 1:] >= 0)
            else:
                same_turn = torch.ones(R, L - 1, dtype=torch.bool, device=device)
            if seg_r is not None:    # packed: also don't predict across a document boundary
                same_turn = same_turn & (seg_r[:, :-1] == seg_r[:, 1:])
            shift_hx0 = h_x0[:, :L - 1, :]
            shift_cl = clean_lab[:, 1:]
            selc = (shift_cl != -100) & same_turn
            ar_loss = self._masked_ce(shift_hx0[selc], shift_cl[selc])
            loss = (diff_loss + ar_w * ar_loss) / (1.0 + ar_w)
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss

        logs = {"diff_loss": diff_loss.detach(), "ar_loss": ar_loss.detach(), "ntok": ntok}
        if return_logits:
            # [2B, L, vocab], view-major like `noisy` (rows [0,B) = view a, [B,2B) = view b)
            return self.model.lm_head(h_xt.transpose(0, 1).reshape(V * R, L, D)), loss, logs
        return loss, logs

    def forward_ar(self, input_ids, labels, attention_mask, seg_id=None,
                   return_logits: bool = False):
        """Pure-AR SFT for the hybrid decoder. ``seg_id is None`` -> the inherited native causal
        forward, unchanged. ``seg_id is not None`` (sequence packing) -> document-isolated causal:

          * FULL layers get a **block-diagonal causal** mask (causal AND ``seg_id[q] == seg_id[kv]``,
            padding never attended) — inherited ``_ar_doc_mask``, flex ``BlockMask`` on GPU / dense
            additive on CPU.
          * LINEAR layers are recurrent state machines, so a mask cannot isolate them; instead the
            gated-delta state and the causal-conv window are RESET at each document boundary via
            ``cu_seq_lens_q`` (FLA varlen) + ``seq_idx`` (causal_conv1d), derived from ``seg_id``.
            Without the reset, a later document in the row would read the earlier documents' state.

        Unlike ``forward_packed_hybrid`` there is only one region (no ``[S|x_t] | [x_0]`` split) and
        no block-diffusion structure, so the whole row is a single causal run per document."""
        if seg_id is None:
            return super().forward_ar(input_ids, labels, attention_mask, return_logits=return_logits)

        device = input_ids.device
        B, L = input_ids.shape
        segid = self.ar_segid(attention_mask, seg_id)                  # [B, L] int, -1 = pad
        full_mask, impl = self._ar_doc_mask(segid, dtype=self.tm.embed_tokens.weight.dtype)
        valid2d = attention_mask.bool()
        cu = _cu_seqlens_from_seg(segid)                                # gated-delta state reset
        seqidx = _seq_idx_from_seg(segid)                               # causal-conv window reset

        hidden = self.tm.embed_tokens(input_ids)
        position_ids = torch.arange(L, device=device)[None, :].expand(B, -1)
        pos_emb = self.tm.rotary_emb(hidden, position_ids=position_ids)
        gc = self.grad_checkpoint and self.training and L >= self.gc_min_len
        self._set_attn_impl(impl)   # sticky (must survive grad-checkpoint recompute) — see base class
        hidden = self._run_decoder_stack_ar(hidden, position_ids, pos_emb, full_mask, valid2d,
                                            cu=cu, seq_idx=seqidx, gc_active=gc)
        hidden = self.tm.norm(hidden)
        return self._ar_shift_loss(hidden, labels, return_logits)

    # ------------------------------------------------------------------
    # packed (sequence-packing) forward — CAUSAL linear mode, document-isolated
    # ------------------------------------------------------------------
    def forward_packed_hybrid(self, input_ids, labels, attention_mask, seg_id, resp_pos,
                              return_logits=False):
        """Packed hybrid forward: multiple ``[prefix | response]`` documents per row, document-
        isolated. Mirrors ``HFBlockDiffusion.forward_packed`` (same ``[prefix|x_t] | [x_0]`` double-
        batch layout, same ``seg``/``segid`` metadata and losses) but routes the decoder through the
        linear-aware hybrid stack:

          * FULL layers get the document-masked block-diffusion mask (``_build_packed_block_mask`` /
            dense-additive with ``segid`` isolation).
          * LINEAR layers are split into region-A ``[S|x_t]`` and region-B ``[x_0]`` (as in the
            single-turn hybrid path) and additionally receive per-region ``cu_seq_lens_q`` + ``seq_idx``
            derived from ``segidA`` / ``segidB``, so the gated-delta recurrence and the causal conv
            RESET at every document boundary — no cross-document leakage in either layer type.

        Only ``linear_block_mode='causal'`` is supported (asserted by the caller)."""
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        answer_pos = (labels != -100) & valid
        bd = self.bd_size
        NB = L // bd + 2  # max response-blocks per segment
        zero = torch.zeros_like(resp_pos)
        rblk_tok = torch.where(answer_pos, resp_pos.clamp(min=0) // bd, zero)
        block_key = torch.where(answer_pos, seg_id.clamp(min=0) * NB + rblk_tok, zero)

        # 1) complementary noised views (per-segment blocks) -> [2B, L]
        noisy, view_labels, rate = self._make_views_packed(input_ids, labels, answer_pos, block_key)
        clean = input_ids.repeat(2, 1)
        valid2 = valid.repeat(2, 1)
        seg2 = seg_id.repeat(2, 1)
        resp2 = resp_pos.repeat(2, 1)
        ans2 = answer_pos.repeat(2, 1)
        BB = 2 * B

        regionA = self.tm.embed_tokens(noisy)   # [BB, L, D]
        cleanA = self.tm.embed_tokens(clean)

        # 2) region B = clean x_0 copy of ALL response tokens (multi-span gather, index order)
        rcount = ans2.sum(dim=1)
        rpad = bucketed_clean_len(int(rcount.max().item()), self.bd_size, self.response_buckets)
        arng = torch.arange(L, device=device)
        order = torch.argsort((~ans2).int() * (L + 1) + arng[None, :], dim=1)
        src = order[:, :rpad]
        src_c = src.clamp(max=L - 1)
        b_valid = torch.arange(rpad, device=device)[None, :] < rcount[:, None]
        D = cleanA.size(-1)
        regionB = torch.gather(cleanA, 1, src_c.unsqueeze(-1).expand(-1, -1, D))

        combined = torch.cat([regionA, regionB], dim=1)  # [BB, L+rpad, D]
        arL = torch.arange(L, device=device)[None, :].expand(BB, -1)
        position_ids = torch.cat([arL, src_c], dim=1)

        # 3) per-token metadata (seg tag / original pos / resp-block / segment id) — as forward_packed
        segtagA = torch.where(valid2, torch.where(ans2, torch.full_like(arL, _XT),
                                                  torch.full_like(arL, _SHARED)),
                              torch.full_like(arL, _PAD))
        rblkA = torch.where(ans2, resp2.clamp(min=0) // bd, torch.zeros_like(resp2))
        segidA = torch.where(valid2, seg2, torch.full_like(seg2, -1))
        segtagB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        rblkB = torch.gather(rblkA, 1, src_c)
        segidB = torch.where(b_valid, torch.gather(segidA, 1, src_c), torch.full_like(src, -1))

        seg = torch.cat([segtagA, segtagB], dim=1).int()
        opos = torch.cat([arL, src_c], dim=1).int()
        rblk = torch.cat([rblkA, rblkB], dim=1).int()
        segid = torch.cat([segidA, segidB], dim=1).int()
        full_mask = self._full_mask(seg, opos, rblk, x0_token_causal=self.ar_loss_weight > 0,
                                    segid=segid)

        # 4) linear-layer inputs: 2D padding mask + per-region document-boundary metadata.
        valid2d = seg != _PAD                                      # [BB, T] bool
        cuA = _cu_seqlens_from_seg(segidA)                         # region A (=[S|x_t]) resets
        seqidxA = _seq_idx_from_seg(segidA)
        cuB = _cu_seqlens_from_seg(segidB)                         # region B (=[x_0]) resets
        seqidxB = _seq_idx_from_seg(segidB)

        # 5) run the hybrid decoder stack (gradient-checkpointed for long context — same gate as dense)
        gc = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        pos_emb = self.tm.rotary_emb(combined, position_ids=position_ids)
        hidden = self._run_decoder_stack_hybrid(combined, position_ids, pos_emb, full_mask,
                                                valid2d, L, None, gc_active=gc,
                                                cuA=cuA, seqidxA=seqidxA, cuB=cuB, seqidxB=seqidxB)
        hidden = self.tm.norm(hidden)

        # 6) diffusion loss over region A (x_t), token-shifted, masked-only (shift stays in-segment)
        shift_h = hidden[:, : L - 1, :]
        shift_labels = view_labels[:, 1:]
        sel = shift_labels != -100
        diff_loss = self._diff_ce(shift_h[sel], shift_labels[sel], rate[:, 1:][sel])
        ntok = int(sel.sum())

        # 6b) AR loss over region B (clean x_0): next-token within the SAME segment only
        ar_w = self.ar_loss_weight
        if ar_w > 0:
            hidB = hidden[:, L:, :]
            resp_ids = torch.gather(clean, 1, src_c)  # [BB, rpad] region-B token ids
            ar_sel = b_valid[:, 1:] & (segidB[:, :-1] == segidB[:, 1:])
            ar_loss = self._masked_ce(hidB[:, :-1, :][ar_sel], resp_ids[:, 1:][ar_sel])
            loss = (diff_loss + ar_w * ar_loss) / (1.0 + ar_w)
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss

        logs = {"diff_loss": diff_loss.detach(), "ar_loss": ar_loss.detach(), "ntok": ntok}
        if return_logits:
            return self.model.lm_head(hidden[:, :L, :]), loss, logs
        return loss, logs

    # ------------------------------------------------------------------
    def forward(self, input_ids, labels, attention_mask, seg_id=None, resp_pos=None,
                resp_block=None, turn_id=None, return_logits: bool = False):
        # AR-only short-circuits the diffusion machinery. seg_id MUST be routed through: with
        # sequence packing the AR forward needs the document-isolated causal mask + linear-layer
        # state reset, else packed documents silently cross-attend (and leak recurrent state).
        if self.ar_only:
            return self.forward_ar(input_ids, labels, attention_mask, seg_id=seg_id,
                                   return_logits=return_logits)
        # FLARE two-stream ([x0;xt]) handles single-turn AND multi-turn (scattered response spans)
        # uniformly. Packed two-stream (per-document cu_seqlens/state reset) is a follow-up.
        if self.linear_block_mode == "bidirectional":
            # single-turn, multi-turn, AND packed multi-turn (seg_id -> per-doc cu_seqlens isolation).
            return self.forward_flare(input_ids, labels, attention_mask, seg_id=seg_id,
                                      resp_block=resp_block, turn_id=turn_id,
                                      return_logits=return_logits)
        if resp_block is not None or turn_id is not None:
            raise NotImplementedError(
                "HFBlockDiffusionHybrid (causal mode) does not support the multi-turn path; use "
                "linear_block_mode='bidirectional' (FLARE two-stream) for multi-turn."
            )
        if seg_id is not None:
            # Sequence packing: multiple [prefix|response] documents per row. Supported for CAUSAL
            # linear mode (per-document cu_seqlens/seq_idx reset); bidirectional-packed and the
            # merged-views layout are documented follow-ups.
            if self.linear_block_mode != "causal":
                raise NotImplementedError(
                    "Packed (seg_id) forward is implemented only for linear_block_mode='causal'; "
                    "bidirectional block-diffusion + packing is a documented follow-up."
                )
            if self.merged_views:
                raise NotImplementedError(
                    "HFBlockDiffusionHybrid packing does not support the merged-views layout yet."
                )
            return self.forward_packed_hybrid(input_ids, labels, attention_mask, seg_id, resp_pos,
                                              return_logits=return_logits)
        device = input_ids.device
        B, L = input_ids.shape

        # response span = first..last labelled token; assert contiguous (single-turn). Mirrors
        # HFBlockDiffusion.forward.
        ans = labels != -100
        assert ans.any(dim=1).all(), "every sample must have at least one response token"
        s0 = torch.argmax(ans.int(), dim=1)
        last = L - 1 - torch.argmax(torch.flip(ans, [1]).int(), dim=1)
        arangeL = torch.arange(L, device=device)[None]
        assert (ans == ((arangeL >= s0[:, None]) & (arangeL <= last[:, None]))).all(), (
            "response (loss) tokens must form one contiguous span"
        )
        resp_len = last - s0 + 1

        # 1) two complementary noised views -> [2B, L]
        noisy, view_labels, rate = self._make_views(input_ids, labels, s0)
        clean = input_ids.repeat(2, 1)
        s0 = s0.repeat(2)
        resp_len = resp_len.repeat(2)
        valid = attention_mask.bool().repeat(2, 1)
        BB = 2 * B

        # 2) embeddings for region A (noised) and the clean copy (source of region B)
        regionA = self.tm.embed_tokens(noisy)
        cleanA = self.tm.embed_tokens(clean)

        # 3) region B = clean response span, right-padded to a bucketed length (stable T)
        rpad = bucketed_clean_len(int(resp_len.max().item()), self.bd_size, self.response_buckets)
        r_idx = torch.arange(rpad, device=device)[None, :]
        src = s0[:, None] + r_idx
        src_c = src.clamp(max=L - 1)
        b_valid = r_idx < resp_len[:, None]
        D = cleanA.size(-1)
        regionB = torch.gather(cleanA, 1, src_c.unsqueeze(-1).expand(-1, -1, D))

        combined = torch.cat([regionA, regionB], dim=1)            # [BB, L+rpad, D]
        posA = torch.arange(L, device=device)[None, :].expand(BB, -1)
        position_ids = torch.cat([posA, src_c], dim=1)

        # 4) per-token segment metadata for the full-attention mask (identical to the dense path)
        is_resp_A = (posA >= s0[:, None]) & valid
        segA = torch.where(
            valid,
            torch.where(is_resp_A, torch.full_like(posA, _XT), torch.full_like(posA, _SHARED)),
            torch.full_like(posA, _PAD),
        )
        rblkA = ((posA - s0[:, None]).clamp(min=0) // self.bd_size)
        segB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        rblkB = r_idx.expand(BB, -1) // self.bd_size

        seg = torch.cat([segA, segB], dim=1).int()
        opos = torch.cat([posA, src_c], dim=1).int()
        rblk = torch.cat([rblkA, rblkB], dim=1).int()
        full_mask = self._full_mask(seg, opos, rblk, x0_token_causal=self.ar_loss_weight > 0)

        # linear-layer inputs: a 2D padding mask (valid, non-pad tokens) + region-A block ids.
        valid2d = seg != _PAD                                      # [BB, T] bool
        # region-A block ids: prompt tokens -> their position (singleton blocks => causal);
        # x_t response tokens -> L + response-block (contiguous per block); pad -> -1. Only used by
        # the bidirectional scan; contiguity (not monotonicity) is what defines a block.
        blk_A = torch.where(is_resp_A, L + rblkA, posA)
        blk_A = torch.where(valid, blk_A, torch.full_like(posA, -1)).long()

        # 5) run the hybrid decoder stack (gradient-checkpointed for long context — same gate as dense)
        gc = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        pos_emb = self.tm.rotary_emb(combined, position_ids=position_ids)
        hidden = self._run_decoder_stack_hybrid(combined, position_ids, pos_emb, full_mask,
                                                valid2d, L, blk_A, gc_active=gc)
        hidden = self.tm.norm(hidden)

        # 6) losses — identical to HFBlockDiffusion.forward (region A diffusion CE + region B AR CE)
        shift_h = hidden[:, : L - 1, :]
        shift_labels = view_labels[:, 1:]
        sel = shift_labels != -100
        diff_loss = self._diff_ce(shift_h[sel], shift_labels[sel], rate[:, 1:][sel])
        ntok = int(sel.sum())

        ar_w = self.ar_loss_weight
        if ar_w > 0:
            hidB = hidden[:, L:, :]
            resp_ids = torch.gather(clean, 1, src_c)
            ar_sel = (r_idx + 1 < resp_len[:, None])[:, :-1]
            ar_loss = self._masked_ce(hidB[:, :-1, :][ar_sel], resp_ids[:, 1:][ar_sel])
            loss = (diff_loss + ar_w * ar_loss) / (1.0 + ar_w)
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss

        logs = {"diff_loss": diff_loss.detach(), "ar_loss": ar_loss.detach(), "ntok": ntok}
        if return_logits:
            return self.model.lm_head(hidden[:, :L, :]), loss, logs
        return loss, logs
