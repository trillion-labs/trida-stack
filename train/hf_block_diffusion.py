"""Text-only block-diffusion (+ optional AR aux loss) on a HuggingFace causal LM.

Shared-prefix construction (per complementary view, batch doubled to ``2B``)::

    sequence = [   S (prompt)   |  x_t  (response, noised)  |  x_0 (response, clean)  ]
                 region A (len L, prompt + noised response)      region B (len rpad)

Attention (custom flex ``BlockMask``, ``_build_block_mask``):
  * S (prompt): causal among itself, visible to everything after it (conditioning).
  * x_t response block k: bidirectional within its own block + previous response blocks in x_0.
  * x_0 response: block-causal over itself, OR token-causal when ``ar_loss_weight > 0`` (so the
    auxiliary next-token CE on x_0 is a valid AR objective; the x_t diffusion loss is unaffected).
Loss: token-shifted masked CE over region A (diffusion) + token-shifted CE over region B (AR),
combined as ``(diff + w*ar) / (1 + w)``.

Assumes each row has a SINGLE contiguous response (loss) span — true for single-turn SFT and
asserted at runtime. The collator right-pads and sets ``labels = -100`` outside the response.
"""

import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from transformers import AutoModelForCausalLM, AutoTokenizer

# segment tags used by the flex mask_mod
_PAD, _SHARED, _XT, _X0 = 0, 1, 2, 3


def bucketed_clean_len(needed, block_size, buckets):
    """Snap ``needed`` up to the smallest covering bucket (each rounded up to a multiple of
    ``block_size``); fall back to ``needed`` (block-aligned) if it exceeds every bucket or none are
    given. Keeps the flex-attention sequence length stable across batches so the kernel/model graph
    compile a handful of times then cache, and never truncates (result is always ``>= needed``)."""
    needed = math.ceil(max(needed, 1) / block_size) * block_size
    if not buckets:
        return needed
    for b in sorted(buckets):
        b = math.ceil(b / block_size) * block_size
        if b >= needed:
            return b
    return needed


class HFBlockDiffusion(nn.Module):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        bd_size: int = 32,
        mask_token: str = "<|mask|>",
        dtype: torch.dtype = torch.bfloat16,
        max_length: int = 16384,
        max_response_length: int = 4096,
        response_buckets=(512, 1024, 2048, 4096),
        ar_loss_weight: float = 0.0,
        grad_checkpoint: bool = False,
        gc_min_len: int = 0,
        activation_offload: bool = False,
        fused_ce: bool = False,
        compile_layers: bool = False,
        compile_mlp: bool = False,
        compile_glue: bool = False,
        merged_views: bool = False,
        ar_only: bool = False,
        loss_weighting: str = "weighted",
        pos_decay_gamma: float = 0.0,
        full_mask: bool = False,
        trust_remote_code: bool = False,
        attn_impl: str = "flex_attention",
        within_block_causal: bool = False,
        mask_pattern: str = "random",
    ):
        super().__init__()
        self.bd_size = bd_size
        self.loss_weighting = loss_weighting
        # full_mask: mask EVERY response token per block (gamma=1) and use a SINGLE noisy view
        # (complementary masking is moot when the whole block is masked). The noisy block is
        # predicted purely from earlier clean blocks — matches inference's fully-masked block.
        self.full_mask = full_mask
        # DFlash-style intra-block positional loss decay: a diffusion token at 0-indexed position p
        # within its bd_size block is weighted exp(-p / pos_decay_gamma) (p=0 -> 1.0, later -> decay).
        # Early-in-block errors invalidate later parallel-committed tokens, so up-weighting early
        # positions targets commit/acceptance length. 0 disables. Multiplies onto the base weighting.
        self.pos_decay_gamma = pos_decay_gamma
        # When True, x_t within-block attention is token-causal (append `okv <= oq` to every
        # `xt_diag`) instead of bidirectional; combined with x0_causal + linear_block_mode="causal"
        # this makes the model globally causal (=> vLLM-servable). Default False = original behavior.
        self.within_block_causal = within_block_causal
        # 'random' = complementary random-rate masks (FLARE objective); 'canvas' = decoder-canvas suffix masks
        # (draft-aligned fine-tune), single view, uniform weighting.
        self.mask_pattern = mask_pattern
        assert mask_pattern in ("random", "canvas"), mask_pattern
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.response_buckets = tuple(response_buckets) if response_buckets else None
        self.ar_loss_weight = ar_loss_weight
        self.grad_checkpoint = grad_checkpoint
        self.gc_min_len = gc_min_len
        self.activation_offload = activation_offload
        self.fused_ce = fused_ce
        self.merged_views = merged_views
        self.ar_only = ar_only
        # Only consulted by the PACKED AR path (`forward_ar_packed`) to pick between a flex
        # ``BlockMask`` (production / GPU — sparse, so 32k fits) and a dense ``[B,1,T,T]`` additive
        # mask (CPU tests — FlexAttention has no CPU backward). The diffusion paths hardcode flex
        # below, unchanged; ``HFBlockDiffusionHybrid`` overrides this attribute with its own
        # ``attn_impl`` after ``super().__init__``.
        self.attn_impl = attn_impl
        self._attn_impl_active = None    # set by _set_attn_impl on the first packed-AR forward

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trust_remote_code)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=dtype, trust_remote_code=trust_remote_code
        )

        if not ar_only:
            # Add the learned block-diffusion [MASK] token and grow the embedding / lm_head by one row.
            if mask_token not in self.tokenizer.get_vocab():
                self.tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
                self.model.resize_token_embeddings(len(self.tokenizer))
            self.mask_id = self.tokenizer.convert_tokens_to_ids(mask_token)
        else:
            # Pure AR SFT: no [MASK] token / no embedding resize -> the checkpoint keeps the base
            # vocab and is a plain HF causal LM (served by the AR server as-is).
            self.mask_id = (self.tokenizer.convert_tokens_to_ids(mask_token)
                            if mask_token in self.tokenizer.get_vocab() else None)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.pad_id = self.tokenizer.pad_token_id

        if not ar_only:
            # Inject flex attention so we can pass a block-diffusion BlockMask as the attention mask.
            self.model.config._attn_implementation = "flex_attention"
            self.tm.config._attn_implementation = "flex_attention"

            if compile_layers:
                layers = self.tm.layers
                for i in range(len(layers)):
                    layers[i] = torch.compile(layers[i])
            elif compile_mlp or compile_glue:
                for layer in self.tm.layers:
                    layer.mlp = torch.compile(layer.mlp, dynamic=False)
                    layer.input_layernorm = torch.compile(layer.input_layernorm, dynamic=False)
                    layer.post_attention_layernorm = torch.compile(layer.post_attention_layernorm,
                                                                   dynamic=False)
                    if compile_glue and getattr(layer, "layer_type", None) == "linear_attention":
                        layer.linear_attn = torch.compile(layer.linear_attn, dynamic=False)
        else:
            if grad_checkpoint:
                self.model.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- handles into the HF model ----
    @property
    def tm(self):  # the base decoder (LlamaModel/Qwen3Model): embed_tokens/rotary_emb/layers/norm
        return self.model.get_decoder()

    # ------------------------------------------------------------------
    # noising: two complementary views of the response, stacked -> [2B, L]
    # ------------------------------------------------------------------
    def _make_views(self, input_ids, labels, s0, eps: float = 1e-3):
        B, L = input_ids.shape
        device = input_ids.device
        answer_pos = labels != -100  # [B, L]

        rel = (torch.arange(L, device=device)[None, :] - s0[:, None]).clamp(min=0)
        rblk = rel // self.bd_size  # response-relative block id
        max_blocks = (L + self.bd_size - 1) // self.bd_size
        p_block = (1 - eps) * torch.rand(B, max_blocks, device=device) + eps
        p_tok = torch.gather(p_block, 1, rblk)  # [B, L]
        mask_indices = (torch.rand(B, L, device=device) < p_tok) & answer_pos

        def view(selected):
            apply = selected & answer_pos
            noisy = torch.where(apply, self.mask_id, input_ids)
            vlabels = labels.clone()
            vlabels[~apply] = -100
            return noisy, vlabels

        noisy_a, lab_a = view(mask_indices)
        noisy_b, lab_b = view(~mask_indices)
        # rate: the per-block mask rate of each token (both views share it) -> [2B, L], for 1/gamma
        rate = p_tok.repeat(2, 1)
        return torch.cat([noisy_a, noisy_b], 0), torch.cat([lab_a, lab_b], 0), rate

    # ------------------------------------------------------------------
    # flex BlockMask for the [S | x_t | x_0] layout
    # ------------------------------------------------------------------
    @staticmethod
    def _build_block_mask(seg, opos, rblk, x0_token_causal=False, within_block_causal=False):
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]

            kv_ok = skv != _PAD
            to_shared = (skv == _SHARED) & (okv <= oq)  # attend prompt causally
            # own block: bidirectional (default) or token-causal (`within_block_causal`)
            xt_diag = (sq == _XT) & (skv == _XT) & (bq == bkv)
            if within_block_causal:
                xt_diag = xt_diag & (okv <= oq)
            xt_offset = (sq == _XT) & (skv == _X0) & (bkv < bq)  # previous clean blocks
            if x0_token_causal:
                x0_causal = (sq == _X0) & (skv == _X0) & (okv <= oq)  # AR: token-causal
            else:
                x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)  # block-causal
            allowed = kv_ok & (to_shared | xt_diag | xt_offset | x0_causal)
            # Guarantee every query attends at least itself so pad/edge rows never softmax over an
            # empty key set (which would NaN the row).
            return allowed | (q_idx == kv_idx)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device, _compile=True)

    # ------------------------------------------------------------------
    # packed path: several [prefix | response] segments per row, segment-isolated (doc masking)
    # ------------------------------------------------------------------
    def _gen_mask_indices(self, answer_pos, block_key, eps: float = 1e-3):
        """Per-token keep/mask draw: one Bernoulli rate per ``block_key`` (unique per
        (segment, response-block)) so every packed block is noised independently."""
        B, L = answer_pos.shape
        device = answer_pos.device
        nkey = int(block_key.max().item()) + 1 if block_key.numel() else 1
        p_key = (1 - eps) * torch.rand(B, nkey, device=device) + eps
        p_tok = torch.gather(p_key, 1, block_key.clamp(min=0))
        return (torch.rand(B, L, device=device) < p_tok) & answer_pos, p_tok

    def _make_views_packed(self, input_ids, labels, answer_pos, block_key,
                           mask_indices=None, p_tok=None, eps: float = 1e-3):
        """Two complementary views. ``mask_indices`` (view-1 masked set) and ``p_tok`` (per-token
        block rate) may be supplied so callers share one draw across code paths (e.g. the
        merged-views equivalence test); pass them together to keep both paths bit-identical."""
        if mask_indices is None:
            mask_indices, p_tok = self._gen_mask_indices(answer_pos, block_key, eps)
        elif p_tok is None:
            _, p_tok = self._gen_mask_indices(answer_pos, block_key, eps)

        def view(selected):
            apply = selected & answer_pos
            noisy = torch.where(apply, self.mask_id, input_ids)
            vlabels = labels.clone()
            vlabels[~apply] = -100
            return noisy, vlabels

        na, la = view(mask_indices)
        nb, lb = view(~mask_indices)
        rate = p_tok.repeat(2, 1)  # [2B, L] per-token block rate (shared by both views)
        return torch.cat([na, nb], 0), torch.cat([la, lb], 0), rate

    @staticmethod
    def _build_packed_block_mask(seg, opos, rblk, segid, x0_token_causal=False,
                                 within_block_causal=False):
        """``_build_block_mask`` rules + segment isolation (``same_seg`` block-diagonal across packed
        segments = document masking) + a self-diagonal so pad/empty rows never NaN. ``segid < 0`` =
        padding (never matches a real query's segid)."""
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]
            same_seg = segid[b, q_idx] == segid[b, kv_idx]
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
            rule = to_shared | xt_diag | xt_offset | x0_causal
            return (kv_ok & same_seg & rule) | (q_idx == kv_idx)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device, _compile=True)

    @staticmethod
    def _build_merged_block_mask(seg, opos, rblk, segid, view, x0_token_causal=False,
                                 within_block_causal=False):
        """``_build_packed_block_mask`` + VIEW isolation. In the merged layout the two
        complementary x_t views live in ONE sequence (not stacked in the batch), so the only
        extra rule vs the packed mask is that x_t↔x_t block attention must also be same-view
        (``vq == vkv``) — otherwise a masked token could peek at its complement. Shared regions
        (prompt ``_SHARED``, clean ``_X0``) carry ``view = 0`` and are seen by both views via the
        view-agnostic ``to_shared`` / ``xt_offset`` / ``x0_causal`` rules."""
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            sq, skv = seg[b, q_idx], seg[b, kv_idx]
            oq, okv = opos[b, q_idx], opos[b, kv_idx]
            bq, bkv = rblk[b, q_idx], rblk[b, kv_idx]
            vq, vkv = view[b, q_idx], view[b, kv_idx]
            same_seg = segid[b, q_idx] == segid[b, kv_idx]
            kv_ok = skv != _PAD
            to_shared = (skv == _SHARED) & (okv <= oq)
            xt_diag = (sq == _XT) & (skv == _XT) & (bq == bkv) & (vq == vkv)  # + same view
            if within_block_causal:
                xt_diag = xt_diag & (okv <= oq)
            xt_offset = (sq == _XT) & (skv == _X0) & (bkv < bq)
            if x0_token_causal:
                x0_causal = (sq == _X0) & (skv == _X0) & (okv <= oq)
            else:
                x0_causal = (sq == _X0) & (skv == _X0) & (bkv <= bq)
            rule = to_shared | xt_diag | xt_offset | x0_causal
            return (kv_ok & same_seg & rule) | (q_idx == kv_idx)

        B, T = seg.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=seg.device, _compile=True)

    # ------------------------------------------------------------------
    def _masked_ce(self, hidden_sel, target_sel, weight=None, chunk: int = 4096):
        """Cross-entropy over the selected-token hidden states, projected through ``lm_head``.

        ``weight`` (optional, ``[N]``): per-token loss weights. When given the return is the WEIGHTED
        SUM ``Σ ceᵢ·wᵢ`` (the caller pre-normalizes ``w`` to sum 1, so this is a weighted mean);
        when ``None`` it is the plain mean ``mean(ce)``.

        Two backends (both avoid the full ``[N, vocab]`` fp32 logit tensor that OOMs at long context
        / large vocab — e.g. 32k packing, 150k-vocab Qwen3):
        * ``fused_ce`` (Liger ``fused_linear_cross_entropy``, ``reduction='none'``): fuses the
          ``lm_head`` matmul + per-token CE and never materializes logits (tiles internally).
        * else: project + per-token CE in **checkpointed chunks**, bounding peak logit memory to
          ``[chunk, vocab]``. The per-token CE vector is then reduced (weighted-sum or mean)."""
        n = int(hidden_sel.shape[0])
        if n == 0:
            return hidden_sel.sum() * 0.0  # zero, still attached to the graph

        if self.fused_ce:
            from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy
            w = self.model.lm_head.weight
            b = getattr(self.model.lm_head, "bias", None)
            # Under FSDP2 the (root) lm_head weight is a DTensor; gather the full [vocab, D] tensor
            # (its backward reduce-scatters the grad to the shards). target_sel is pre-filtered
            # (no -100), so ignore_index is moot. reduction='none' -> per-token CE we reduce here.
            if hasattr(w, "full_tensor"):
                w = w.full_tensor()
            if b is not None and hasattr(b, "full_tensor"):
                b = b.full_tensor()
            ce = liger_fused_linear_cross_entropy(
                hidden_sel, w, target_sel, bias=b, ignore_index=-100, reduction="none")
            return ce.mean() if weight is None else (ce * weight).sum()

        ckpt = self.training and torch.is_grad_enabled() and hidden_sel.requires_grad

        def ce_none(h, t):
            return F.cross_entropy(self.model.lm_head(h).float(), t, reduction="none")

        parts = []
        for i in range(0, n, chunk):
            h, t = hidden_sel[i:i + chunk], target_sel[i:i + chunk]
            c = checkpoint(ce_none, h, t, use_reentrant=False) if ckpt else ce_none(h, t)
            parts.append(c)
        ce = torch.cat(parts)  # [N] per-token CE
        return ce.mean() if weight is None else (ce * weight).sum()

    # ------------------------------------------------------------------
    def _diff_ce(self, hidden_sel, target_sel, rate_sel, pos_sel=None):
        """Diffusion CE over masked tokens with the configured weighting. ``loss_weighting=
        'weighted'`` applies the MDLM/WeDLM 1/gamma ELBO weight (``gamma`` = the per-block mask rate
        of each token), normalized to sum 1 across the masked tokens; ``'uniform'`` = plain mean CE.
        ``rate_sel`` is the per-token gamma aligned with ``hidden_sel``/``target_sel`` (``[N]``).
        ``pos_sel`` (optional, ``[N]``): 0-indexed intra-block position of each token; when
        ``pos_decay_gamma > 0`` multiplies an ``exp(-pos/gamma)`` DFlash decay onto the weight."""
        w = None
        if self.loss_weighting == "weighted" and rate_sel is not None and rate_sel.numel():
            w = 1.0 / (rate_sel.float() + 1e-8)
        if self.pos_decay_gamma > 0 and pos_sel is not None and pos_sel.numel():
            pw = torch.exp(-pos_sel.float() / self.pos_decay_gamma)
            w = pw if w is None else w * pw
        if w is not None:
            w = w / w.sum()
            return self._masked_ce(hidden_sel, target_sel, weight=w)
        return self._masked_ce(hidden_sel, target_sel)

    def forward_packed(self, input_ids, labels, attention_mask, seg_id, resp_pos,
                       return_logits=False, _mask_indices=None, _p_tok=None):
        """Packed-row forward (double-batch layout ``[prefix|x_t] | [x_0]``, replicated to 2B).
        Generalizes ``forward`` to MULTIPLE ``[prefix | response]`` segments per row: segments never
        cross-attend (``seg_id`` block-diagonal), each keeps its own block-diffusion structure, and
        the token-shift stays within a segment (segment layout is ``[prefix | response]``). Positions
        are continuous per row — RoPE is relative and attention is intra-segment, so this equals the
        unpacked result. ``resp_pos`` is the response-relative index (``-1`` off-response); pad tokens
        have ``seg_id = -1``."""
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
        noisy, view_labels, rate = self._make_views_packed(input_ids, labels, answer_pos, block_key,
                                                           mask_indices=_mask_indices, p_tok=_p_tok)
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

        # 3) per-token metadata (seg tag / original pos / resp-block / segment id)
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
        block_mask = self._build_packed_block_mask(seg, opos, rblk, segid,
                                                   x0_token_causal=self.ar_loss_weight > 0,
                                                   within_block_causal=self.within_block_causal)

        # 4) decoder stack
        pos_emb = self.tm.rotary_emb(combined, position_ids=position_ids)
        gc = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        hidden = self._run_decoder_stack(combined, position_ids, pos_emb, block_mask, gc)
        hidden = self.tm.norm(hidden)

        # 5) diffusion loss over region A (x_t), token-shifted, masked-only (shift stays in-segment)
        shift_h = hidden[:, : L - 1, :]
        shift_labels = view_labels[:, 1:]
        sel = shift_labels != -100
        diff_loss = self._diff_ce(shift_h[sel], shift_labels[sel], rate[:, 1:][sel])
        ntok = int(sel.sum())

        # 5b) AR loss over region B (clean x_0): next-token within the SAME segment only
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

    def forward_packed_merged(self, input_ids, labels, attention_mask, seg_id, resp_pos,
                              return_logits=False, _mask_indices=None, _p_tok=None):
        """Merged-views packed forward — mathematically identical to ``forward_packed`` but the
        shared prompt is stored ONCE instead of duplicated across the two complementary views.

        Layout (batch ``B``, not ``2B``):  ``[ regionA | xt2 | x0 ]``
          * ``regionA`` = view-1 noised, FULL row (``L``). The prompt lives here, once.
          * ``xt2``     = view-2's response tokens only, gathered (``rpad``).
          * ``x0``      = clean response copy, gathered (``rpad``) — shared by both views.

        View isolation (``_build_merged_block_mask``) keeps view-1 and view-2 x_t from cross-
        attending. Each response token is masked in exactly one view and supervised exactly once:
          * view-1 tokens  -> ``regionA`` shift (as in ``forward_packed``);
          * view-2 tokens, EXCEPT the per-segment first response token -> ``xt2`` shift;
          * view-2's per-segment FIRST response token -> folded into ``regionA`` (its shift
            predecessor is the segment's last prefix token, whose hidden is view-independent,
            so it is bit-identical to what ``forward_packed`` computes in its 2nd batch half).
        ``ntok`` and both losses match ``forward_packed`` to floating point (verified by
        ``tests/test_merged_views.py``)."""
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        answer_pos = (labels != -100) & valid
        bd = self.bd_size
        NB = L // bd + 2
        zero = torch.zeros_like(resp_pos)
        rblk_tok = torch.where(answer_pos, resp_pos.clamp(min=0) // bd, zero)
        block_key = torch.where(answer_pos, seg_id.clamp(min=0) * NB + rblk_tok, zero)

        # complementary masks (view-1 = mask_indices, view-2 = its complement over response)
        if _mask_indices is None:
            mask_indices, p_tok = self._gen_mask_indices(answer_pos, block_key)
        else:
            mask_indices = _mask_indices
            p_tok = _p_tok if _p_tok is not None else self._gen_mask_indices(answer_pos, block_key)[1]
        m1 = mask_indices & answer_pos
        m2 = (~mask_indices) & answer_pos
        noisy1 = torch.where(m1, self.mask_id, input_ids)
        noisy2 = torch.where(m2, self.mask_id, input_ids)
        lab1 = labels.clone(); lab1[~m1] = -100
        lab2 = labels.clone(); lab2[~m2] = -100

        regionA = self.tm.embed_tokens(noisy1)  # [B, L, D] — prompt (once) + view-1 x_t

        # response gather (position order), like region B in forward_packed
        rcount = answer_pos.sum(dim=1)
        rpad = bucketed_clean_len(int(rcount.max().item()), self.bd_size, self.response_buckets)
        arL = torch.arange(L, device=device)[None, :].expand(B, -1)
        order = torch.argsort((~answer_pos).int() * (L + 1) + torch.arange(L, device=device)[None, :], dim=1)
        src = order[:, :rpad]
        src_c = src.clamp(max=L - 1)
        b_valid = torch.arange(rpad, device=device)[None, :] < rcount[:, None]  # [B, rpad]

        xt2 = self.tm.embed_tokens(torch.gather(noisy2, 1, src_c))       # [B, rpad, D] view-2 resp
        x0_ids = torch.gather(input_ids, 1, src_c)
        x0 = self.tm.embed_tokens(x0_ids)                                # [B, rpad, D] clean resp

        combined = torch.cat([regionA, xt2, x0], dim=1)                  # [B, L + 2*rpad, D]
        position_ids = torch.cat([arL, src_c, src_c], dim=1)

        # per-token metadata (seg tag / original pos / resp-block / segment id / view)
        segtagA = torch.where(valid, torch.where(answer_pos, torch.full_like(arL, _XT),
                                                 torch.full_like(arL, _SHARED)),
                              torch.full_like(arL, _PAD))
        viewA = torch.where(answer_pos, torch.ones_like(arL), torch.zeros_like(arL))  # resp=view1
        segidA = torch.where(valid, seg_id, torch.full_like(seg_id, -1))
        rblk_g = torch.gather(rblk_tok, 1, src_c)
        segid_g = torch.gather(seg_id, 1, src_c)
        segtag_x2 = torch.where(b_valid, torch.full_like(src, _XT), torch.full_like(src, _PAD))
        view_x2 = torch.where(b_valid, torch.full_like(src, 2), torch.zeros_like(src))  # view2
        segid_x2 = torch.where(b_valid, segid_g, torch.full_like(src, -1))
        segtagB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        segidB = torch.where(b_valid, segid_g, torch.full_like(src, -1))

        seg = torch.cat([segtagA, segtag_x2, segtagB], dim=1).int()
        opos = torch.cat([arL, src_c, src_c], dim=1).int()
        rblk = torch.cat([rblk_tok, rblk_g, rblk_g], dim=1).int()
        segid = torch.cat([segidA, segid_x2, segidB], dim=1).int()
        view = torch.cat([viewA, view_x2, torch.zeros_like(src)], dim=1).int()
        block_mask = self._build_merged_block_mask(seg, opos, rblk, segid, view,
                                                   x0_token_causal=self.ar_loss_weight > 0,
                                                   within_block_causal=self.within_block_causal)

        pos_emb = self.tm.rotary_emb(combined, position_ids=position_ids)
        gc = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        hidden = self._run_decoder_stack(combined, position_ids, pos_emb, block_mask, gc)
        hidden = self.tm.norm(hidden)

        hA = hidden[:, :L, :]
        hX = hidden[:, L:L + rpad, :]
        hB = hidden[:, L + rpad:, :]

        # 5) diffusion loss.
        # regionA (view-1) shift + view-2's per-segment FIRST response token (predecessor is the
        # segment's last prefix token, in regionA). resp_start marks each segment's first response
        # token; for those masked in view-2, override regionA's label so its shift supervises them.
        resp_start = answer_pos & ~F.pad(answer_pos[:, :-1], (1, 0))
        A_labels = lab1.clone()
        override = resp_start & m2
        A_labels[override] = labels[override]
        shift_hA = hA[:, :L - 1, :]
        shift_lA = A_labels[:, 1:]
        selA = shift_lA != -100

        # xt2 (view-2) shift supervises response tokens AFTER each segment's first; the same-segment
        # gate drops the cross-segment pair (and the first-of-segment token, handled above).
        xt2_lab = torch.where(b_valid, torch.gather(lab2, 1, src_c), torch.full_like(src, -100))
        shift_hX = hX[:, :-1, :]
        shift_lX = xt2_lab[:, 1:]
        sameseg_X = segid_x2[:, :-1] == segid_x2[:, 1:]
        selX = (shift_lX != -100) & sameseg_X

        h_sel = torch.cat([shift_hA[selA], shift_hX[selX]], dim=0)
        t_sel = torch.cat([shift_lA[selA], shift_lX[selX]], dim=0)
        # per-token block rate aligned with each selection (regionA uses physical positions; xt2 uses
        # the gathered response positions) — same order as h_sel, for the 1/gamma weighting.
        rate_A = p_tok[:, 1:]
        rate_X = torch.gather(p_tok, 1, src_c)[:, 1:]
        rate_sel = torch.cat([rate_A[selA], rate_X[selX]], dim=0)
        diff_loss = self._diff_ce(h_sel, t_sel, rate_sel)
        ntok = int(selA.sum().item() + selX.sum().item())

        # 5b) AR loss over the single clean x_0 copy (next-token within a segment).
        ar_w = self.ar_loss_weight
        if ar_w > 0:
            ar_sel = b_valid[:, 1:] & (segidB[:, :-1] == segidB[:, 1:])
            ar_loss = self._masked_ce(hB[:, :-1, :][ar_sel], x0_ids[:, 1:][ar_sel])
            loss = (diff_loss + ar_w * ar_loss) / (1.0 + ar_w)
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss

        logs = {"diff_loss": diff_loss.detach(), "ar_loss": ar_loss.detach(), "ntok": ntok}
        if return_logits:
            return self.model.lm_head(hA), loss, logs
        return loss, logs

    # ------------------------------------------------------------------
    # multi-turn path: supervise EVERY assistant turn of one conversation, causally ordered
    # ------------------------------------------------------------------
    def forward_multiturn(self, input_ids, labels, attention_mask, resp_block, turn_id,
                          return_logits=False):
        """Supervise all assistant turns in ONE conversation sequence.

        Generalizes ``forward`` to multiple (non-contiguous) response spans that are CAUSALLY
        ORDERED — unlike ``forward_packed``'s isolated segments. A later turn's noised (x_t) tokens
        attend every earlier turn's clean (x_0) copy plus the interleaved prompt (user/tool) tokens,
        so conditioning matches inference (earlier turns are already-generated context).

        This works by feeding a GLOBAL, conversation-ordered block id (``resp_block``: unique per
        (turn, bd-block), ``-1`` off-response) to the shared ``_build_block_mask``: ``xt_offset``
        (x_t attends x_0 with ``bkv < bq``) then spans all earlier turns' blocks, and ``xt_diag``
        (``bq == bkv``) stays within one turn's block. ``turn_id`` (assistant-turn ordinal, ``-1``
        off-response) only bounds the auxiliary AR next-token loss so it never crosses a turn.
        """
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()
        answer_pos = (labels != -100) & valid
        block_key = torch.where(answer_pos, resp_block.clamp(min=0), torch.zeros_like(resp_block))

        # 1) complementary noised views, independent noise per global response-block -> [2B, L]
        noisy, view_labels, rate = self._make_views_packed(input_ids, labels, answer_pos, block_key)
        clean = input_ids.repeat(2, 1)
        valid2 = valid.repeat(2, 1)
        ans2 = answer_pos.repeat(2, 1)
        rblk2 = torch.where(ans2, resp_block.repeat(2, 1).clamp(min=0), torch.zeros_like(ans2, dtype=torch.long))
        turn2 = turn_id.repeat(2, 1)
        BB = 2 * B

        regionA = self.tm.embed_tokens(noisy)
        cleanA = self.tm.embed_tokens(clean)

        # 2) region B = clean x_0 copy of ALL response tokens (multi-span gather, position order)
        rcount = ans2.sum(dim=1)
        rpad = bucketed_clean_len(int(rcount.max().item()), self.bd_size, self.response_buckets)
        arng = torch.arange(L, device=device)
        order = torch.argsort((~ans2).int() * (L + 1) + arng[None, :], dim=1)
        src = order[:, :rpad]
        src_c = src.clamp(max=L - 1)
        b_valid = torch.arange(rpad, device=device)[None, :] < rcount[:, None]
        D = cleanA.size(-1)
        regionB = torch.gather(cleanA, 1, src_c.unsqueeze(-1).expand(-1, -1, D))

        combined = torch.cat([regionA, regionB], dim=1)
        arL = torch.arange(L, device=device)[None, :].expand(BB, -1)
        position_ids = torch.cat([arL, src_c], dim=1)

        # 3) per-token metadata (seg tag / original pos / GLOBAL resp-block / turn id)
        segtagA = torch.where(valid2, torch.where(ans2, torch.full_like(arL, _XT),
                                                  torch.full_like(arL, _SHARED)),
                              torch.full_like(arL, _PAD))
        rblkA = rblk2
        turnA = turn2
        segtagB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        rblkB = torch.gather(rblkA, 1, src_c)
        turnB = torch.gather(turnA, 1, src_c)

        seg = torch.cat([segtagA, segtagB], dim=1).int()
        opos = torch.cat([arL, src_c], dim=1).int()
        rblk = torch.cat([rblkA, rblkB], dim=1).int()
        # no segment isolation: later turns SEE earlier turns (global block ordering does the causality)
        block_mask = self._build_block_mask(seg, opos, rblk, x0_token_causal=self.ar_loss_weight > 0,
                                            within_block_causal=self.within_block_causal)

        pos_emb = self.tm.rotary_emb(combined, position_ids=position_ids)
        gc = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        hidden = self._run_decoder_stack(combined, position_ids, pos_emb, block_mask, gc)
        hidden = self.tm.norm(hidden)

        # 4) diffusion loss over region A (x_t), token-shifted, masked-only. The shift naturally stays
        # within a turn: the token after a turn's last response token is a prompt token (label -100).
        shift_h = hidden[:, : L - 1, :]
        shift_labels = view_labels[:, 1:]
        sel = shift_labels != -100
        diff_loss = self._diff_ce(shift_h[sel], shift_labels[sel], rate[:, 1:][sel])
        ntok = int(sel.sum())

        # 4b) AR loss over region B (clean x_0): next-token within the SAME turn only
        ar_w = self.ar_loss_weight
        if ar_w > 0:
            hidB = hidden[:, L:, :]
            resp_ids = torch.gather(clean, 1, src_c)
            ar_sel = b_valid[:, 1:] & (turnB[:, :-1] == turnB[:, 1:]) & (turnB[:, 1:] >= 0)
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
    def _run_decoder_stack(self, hidden, position_ids, position_embeddings, block_mask, gc_active):
        def call_layer(layer, h):
            out = layer(
                h,
                attention_mask=block_mask,
                position_ids=position_ids,
                position_embeddings=position_embeddings,
                past_key_values=None,
                use_cache=False,
            )
            return out[0] if isinstance(out, tuple) else out

        # Offload the checkpoint-boundary activations (the per-layer saved inputs, ~[BB,T,D] each)
        # to pinned CPU so long-context (e.g. 32k) fits: at 32k the 36 retained boundaries dominate
        # GPU memory. save_on_cpu moves saved-for-backward tensors to host RAM, staged back during
        # recompute. Costs CPU<->GPU transfer (slower) but is the key long-context memory lever.
        import contextlib
        offload = (torch.autograd.graph.save_on_cpu(pin_memory=True)
                   if (self.activation_offload and gc_active) else contextlib.nullcontext())
        with offload:
            for layer in self.tm.layers:
                if gc_active:
                    hidden = checkpoint(lambda x, l=layer: call_layer(l, x), hidden, use_reentrant=False)
                else:
                    hidden = call_layer(layer, hidden)
        return hidden

    # ------------------------------------------------------------------
    # pure-AR path (--ar_only), unpacked + packed (sequence packing, document-isolated)
    # ------------------------------------------------------------------
    @staticmethod
    def ar_segid(attention_mask, seg_id):
        """Per-token document id for a packed AR row: the collator's ``seg_id`` on real tokens,
        ``-1`` on padding (``attention_mask == 0``). ``-1`` never equals a real document's id, so the
        ``same_seg`` rule below both isolates documents AND blocks attention to padding."""
        return torch.where(attention_mask.bool(), seg_id, torch.full_like(seg_id, -1)).int()

    @staticmethod
    def _build_ar_doc_block_mask(segid):
        """Flex ``BlockMask`` for a packed AR row: **block-diagonal causal** — a token attends only
        earlier-or-equal positions OF ITS OWN DOCUMENT. Sparse (never materializes ``[B,T,T]``), which
        is what makes 32k packing viable; same construction style as ``_build_packed_block_mask``.
        The diagonal (``q_idx == kv_idx``) always passes (causal + same-document), so no query ever
        softmaxes over an empty key set — including padding rows (all ``segid = -1``)."""
        from torch.nn.attention.flex_attention import create_block_mask

        def mask_mod(b, h, q_idx, kv_idx):
            return (kv_idx <= q_idx) & (segid[b, q_idx] == segid[b, kv_idx])

        B, T = segid.shape
        return create_block_mask(mask_mod, B=B, H=None, Q_LEN=T, KV_LEN=T, device=segid.device,
                                 _compile=True)

    @staticmethod
    def _build_ar_doc_dense_mask(segid, dtype=torch.float32):
        """Dense ``[B, 1, T, T]`` additive twin of ``_build_ar_doc_block_mask`` (0 = allowed, -inf =
        blocked) for attention backends that take an ordinary 4-D mask (``sdpa``/``eager``). O(T^2)
        memory — CPU tests / short context only; production uses the flex path above. ``dtype`` must
        match the query dtype (``sdpa`` rejects a mismatched float mask)."""
        B, T = segid.shape
        ar = torch.arange(T, device=segid.device)
        causal = ar[:, None] >= ar[None, :]                          # [T, T]
        same_seg = segid[:, :, None] == segid[:, None, :]            # [B, T, T]
        allowed = causal[None] & same_seg
        add = torch.zeros(B, T, T, device=segid.device, dtype=dtype)
        return add.masked_fill(~allowed, torch.finfo(dtype).min)[:, None]

    def _ar_doc_mask(self, segid, dtype=torch.float32):
        """Return ``(mask, attn_impl_override)`` for the packed-AR document-isolated causal mask.
        ``attn_impl_override`` is the attention implementation the mask REQUIRES (``flex_attention``
        for a ``BlockMask``) or ``None`` to keep the model's native implementation (a dense additive
        mask is consumed correctly by both ``sdpa`` and ``eager``)."""
        if self.attn_impl == "flex_attention":
            return self._build_ar_doc_block_mask(segid), "flex_attention"
        return self._build_ar_doc_dense_mask(segid, dtype=dtype), None

    def _set_attn_impl(self, impl):
        """Force ``config._attn_implementation`` (both the CausalLM- and decoder-level configs) so a
        flex ``BlockMask`` reaches the flex kernel. ``impl=None`` is a no-op (a dense additive mask
        works with the native implementation).

        STICKY on purpose — the switch is NOT restored after the forward. A forward-scoped context
        manager is wrong here: gradient checkpointing RECOMPUTES each layer during ``backward()``,
        after the context has exited, and the recompute then dispatches to ``sdpa`` with a
        ``BlockMask`` -> ``TypeError: attn_mask must be Tensor, not BlockMask`` (observed on a GPU
        smoke). Packing is a per-run data-pipeline choice, so flipping once on the first packed
        forward is sufficient; the UNPACKED AR path is untouched as long as no packed forward
        preceded it (i.e. always, since a run uses one collator)."""
        if impl is None or getattr(self, "_attn_impl_active", None) == impl:
            return
        self.model.config._attn_implementation = impl
        self.tm.config._attn_implementation = impl
        self._attn_impl_active = impl

    def _ar_shift_loss(self, hidden, labels, return_logits: bool = False):
        """Ordinary causal-LM shift (``hidden[:, :-1]`` predicts ``labels[:, 1:]``) + masked CE.

        Cross-document safety under sequence packing: ``pack_rows_to_batch`` ASSERTS that every
        packed segment starts with a ``label == -100`` prefix token, so the token that follows a
        document's last response token is the NEXT document's masked prefix and is therefore never
        selected (``shift_labels != -100``). No pair in the selection straddles a document boundary,
        so no extra masking is needed here."""
        shift_h = hidden[:, :-1, :]
        shift_labels = labels[:, 1:]
        sel = shift_labels != -100
        loss = self._masked_ce(shift_h[sel], shift_labels[sel])
        logs = {"diff_loss": torch.zeros((), device=hidden.device),
                "ar_loss": loss.detach(), "ntok": int(sel.sum())}
        if return_logits:
            return self.model.lm_head(hidden), loss, logs
        return loss, logs

    def forward_ar_packed(self, input_ids, labels, attention_mask, seg_id,
                          return_logits: bool = False):
        """Packed pure-AR forward: several ``[prefix | response]`` documents per row, DOCUMENT
        ISOLATED. Identical objective to ``forward_ar`` — the only difference is the attention mask:
        the native full causal mask is replaced by a **block-diagonal causal** one (causal AND
        ``seg_id[q] == seg_id[kv]``), so packed documents never see each other and padding is never
        attended. Without this, ``--ar_only --pack`` would silently let documents in the same row
        cross-attend (data corruption).

        Positions stay continuous per row (``position_ids = arange(L)``, HF's default): RoPE is
        relative and attention is intra-document, so this equals the unpacked result — the same
        argument the dense packed diffusion path (``forward_packed``) relies on. HF's native
        gradient checkpointing (enabled in ``__init__``) still applies, since the whole stack runs
        through ``self.tm``."""
        dtype = self.tm.embed_tokens.weight.dtype
        mask, impl = self._ar_doc_mask(self.ar_segid(attention_mask, seg_id), dtype=dtype)
        self._set_attn_impl(impl)
        out = self.tm(input_ids=input_ids, attention_mask=mask, use_cache=False)
        return self._ar_shift_loss(out.last_hidden_state, labels, return_logits)

    def forward_ar(self, input_ids, labels, attention_mask, seg_id=None,
                   return_logits: bool = False):
        """Pure autoregressive SFT: standard causal-LM next-token loss on the supervised (response)
        tokens — NO diffusion masking / noised views. Runs HF's native causal forward (single [B,L]
        sequence, native gradient checkpointing) and projects only the selected tokens through
        ``lm_head`` via ``_masked_ce`` (no full [N, vocab] logit tensor). ``labels`` already carry
        -100 on prompt / non-supervised tokens (from the same multiturn collator), so the shift is
        the ordinary causal-LM shift and never predicts across a turn boundary into a prompt.

        ``seg_id`` (sequence packing) routes to ``forward_ar_packed``, which swaps the native causal
        mask for a document-isolated one. ``seg_id is None`` keeps this path bit-identical."""
        if seg_id is not None:
            return self.forward_ar_packed(input_ids, labels, attention_mask, seg_id,
                                          return_logits=return_logits)
        out = self.tm(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        return self._ar_shift_loss(out.last_hidden_state, labels, return_logits)

    # ------------------------------------------------------------------
    def forward(self, input_ids, labels, attention_mask, seg_id=None, resp_pos=None,
                resp_block=None, turn_id=None, return_logits: bool = False):
        # Pure AR SFT short-circuits the whole diffusion path (mask views / block masks unused).
        # seg_id MUST be routed through: with sequence packing the AR forward needs the
        # document-isolated causal mask, else packed documents silently cross-attend.
        if self.ar_only:
            return self.forward_ar(input_ids, labels, attention_mask, seg_id=seg_id,
                                   return_logits=return_logits)
        # Multi-turn rows (every assistant turn supervised, causally ordered) carry resp_block/turn_id.
        if resp_block is not None:
            return self.forward_multiturn(input_ids, labels, attention_mask, resp_block, turn_id,
                                          return_logits=return_logits)
        # Packed rows (multiple [prefix|response] segments) carry seg_id/resp_pos -> packed path.
        if seg_id is not None:
            fn = self.forward_packed_merged if self.merged_views else self.forward_packed
            return fn(input_ids, labels, attention_mask, seg_id, resp_pos,
                      return_logits=return_logits)
        device = input_ids.device
        B, L = input_ids.shape
        valid = attention_mask.bool()

        # response span = first..last labelled token; assert contiguous (single-turn)
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
        valid = valid.repeat(2, 1)
        BB = 2 * B

        # 2) embeddings for region A (noised) and the clean copy (source of region B)
        regionA = self.tm.embed_tokens(noisy)  # [BB, L, D]
        cleanA = self.tm.embed_tokens(clean)   # [BB, L, D]

        # 3) region B = clean response span, right-padded to a bucketed length (stable T)
        rpad = bucketed_clean_len(int(resp_len.max().item()), self.bd_size, self.response_buckets)
        r_idx = torch.arange(rpad, device=device)[None, :]
        src = s0[:, None] + r_idx
        src_c = src.clamp(max=L - 1)
        b_valid = r_idx < resp_len[:, None]
        D = cleanA.size(-1)
        regionB = torch.gather(cleanA, 1, src_c.unsqueeze(-1).expand(-1, -1, D))  # [BB, rpad, D]

        combined = torch.cat([regionA, regionB], dim=1)  # [BB, L+rpad, D]

        # positions: region A = physical index; region B = the response's original positions
        posA = torch.arange(L, device=device)[None, :].expand(BB, -1)
        position_ids = torch.cat([posA, src_c], dim=1)  # [BB, T]

        # 4) per-token segment metadata for the mask
        is_resp_A = (posA >= s0[:, None]) & valid
        segA = torch.where(
            valid,
            torch.where(is_resp_A, torch.full_like(posA, _XT), torch.full_like(posA, _SHARED)),
            torch.full_like(posA, _PAD),
        )
        oposA = posA
        rblkA = ((posA - s0[:, None]).clamp(min=0) // self.bd_size)

        segB = torch.where(b_valid, torch.full_like(src, _X0), torch.full_like(src, _PAD))
        oposB = src_c
        rblkB = r_idx.expand(BB, -1) // self.bd_size

        seg = torch.cat([segA, segB], dim=1).int()
        opos = torch.cat([oposA, oposB], dim=1).int()
        rblk = torch.cat([rblkA, rblkB], dim=1).int()
        block_mask = self._build_block_mask(seg, opos, rblk, x0_token_causal=self.ar_loss_weight > 0,
                                            within_block_causal=self.within_block_causal)

        # 5) run the decoder stack with the injected block mask
        position_embeddings = self.tm.rotary_emb(combined, position_ids=position_ids)
        gc_active = self.grad_checkpoint and self.training and combined.shape[1] >= self.gc_min_len
        hidden = self._run_decoder_stack(combined, position_ids, position_embeddings, block_mask, gc_active)
        hidden = self.tm.norm(hidden)

        # 6) diffusion loss: token-shifted masked CE over region A (x_t / response)
        shift_h = hidden[:, : L - 1, :]
        shift_labels = view_labels[:, 1:]
        sel = shift_labels != -100
        diff_loss = self._diff_ce(shift_h[sel], shift_labels[sel], rate[:, 1:][sel])
        ntok = int(sel.sum())

        # 6b) auxiliary AR loss on the clean x_0 stream (region B), token-shifted next-token CE
        ar_w = self.ar_loss_weight
        if ar_w > 0:
            hidB = hidden[:, L:, :]  # [BB, rpad, D]
            resp_ids = torch.gather(clean, 1, src_c)  # [BB, rpad] clean response tokens
            ar_sel = (r_idx + 1 < resp_len[:, None])[:, :-1]  # slot j valid AND j+1 in-response
            ar_loss = self._masked_ce(hidB[:, :-1, :][ar_sel], resp_ids[:, 1:][ar_sel])
            loss = (diff_loss + ar_w * ar_loss) / (1.0 + ar_w)
        else:
            ar_loss = torch.zeros((), device=device)
            loss = diff_loss

        logs = {"diff_loss": diff_loss.detach(), "ar_loss": ar_loss.detach(), "ntok": ntok}
        if return_logits:  # full [BB, L, V] logits — memory-heavy; for equivalence testing only
            return self.model.lm_head(hidden[:, :L, :]), loss, logs
        return loss, logs

    # ------------------------------------------------------------------
    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        with open(os.path.join(path, "block_diffusion.json"), "w") as f:
            json.dump(
                {
                    "bd_size": self.bd_size,
                    "mask_id": self.mask_id,
                    "ar_loss_weight": self.ar_loss_weight,
                    "max_length": self.max_length,
                    "max_response_length": self.max_response_length,
                    "response_buckets": list(self.response_buckets) if self.response_buckets else None,
                },
                f,
                indent=2,
            )
