# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# This file is derived from vLLM's in-tree Qwen3.5 model implementation
# (model_executor/models/qwen3_5.py) and its block-diffusion ModelState path
# (modeled on model_executor/models/diffusion_gemma.py).
# Upstream: vllm-project/vllm (https://github.com/vllm-project/vllm), Apache-2.0.
# Modifications by Trillion Labs are likewise released under Apache-2.0.
"""Trida block-diffusion on vLLM 0.27.x via the native ModelState path.

Marries vLLM's in-tree Qwen3.5 GDN backbone with the in-tree block-diffusion
`ModelState` machinery (modeled on `diffusion_gemma.py`).

Status: FULL two-stream path implemented.
- Commit sampler: lean confidence-shift / threshold gate (fp32-accumulated
  softmax so the gate is lossless).
- GDN two-stream: (a) block-ENTRY snapshot/restore of the recurrent+conv state
  in the ModelState, and (b) block-END readout for the noisy/denoise pass via a
  verbatim port of the SGLang block-causal kernel (block_causal_readout.py),
  patched onto the vLLM GDN layer's `_forward_core`. The clean/commit + prompt
  passes stay token-causal (stock kernel); only the denoise readout is swapped.

Differences vs DiffusionGemma we deliberately keep LEAN (see PERF notes):
- commit decision is top-1 prob + threshold (argmax + gather), NOT a full
  top-k/p sort over the 248k vocab.
- NO self-conditioning MLP (our Qwen3.5 backbone has none; DiffusionGemma's
  does) — so no `probs @ embed_weight` matmul, no normalizer, no sc buffer.
- NO entropy-bound accept mask / circular stability history / Gumbel sampling.
  Convergence is: a position is "done" once its top-1 prob >= threshold; the
  block commits when all real positions are done (or max steps hit).
"""
from __future__ import annotations

import os
from typing import Any

# Debug (env-gated, zero effect when unset): dump per-position denoise
# confidences each commit round to diagnose parallel-commit behavior.
_DBG_DUMP_CONF = os.environ.get("TRIDA_DUMP_CONF") == "1"

# Debug (env-gated): force the ¼ full-attention layers CAUSAL even on denoise
# rows, WITHOUT changing the GDN denoise-mode decision. Isolates whether the FA
# backend's per-request causal=False actually produces bidirectional within-block
# attention on this build. If tok/forward + confidences are unchanged vs the
# default, bidirectional never engaged (the noisy stream was effectively causal).
_DBG_FORCE_CAUSAL_DENOISE = os.environ.get("TRIDA_FORCE_CAUSAL_DENOISE") == "1"

# Diagnostics (env-gated, zero effect when unset). Single-request serving
# (max_num_seqs=1, sequential) is assumed; counters reset on each new request.
#   TRIDA_COUNT_FWD=1     -> [FWD] per-step running forward counts
#                            (prefill / denoise / commit) => true tokens/forward.
#   TRIDA_TRACK_REJECTS=1 -> [CAL] every masked-position prediction with its
#                            confidence, the token that eventually committed
#                            there, and its type (committed/rejected/forced/flush)
#                            => calibration curve; plus [BLK] committed ids per
#                            block for joining against the causal/AR ground truth.
_DBG_COUNT_FWD = os.environ.get("TRIDA_COUNT_FWD") == "1"
_DBG_TRACK_REJECTS = os.environ.get("TRIDA_TRACK_REJECTS") == "1"
#   TRIDA_TIME_PHASES=1   -> [TIME] per-decode-step wall split under REAL cuda-graph:
#                            snap (GDN snapshot/restore) / fwd (model forward + runner
#                            glue) / samp (commit gate). Syncs at phase boundaries.
_DBG_TIME_PHASES = os.environ.get("TRIDA_TIME_PHASES") == "1"
#   TRIDA_TIME_GDN=1      -> also time our eager GDN override per layer (24x/step),
#                            reported as gdn_ms/gdn_n in the [TIME] line.
_DBG_TIME_GDN = os.environ.get("TRIDA_TIME_GDN") == "1"
#   TRIDA_DUMP_STATE=1    -> [STATE] per active request at the START of every step
#                            (before snapshot/restore mutates anything): per GDN layer, a
#                            fixed 4-dim random projection of the ssm_state row + 2-dim of
#                            the conv_state row. Relative differences of these fingerprints
#                            track the true state difference (JL), so a block-by-block
#                            comparison against a fresh causal prefill of the same emitted
#                            prefix localizes where the commit chain drifts.
_DBG_DUMP_STATE = os.environ.get("TRIDA_DUMP_STATE") == "1"
# TRIDA_DUMP_HID=1 : per decoder layer, fingerprint the residual stream of the
#                    LAST canvas_length rows of every forward ([HID] layer= proj=
#                    [[2 dims] per position]) + RoPE positions of those rows
#                    ([HIDPOS]). Uses module hooks -> run the server eager.
_DBG_DUMP_HID = os.environ.get("TRIDA_DUMP_HID") == "1"
# TRIDA_CHARACTERIZE=1 (experiment E0): measure the MODEL, not the decode outcome -- per layer, how
# far the hidden state at a given absolute position moves between consecutive forwards. This is the
# ES-dLLM diagnostic (their Eq. 1) and it is the step we skipped before designing three mechanisms.
_CHARACTERIZE = os.environ.get("TRIDA_CHARACTERIZE") == "1"
# TRIDA_SHAPELOG=1: one line per decode step with the batch shape the step ran at. The concurrency
# crash is DETERMINISTIC with cuda graphs on (6/6 repeats died at the same point) and never happens
# eager, so it is a captured graph being replayed on a shape it was not captured for. This prints
# the shape sequence; the last line before the death is the shape to look for in the capture set.
_SHAPELOG = os.environ.get("TRIDA_SHAPELOG") == "1"
# TRIDA_FALLBACK_LOG=1: log every mixed-batch fallback with the metadata the stock GDN kernel gets.
# The crash is inside that kernel, and C=4 survives batches that look identical in shape to the ones
# that kill C=2 and C=8 -- so the discriminator is in this state, not in the shape.
_FALLBACK_LOG = os.environ.get("TRIDA_FALLBACK_LOG") == "1"
# TRIDA_GUARDS=1: bounds guards added while hunting the concurrency crash. One calls int(sl.max()),
# a device sync per forward; P0 measured 3.6 syncs/fwd against AR's 0, so they stay OFF unless asked.
# Never leave a sync in the hot path for a check that has never fired.
_GUARDS = os.environ.get("TRIDA_GUARDS") == "1"
# TRIDA_TRACE_JSONL=<path>: structured per-request decode trace (one JSON line per
#                    finished request): per block, per denoise round, per masked
#                    position: conf, pred, top-8 (token, logprob) of the shifted
#                    readout, commit decision + type; per block: ids, next seed,
#                    round count; per step: host wall ms + step type; prefill fwds.
#                    Costs one host sync per decode step (trace servers only).
_TRACE_PATH = os.environ.get("TRIDA_TRACE_JSONL", "")
_TRACE_ON = bool(_TRACE_PATH)
_TRACE: dict[int, dict] = {}      # slot -> in-flight record
_TRACE_META: dict[int, dict] = {} # slot -> {"req_id", "prompt_len"}
_TRACE_TOPK = int(os.environ.get("TRIDA_TRACE_TOPK", "8"))
# TRIDA_GDN_PACKED=0 : use the legacy multi-launch denoise GDN path instead of the
#                      fused packed kernel (A/B for Fix B3a).
_GDN_PACKED = os.environ.get("TRIDA_GDN_PACKED", "1") != "0"   # default ON: full-set GSM8K 76.6% vs 77.0% (same 1319 items), +33% tok/s
_DBG_NAN = os.environ.get("TRIDA_DEBUG_NAN") == "1"   # per-layer NaN/absmax probe in the denoise GDN path
# TRIDA_SELFSPEC_N=<N> : AR-Trust / self-speculative decoding (FLARE HybridDiffusionSelfSpec,
#                        greedy). Canvas = 2N-1 slots [pending, specs, MASKs]; one forward per
#                        step; specs verified against the exact AR logits; emits 1..N tokens per
#                        step. Requires canvas_length == 2N-1. Attention: whole canvas causal
#                        (FA3 has no intra-block custom mask): verify stays exact, drafts weaker.
_SELFSPEC_N = int(os.environ.get("TRIDA_SELFSPEC_N", "0"))
_SELFSPEC = _SELFSPEC_N > 0
# TRIDA_SS_FUSED=0 : use the two-kernel self-spec GDN path (conv kernel + packed) instead of the
#                    single fused layer kernel (Fix S2). Default fused.
_SS_FUSED = os.environ.get("TRIDA_SS_FUSED", "1") != "0"
# TRIDA_SS_SPECSHAPE=1 (default, self-spec only): present the step in vLLM's spec-decode shape --
#   query = 1 bonus token (slot 0, vLLM's last sampled token) + (2N-2) draft tokens (specs+MASKs);
#   diffusion canvas_length must then be 2N-2. Lets the runner classify the step as a UNIFORM
#   decode so FULL cuda graphs capture the whole forward. TRIDA_SS_SPECSHAPE=0 = canvas-as-query.
_SS_SPECSHAPE = _SELFSPEC and os.environ.get("TRIDA_SS_SPECSHAPE", "1") != "0"
# self-spec verify: 1 = stock speculative REJECTION sampling (drafts sampled from the truncated top-k/top-p
# proposal, per-request temperature honoured; temperature 0 == the exact-greedy path), 0 = exact-greedy only.
_SS_SAMPLE = os.environ.get("TRIDA_SS_SAMPLE", "1") == "1"
# draft policy under sampling verify: "sampled" (drafts ~ truncated proposal q) or "argmax" (drafts = mode of the
# MASK-slot logits, q one-hot -> accept prob = p(mode)); the verify target p and the recovery are unchanged.
_SS_DRAFT = os.environ.get("TRIDA_SS_DRAFT", "sampled").lower()
if _SS_DRAFT not in ("sampled", "argmax"):
    raise ValueError(f"TRIDA_SS_DRAFT must be 'sampled' or 'argmax', got {_SS_DRAFT!r}")


def _env_float(name: str, default: str) -> float:
    """Float env knob. The plugin is imported in EVERY vLLM process, so a typo here would otherwise
    surface as an unexplained traceback in the launcher, EngineCore and each worker at once."""
    raw = os.environ.get(name) or default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{name} must be a number, got {raw!r}") from None

# TRIDA_SS_DRAFT_FUSED=1 (default): fused Gumbel-max draft proposal + q write (one Triton kernel, per-request seeded);
# 0 = the torch path (softmax + multinomial + one-hot + index_put). Both are exact-greedy at temperature 0.
_SS_DRAFT_FUSED = os.environ.get("TRIDA_SS_DRAFT_FUSED", "1") == "1"
# proposal temperature for q (default: the request temperature); the verify target p always uses request params
_SS_DRAFT_TEMP = _env_float("TRIDA_SS_DRAFT_TEMP", "0")
# draft confidence gate in the sampled path: offer a draft only while q(draft) >= gate (0 = off)
_SS_GATE = _env_float("TRIDA_SS_GATE", "0")
# TRIDA_SS_CARRY=1 (experiment E2): on a rejection, re-offer the surviving drafts from the previous
# forward instead of resetting to a cold canvas. Verification is unchanged, so output stays lossless.
_SS_CARRY = _SELFSPEC and os.environ.get("TRIDA_SS_CARRY", "0") == "1"
# TRIDA_SS_CARRY_P=<float> (experiment E2b): carry a survivor only when its DRAFT-TIME confidence is at
# least this. 0.0 reproduces the ungated E2 arm exactly. Requires TRIDA_SIGNALS=1, which is what fills
# sspec_draft_p. Lossless either way: this changes which candidates are OFFERED, never what is verified.
_SS_CARRY_P = float(os.environ.get("TRIDA_SS_CARRY_P", "0") or 0)
# TRIDA_SIGNALS=1 (experiment E1): log per-slot p(drafted), entropy and top1-top2 margin beside the
# realised accept outcome, into the trace JSONL. Needs TRIDA_TRACE_JSONL to be set.
_SIGNALS_ON = bool(os.environ.get("TRIDA_TRACE_JSONL", "")) and os.environ.get("TRIDA_SIGNALS", "0") == "1"
# TRIDA_SS_BIDIR=1 (experiment E3): let the MASK half of the canvas attend to LATER mask slots, as it
# does in training. The verify half stays strictly causal, so the accept test -- and therefore the
# output -- is unchanged. See docs/E3_BIDIR_CANVAS.md; the merge is CPU-verified exact in
# tools/experiments/bidir_merge.py.
_SS_BIDIR = _SELFSPEC and os.environ.get("TRIDA_SS_BIDIR", "0") == "1"
_BIDIR: dict[str, Any] = {"on": False, "lse": None, "warned": False}


def _trace_rec(slot: int) -> dict:
    r = _TRACE.get(slot)
    if r is None:
        r = _TRACE[slot] = {"blocks": [], "cur": {"rounds": []}, "steps": [], "prefill_fwd": 0}
    return r


def _trace_flush(slot: int) -> None:
    rec = _TRACE.pop(slot, None); meta = _TRACE_META.pop(slot, None)
    if rec is None or meta is None or meta.get("req_id", "_warmup_").startswith("_warmup_"):
        return
    rec.pop("cur", None)
    rec.update(meta)
    rec["n_blocks"] = len(rec["blocks"])
    if _CHARACTERIZE and (_DIAG.get("char_skip") or _DIAG.get("char_err")):
        rec["char_skip"] = _DIAG.get("char_skip")
        rec["char_err"] = _DIAG.get("char_err")
    if _CHARACTERIZE and _DIAG.get("char"):
        # E0: per-layer hidden-state drift between consecutive forwards, as [n, mean, %>0.05, %>0.5].
        # Attached to every record; the analyser reads the last one, since the counters are cumulative
        # over the whole server and not per request.
        rec["char_layers"] = {str(li): [st[0], round(st[1] / max(st[0], 1), 6),
                                        round(100 * st[2] / max(st[0], 1), 3),
                                        round(100 * st[3] / max(st[0], 1), 3)]
                              for li, st in sorted(_DIAG["char"].items())}
    if "spec_steps" in rec:   # self-spec: one forward per step, tokens = 1 + accepted
        rec["decode_fwd"] = len(rec["spec_steps"])
        rec["tokens"] = sum(1 + a for _, a in rec["spec_steps"])
    else:
        rec["decode_fwd"] = sum(b["rounds"] for b in rec["blocks"]) + len(rec["blocks"])
        rec["tokens"] = sum(len(b["ids"]) for b in rec["blocks"])
    import json as _json
    with open(_TRACE_PATH, "a") as f:
        f.write(_json.dumps(rec, separators=(",", ":")) + "\n")
_STATE_PROJ: dict[tuple, Any] = {}
import time as _time


def _state_proj(li: int, n_ssm: int, n_conv: int, device: torch.device):
    key = (li, n_ssm, n_conv, str(device))
    if key not in _STATE_PROJ:
        g = torch.Generator(device="cpu").manual_seed(1234 + li)
        rs = torch.randn(4, n_ssm, generator=g).to(device)
        rc = torch.randn(2, n_conv, generator=g).to(device)
        _STATE_PROJ[key] = (rs, rc)
    return _STATE_PROJ[key]
_DIAG: dict[str, Any] = {
    "req": 0, "prefill": 0, "denoise": 0, "commit": 0, "step": 0, "last": "",
    "stash": {},   # slot -> {pos: [(conf, pred, step)]} not-yet-committed preds
    "blk": {},     # slot -> current block index
}

import numpy as np
import torch
import triton
import triton.language as tl
from vllm.triton_utils import tldevice

@triton.jit
def _ss_index_prep_kernel(sl_ptr, par_ptr, pacc_ptr, slots64_ptr, slots32_ptr, slots1_ptr, ssm_idx_ptr, write_idx_ptr,
                          n, n_pad, Rn, Nn, BLOCK: tl.constexpr):
    """Self-spec index prep for one step (replaces ~15 small torch launches).
    rows i <  n : slot s, parity par, prev_acc pacc -> slots=s, slots+1, ssm_idx=(par*Rn+s)*Nn+pacc (ring read),
                  write_idx=(1-par)*Rn+s (ring write)
    rows n..n_pad: cuda-graph padding -> slot Rn (dummy), slots+1=Rn+1, ssm_idx=2*Rn*Nn, write_idx=2*Rn (scratch)."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n_pad
    real = i < n
    s = tl.load(sl_ptr + i, mask=real, other=0).to(tl.int64)
    par = tl.load(par_ptr + i, mask=real, other=0).to(tl.int64)
    pacc = tl.load(pacc_ptr + i, mask=real, other=0).to(tl.int64)
    slot = tl.where(real, s, Rn + s * 0)                       # int64 (s is int64; Rn is an int scalar)
    ssm = tl.where(real, (par * Rn + s) * Nn + pacc, 2 * Rn * Nn + s * 0)
    wr = tl.where(real, (1 - par) * Rn + s, 2 * Rn + s * 0)
    tl.store(slots64_ptr + i, slot, mask=m)
    tl.store(slots32_ptr + i, slot.to(tl.int32), mask=m)
    tl.store(slots1_ptr + i, (slot + 1).to(tl.int32), mask=m)
    tl.store(ssm_idx_ptr + i, ssm.to(tl.int32), mask=m)
    tl.store(write_idx_ptr + i, wr.to(tl.int32), mask=m)

@triton.jit
def _ss_draft_gumbel_kernel(proc_ptr, proc_stride, temp_ptr, seed_ptr, pos_ptr, allacc_ptr, slot_ptr, j_ptr,
                            tok_out_ptr, qbuf_ptr, qbuf_stride0, qbuf_stride1, mask_id, V, BLOCK: tl.constexpr):
    """Fused draft proposal (one program per draft row = one (request, draft slot)):
       token = argmax(proc + Gumbel)  (temp>0; Gumbel seeded by (request seed, pos) like vLLM's gumbel_sample)
             = argmax(proc)           (temp==0, exact-greedy limit)
       and the q row for the next step's rejection sampler is written in the same pass:
       all_acc -> q = proc (the truncated proposal the draft was drawn from);  else -> MASK one-hot (cold restart).
       Replaces softmax + multinomial + argmax + full_like/scatter one-hot + 2x where + index_put."""
    r = tl.program_id(0).to(tl.int64)
    temp = tl.load(temp_ptr + r).to(tl.float32)
    seed = tl.load(seed_ptr + r)
    pos = tl.load(pos_ptr + r)
    allacc = tl.load(allacc_ptr + r) != 0
    slot = tl.load(slot_ptr + r).to(tl.int64)
    j = tl.load(j_ptr + r).to(tl.int64)
    gseed = tl.randint(seed, pos)
    best_v = float("-inf"); best_i = 0
    qrow = qbuf_ptr + slot * qbuf_stride0 + j * qbuf_stride1
    for start in tl.range(0, V, BLOCK):
        blk = start + tl.arange(0, BLOCK)
        m = blk < V
        x = tl.load(proc_ptr + r * proc_stride + blk, mask=m, other=float("-inf")).to(tl.float32)
        if temp > 0.0:
            u = tl.rand(gseed, blk)
            u = tl.maximum(u, 4.6566127342e-10)
            y = tl.where(m, x + (-tl.log(-tldevice.log1p(-u))), float("-inf"))   # vLLM's well-resolved tail form
        else:
            y = x
        bv, bi = tl.max(y, axis=0, return_indices=True)
        take = bv > best_v
        best_i = tl.where(take, start + bi, best_i)
        best_v = tl.where(take, bv, best_v)
        qv = tl.where(allacc, x, tl.where(blk == mask_id, 0.0, float("-inf")))
        tl.store(qrow + blk, qv, mask=m)
    tl.store(tok_out_ptr + r, best_i.to(tl.int64))


import torch.nn as nn

# vLLM 0.27.x internals (present in the vllm-uv27 venv).
from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
from vllm.platforms import current_platform
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor, async_copy_to_gpu
from vllm.v1.worker.gpu.model_states.interface import ModelState, ModelSpecificAttnMetadata
from dataclasses import dataclass as _dataclass


@_dataclass
class _TridaSpecAttnMetadata(ModelSpecificAttnMetadata):
    """Spec-decode extras for the attention/GDN metadata builders (self-spec spec-shape)."""
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(self, kv_cache_group_id: int, num_reqs: int) -> dict:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(self, attn_metadata_builder, num_reqs: int) -> dict:
        from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
        if not isinstance(attn_metadata_builder, GDNAttentionMetadataBuilder):
            return {}
        return {"num_accepted_tokens": None if self.num_accepted_tokens is None else self.num_accepted_tokens[:num_reqs],
                "num_decode_draft_tokens_cpu": None if self.num_decode_draft_tokens_cpu is None else self.num_decode_draft_tokens_cpu[:num_reqs]}
from vllm.v1.worker.gpu.sample.output import SamplerOutput


# ---------------------------------------------------------------------------
# Per-forward GDN readout mode, set by the ModelState in prepare_attn and read
# by the patched GDN `_forward_core`. The two-stream noisy/denoise pass uses
# block-END readout (FLARE), the clean/commit and prompt passes stay token-causal.
#   mode: "denoise" -> mixed block-causal readout (causal_mode=2, num_clean=1:
#         seed reads its own token-causal state, masks read block-END state) over
#         the canvas block, matching SGLang LowConfidenceShift _set_denoise_flags;
#         anything else -> stock token-causal path (super()._forward_core).
#   block_size: the canvas/block length (num logits per denoise request).
# Single engine process, single writer (the model runner drives one step at a
# time), so a plain module dict is safe.
# ---------------------------------------------------------------------------
_TRIDA_GDN_READOUT: dict[str, Any] = {"mode": "other", "block_size": 0}


# ---------------------------------------------------------------------------
# Per-request diffusion bookkeeping (clean=commit / noisy=denoise)
# ---------------------------------------------------------------------------
class TridaDiffusionStates:
    """Per-request state for the two-stream decode, in pre-allocated GPU buffers.

    `is_encoder_phase[req]` == True  -> clean/commit  == causal   (encoder)
                            == False -> noisy/denoise == bidirectional (decoder)

    We keep the canvas lifecycle that the runner's spec-decode data path drives
    (canvas seeds `draft_tokens`; `num_draft_tokens>0` == a decode step), plus a
    `done` mask per canvas position for our confidence-shift commit gate.
    """

    def __init__(
        self,
        max_num_reqs: int,
        canvas_length: int,
        vocab_size: int,
        max_denoising_steps: int,
        device: torch.device,
    ):
        self.device = device
        self.max_num_reqs = max_num_reqs
        self.canvas_length = canvas_length
        self.vocab_size = vocab_size
        self.max_denoising_steps = max_denoising_steps

        self.mask_id = vocab_size - 1  # overwritten by the ModelState with the real id

        self.is_encoder_phase = torch.ones(max_num_reqs, dtype=torch.bool, device=device)
        self.prompt_len = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self.step = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        # Block canvas, laid out [seed, MASK, MASK, ...] (LowConfidenceShift). The
        # seed at position 0 is the carried token predicted by the previous block
        # (or the prompt prefill); positions 1..B-1 start MASK and are denoised.
        self.canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        # Committed best-guess per position (the tokens we emit).
        self.argmax_canvas = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.int64, device=device
        )
        # Per-position committed/confident mask (position 0 = seed is always done).
        self.done = torch.zeros(
            max_num_reqs, canvas_length, dtype=torch.bool, device=device
        )
        # Carried seed token for position 0 of the NEXT block. Set from the last
        # prompt logit (cold start) and re-set from each committed block's last
        # (shifted) logit.
        self.seed = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.has_seed = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        # True on the first denoise step of a fresh block, so the seed emits once.
        self.fresh_block = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)
        # GDN block-entry state snapshots handled by the ModelState (kept here for
        # the remove_request hook parity).
        self.gdn_block_entry_state: dict[int, Any] = {}
        # self-spec: number of speculative tokens currently in canvas slots 1..k
        self.sspec_k = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)
        self.sspec_prev_acc = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)  # n_acc of the previous step
        self.sspec_started = torch.zeros(max_num_reqs, dtype=torch.bool, device=device)    # cache rows imported into the ring
        # CPU mirror of sspec_started. prepare_attn used to decide "does any slot need its cache rows
        # imported" with bool(need.any().item()) on the GPU tensor -- one cudaStreamSynchronize per
        # call, and P0 measured 3.4 of them per forward against AR's zero. The decision only needs
        # host-side truth, so it is kept here and the GPU flag is written only on the rare import.
        self.sspec_started_np = np.zeros(max_num_reqs, dtype=bool)
        self.sspec_par = torch.zeros(max_num_reqs, dtype=torch.int64, device=device)       # per-slot ring parity (state lives in half `par`)
        # E1b: the confidence the model had in each draft AT THE MOMENT IT DREW IT, i.e. read from
        # the MASK-slot logits of the forward that produced it, one step BEFORE the forward that
        # verifies it. This is the only acceptance signal that exists early enough to choose which
        # candidates to offer. The verify-time probability of the same token is not a predictor:
        # acceptance is defined as `drafted == argmax(verify logits)`, so reading p from those same
        # logits restates the accept rule and cannot gate anything.
        self.sspec_draft_p = torch.zeros(max_num_reqs, max(_SELFSPEC_N - 1, 1), device=device)

    def init_block(self, slot_indices) -> None:
        """(Re)initialize a block canvas to [seed, MASK, ...] for the given slots.
        Requires self.seed/has_seed to already hold the carried seed."""
        if isinstance(slot_indices, np.ndarray):
            slot_indices = torch.as_tensor(slot_indices, device=self.device)
        n = int(slot_indices.numel())
        if n == 0:
            return
        self.canvas[slot_indices] = self.mask_id
        self.canvas[slot_indices, 0] = self.seed[slot_indices]
        self.argmax_canvas[slot_indices] = self.mask_id
        self.argmax_canvas[slot_indices, 0] = self.seed[slot_indices]
        self.done[slot_indices] = False
        self.done[slot_indices, 0] = True  # seed committed
        self.step[slot_indices] = 0
        self.fresh_block[slot_indices] = True
        self.sspec_k[slot_indices] = 0
        self.sspec_prev_acc[slot_indices] = 0
        self.sspec_started[slot_indices] = False
        self.sspec_par[slot_indices] = 0

    def add_request(self, req_index: int) -> None:
        self.is_encoder_phase[req_index] = True  # prompt/commit (causal) first
        self.sspec_started_np[req_index] = False
        self.has_seed[req_index] = False
        self.fresh_block[req_index] = False

    def remove_request(self, req_index: int) -> None:
        self.is_encoder_phase[req_index] = False
        self.sspec_started_np[req_index] = False
        self.has_seed[req_index] = False
        self.gdn_block_entry_state.pop(req_index, None)


# ---------------------------------------------------------------------------
# Lean commit step (torch.compiled). Our confidence-shift / threshold gate.
# ---------------------------------------------------------------------------
def _shifted_conf_pred(logits_3d: torch.Tensor, mask_id: int):
    """Dream logit-SHIFT readout (bit-matched to LowConfidenceShiftHybridDiffusion):
    the token at canvas position i (i>=1) is predicted by the logit at position
    i-1. Returns per-position (conf, pred) with position 0 (seed) left as a
    sentinel (never resampled). Softmax in fp32 (lossless gate).

    logits_3d: [nd, CL, vocab].  Returns conf,pred: [nd, CL].
    """
    nd, CL, _ = logits_3d.shape
    # predecessor logits for positions 1..CL-1 are logits[0..CL-2].
    prev = logits_3d[:, : CL - 1, :]                       # [nd, CL-1, vocab]
    probs = torch.softmax(prev, dim=-1, dtype=torch.float32)
    c, p = probs.max(dim=-1)                               # [nd, CL-1]
    neg = torch.full((nd, 1), float("-inf"), device=logits_3d.device)
    conf = torch.cat([neg, c], dim=1)                      # pos 0 => -inf
    pred = torch.cat([torch.full((nd, 1), mask_id, dtype=torch.long,
                                 device=logits_3d.device), p], dim=1)
    return conf, pred


def _next_seed_from_logits(logits_3d: torch.Tensor):
    """Next block's seed = argmax of the LAST position's (shifted) logit
    (mirrors _capture_next_seeds reading full_logits[base+B-1])."""
    last = logits_3d[:, -1, :]                             # [nd, vocab]
    return last.argmax(dim=-1)                             # [nd]


def _denoise_shift_step(
    logits_3d: torch.Tensor,       # [nd, CL, vocab]
    slots: torch.Tensor,           # [nd] int64
    canvas: torch.Tensor,          # [R, CL] in/out
    argmax_canvas: torch.Tensor,   # [R, CL] in/out
    done: torch.Tensor,            # [R, CL] bool in/out
    step_tensor: torch.Tensor,     # [R] in/out
    is_encoder_phase: torch.Tensor,  # [R] bool in/out
    mask_id: int,
    threshold: float,
    max_steps: int,
    is_last_step_forced: bool = False,
    active: torch.Tensor | None = None,  # [nd] bool: gate writes to these rows
) -> None:
    """One LowConfidenceShift denoise round for the given slots (in-place).

    - shift readout (pos i predicted by logit i-1); pos 0 (seed) never resampled
    - commit masks whose conf>threshold; if a row has masks but none pass, force
      the single top-1 (mirrors the reference `need`/`force` fallback)
    - when a block has no masks left (or hit max_steps) flip to commit phase.

    DE-SYNC: callers now pass ALL decode rows in `slots` and an `active` mask
    marking the true denoise rows (the rest are commit rows handled elsewhere).
    Every write is gated with torch.where(active, new, old) so inactive rows are
    untouched — identical to the old subset-select (`slots[dsel]`) path, but with
    a fixed shape (no data-dependent `nonzero()`), no host sync.
    """
    nd, CL, _ = logits_3d.shape
    device = logits_3d.device
    conf, pred = _shifted_conf_pred(logits_3d, mask_id)    # [nd, CL]

    cur = canvas[slots]                                    # [nd, CL]
    m = cur == mask_id                                     # still-masked
    m = m.clone()
    m[:, 0] = False                                        # seed never resampled

    if is_last_step_forced:
        transfer = m
    else:
        cscore = torch.where(m, conf, torch.full_like(conf, float("-inf")))
        transfer = cscore > threshold
        need = m.any(dim=1) & ~transfer.any(dim=1)
        force = torch.zeros_like(transfer)
        force[torch.arange(nd, device=device), cscore.argmax(dim=1)] = True
        transfer = torch.where(need.unsqueeze(1), force, transfer)

        if _DBG_DUMP_CONF:
            # Per active denoise row: masked-position confidences (sorted desc),
            # how many exceed threshold (= would parallel-commit), whether the
            # force fallback fired (nothing passed -> 1 forced). Forces a sync;
            # only when TRIDA_DUMP_CONF=1.
            arows = (active if active is not None
                     else torch.ones(nd, dtype=torch.bool, device=device))
            ai = torch.nonzero(arows, as_tuple=False).flatten().tolist()
            for r in ai:
                mrow = m[r]
                if not bool(mrow.any()):
                    continue
                cvals = conf[r][mrow]
                svals, _ = torch.sort(cvals, descending=True)
                n_pass = int((cvals > threshold).sum().item())
                n_mask = int(mrow.sum().item())
                forced = bool(need[r].item()) if not is_last_step_forced else False
                topk = [round(float(x), 4) for x in svals[:8].tolist()]
                print(f"[DUMP_CONF] row={r} masked={n_mask} thr={threshold} "
                      f"n_pass={n_pass} forced={forced} committed={int(transfer[r].sum().item())} "
                      f"conf_desc={topk}", flush=True)

    if _DBG_TRACK_REJECTS:
        # Calibration tracking (forces syncs; diagnostic-only). For every masked
        # position this round: if it commits now, emit its own prediction AND
        # every earlier (rejected) prediction stashed for it, each compared to
        # the token that actually commits (`final`). Otherwise stash it.
        # Types: committed (cleared gate) / forced (top-1 fallback) /
        #        flush (last-step unconditional) / rejected (below gate).
        _need = None if is_last_step_forced else need
        _fpos = None if is_last_step_forced else cscore.argmax(dim=1)
        arows = (active if active is not None
                 else torch.ones(nd, dtype=torch.bool, device=device))
        _step, _req = _DIAG["step"], _DIAG["req"]
        for r in torch.nonzero(arows, as_tuple=False).flatten().tolist():
            slot = int(slots[r].item())
            blk = _DIAG["blk"].get(slot, 0)
            st = _DIAG["stash"].setdefault(slot, {})
            for i in torch.nonzero(m[r], as_tuple=False).flatten().tolist():
                c = float(conf[r, i].item()); p = int(pred[r, i].item())
                committing = bool(transfer[r, i].item())
                if is_last_step_forced:
                    typ = "flush"
                elif (_need is not None and bool(_need[r].item())
                      and i == int(_fpos[r].item())):
                    typ = "forced"
                else:
                    typ = "committed" if committing else "rejected"
                if committing:
                    print(f"[CAL] req={_req} slot={slot} blk={blk} pos={i} "
                          f"step={_step} conf={c:.4f} pred={p} final={p} type={typ}",
                          flush=True)
                    for (c0, p0, s0) in st.pop(i, []):
                        print(f"[CAL] req={_req} slot={slot} blk={blk} pos={i} "
                              f"step={s0} conf={c0:.4f} pred={p0} final={p} "
                              f"type=rejected", flush=True)
                else:
                    st.setdefault(i, []).append((c, p, _step))

    if _TRACE_ON:
        # Structured per-round trace (one sync). Top-k of the SHIFTED readout for
        # every canvas position >= 1 (position i <- logit i-1).
        arows = (active if active is not None
                 else torch.ones(nd, dtype=torch.bool, device=device))
        prev_lp = torch.log_softmax(logits_3d[:, : CL - 1, :], dim=-1, dtype=torch.float32)
        tk_v, tk_i = prev_lp.topk(_TRACE_TOPK, dim=-1)               # [nd, CL-1, K]
        A = torch.nonzero(arows, as_tuple=False).flatten().tolist()
        if A:
            m_l = m[A].tolist(); tr_l = transfer[A].tolist(); conf_l = conf[A].tolist(); pred_l = pred[A].tolist()
            tkv_l = tk_v[A].tolist(); tki_l = tk_i[A].tolist(); slots_l = slots[A].tolist()
            need_l = ([False] * len(A) if is_last_step_forced else need[A].tolist())
            fpos_l = ([-1] * len(A) if is_last_step_forced else cscore.argmax(dim=1)[A].tolist())
            for j, r in enumerate(A):
                pos = {}
                for i in range(1, CL):
                    if not m_l[j][i]:
                        continue
                    if is_last_step_forced:
                        typ = "flush"
                    elif need_l[j] and i == fpos_l[j]:
                        typ = "forced"
                    else:
                        typ = "committed" if tr_l[j][i] else "rejected"
                    pos[str(i)] = {"c": round(conf_l[j][i], 5), "p": pred_l[j][i], "t": typ,
                                   "top": [[tki_l[j][i - 1][k], round(tkv_l[j][i - 1][k], 4)]
                                           for k in range(_TRACE_TOPK)]}
                _trace_rec(int(slots_l[j]))["cur"]["rounds"].append(pos)

    new_canvas = torch.where(transfer, pred, cur)
    new_done = new_canvas != mask_id
    new_step = step_tensor[slots] + 1
    still_masked = (new_canvas == mask_id).any(dim=1)
    converged = (~still_masked) | (new_step >= max_steps)
    # denoise converged -> commit next (True)
    enc = is_encoder_phase[slots]
    new_enc = torch.where(enc, enc, converged)

    if active is not None:
        # Row-gate every write: keep prior value where a row is NOT an active
        # denoise row (it is a commit row, mutated by the commit path instead).
        arow = active.unsqueeze(1)                         # [nd,1] for [nd,CL]
        new_canvas = torch.where(arow, new_canvas, cur)
        new_done = torch.where(arow, new_done, done[slots])
        new_step = torch.where(active, new_step, step_tensor[slots])
        new_enc = torch.where(active, new_enc, enc)

    canvas[slots] = new_canvas
    argmax_canvas[slots] = new_canvas
    done[slots] = new_done
    step_tensor[slots] = new_step
    is_encoder_phase[slots] = new_enc


@torch.compile(dynamic=True)
def _compute_num_rejected(
    num_logits: torch.Tensor,
    num_sampled: torch.Tensor,
    query_start_loc: torch.Tensor,
) -> torch.Tensor:
    query_lens = query_start_loc[1:] - query_start_loc[:-1]
    num_rejected = num_logits - num_sampled
    is_denoise = (num_logits > 0) & (num_sampled == 0)
    return torch.where(is_denoise, query_lens, num_rejected)



# ---------------------------------------------------------------------------
# Sampler: prefill/decode lifecycle mirrors DiffusionGemma; commit gate is ours.
# ---------------------------------------------------------------------------
class TridaDiffusionSampler:
    def __init__(
        self,
        sampler: Any,
        diffusion_config: Any,
        vocab_size: int,
        diffusion_states: TridaDiffusionStates,
        *,
        confidence_threshold: float,
        mask_id: int,
        req_states: Any,
    ):
        self.sampling_states = sampler.sampling_states
        self._base_sampler = sampler   # stock worker Sampler (for the RejectionSampler)
        self.req_states = req_states
        self.vocab_size = vocab_size
        self.diffusion_states = diffusion_states
        self.canvas_length = (
            diffusion_config.canvas_length if diffusion_config is not None else 32
        )
        if _SS_SPECSHAPE:
            self.canvas_length += 1          # bonus token + drafts
        self.confidence_threshold = confidence_threshold
        self._rs = None                 # stock worker RejectionSampler (built lazily, spec-shape only)
        self._rs_failed = False
        self._ss_draft_logits = None    # [max_num_reqs, 2N-2, V] fp32: processed proposal logits (q) per draft slot
        self._ss_maskq = None           # [1, N-1, V] fp32: constant MASK one-hot q (cold restart on a rejection)
        self._ss_draft_ctr = None       # per-slot draft counter (Gumbel position)
        self._cur_logits = None         # full [num_logits, V] logits of the current step
        self.mask_id = mask_id

        max_num_reqs = diffusion_states.max_num_reqs
        device = diffusion_states.device
        if _SELFSPEC and _SS_SPECSHAPE and _SS_SAMPLE:
            # Eagerly, not on the first decode: at 248k vocab this is ~95 MB at max_num_reqs 16 and
            # ~190 MB at 32. Allocated here it is part of the profiling run vLLM sizes the KV cache
            # against; allocated on the first request it is memory vLLM already promised away, which
            # shows up as an out-of-memory error mid-serve instead of a smaller cache at startup.
            self._ss_alloc_q(max_num_reqs, device)
        self._sampled = torch.zeros(
            max_num_reqs, self.canvas_length, dtype=torch.int32, device=device
        )
        self._num_sampled = torch.zeros(max_num_reqs, dtype=torch.int32, device=device)
        self._decode_slots = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._decode_idx = UvaBackedTensor(max_num_reqs, dtype=torch.int64)
        self._query_lens = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self._num_logits = UvaBackedTensor(max_num_reqs, dtype=torch.int32)

    # -- request lifecycle (called by the runner via ModelState) ------------
    def add_request(self, req_idx: int, prompt_len: int, sampling_params: Any) -> None:
        self.sampling_states.add_request(req_idx, sampling_params)

    def apply_staged_writes(self) -> None:
        self.sampling_states.apply_staged_writes()

    @property
    def penalties_state(self):
        from types import SimpleNamespace
        return SimpleNamespace(output_bin_counts=None)

    # -- prefill: capture the cold-start seed from the last prompt logit -----
    def _write_drafts(self, slots) -> None:
        """draft_tokens <- canvas (spec-shape: drafts are canvas[1:], slot 0 is vLLM's bonus token)."""
        CL = self.canvas_length
        if _SS_SPECSHAPE:
            self.req_states.draft_tokens[slots, : CL - 1] = self.diffusion_states.canvas[slots, 1:]
        else:
            self.req_states.draft_tokens[slots, :CL] = self.diffusion_states.canvas[slots]

    def _finish_prefills(self, input_batch: Any, prefill_indices_np: np.ndarray,
                         logits: torch.Tensor):
        """For requests whose prompt completes this step: sample the first
        SEED from the last prompt logit (LowConfidenceShift _capture_prefill_
        seeds), initialize the [seed, MASK, ...] block, seed draft_tokens, and
        flip to denoise phase."""
        states = self.diffusion_states
        done_prefill_np = (
            input_batch.num_computed_prefill_tokens_np[prefill_indices_np]
            + input_batch.num_scheduled_tokens[prefill_indices_np]
            >= input_batch.prefill_len_np[prefill_indices_np]
        )
        done_idx = prefill_indices_np[done_prefill_np]
        ps = input_batch.idx_mapping_np[done_idx]
        if len(ps) == 0:
            return None
        # Last logit index of each finished prompt = cu_num_logits end - 1.
        cu = input_batch.cu_num_logits_np
        last_logit_np = cu[done_idx + 1] - 1
        last_logit = async_copy_to_gpu(last_logit_np.astype(np.int64), device=logits.device)
        seed = logits[last_logit].argmax(dim=-1)           # [len(ps)]
        ps_gpu = async_copy_to_gpu(ps.astype(np.int64), device=logits.device)
        states.seed[ps_gpu] = seed
        states.has_seed[ps_gpu] = True
        states.init_block(ps_gpu)
        states.sspec_started_np[ps] = False      # mirrors init_block's GPU reset, numpy side
        self._write_drafts(ps_gpu)
        if not _SELFSPEC:
            states.is_encoder_phase.index_fill_(0, ps_gpu, False)
        # self-spec: stays causal (is_encoder_phase True); canvas [t0, MASK...] with k=0.
        return done_idx, seed

    def _handle_prefill(self, input_batch: Any, logits: torch.Tensor) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        fin = self._finish_prefills(input_batch, np.arange(num_reqs), logits)
        sampled = self._sampled[:num_reqs, :1]
        sampled.zero_()
        num_sampled = self._num_sampled[:num_reqs]
        num_sampled.zero_()
        if _SS_SPECSHAPE and fin is not None:
            # spec-shape: the seed IS the bonus token -> emit it (vLLM prepends it to the next query)
            di = async_copy_to_gpu(fin[0].astype(np.int64), device=logits.device)
            sampled[di, 0] = fin[1].to(sampled.dtype); num_sampled[di] = 1
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            # prefill: 1 logit per request; nothing is rejected (spec-shape emits the seed as the
            # bonus token, so num_rejected MUST be 0 -- vLLM's post_update advances num_computed by
            # query_len - num_rejected; reporting 1 here rolled the runner back one token and the
            # first self-spec step overwrote the last prompt token: "Tod\nTo determine ...").
            num_rejected=torch.zeros_like(num_sampled),
        )

    def _ss_alloc_q(self, R, dev):
        """Proposal-logit buffers for the rejection sampler: q per draft slot, plus the constant MASK one-hot."""
        N, V = _SELFSPEC_N, self.vocab_size
        K = self.canvas_length - 1                                     # 2N-2 draft slots
        self._ss_draft_logits = torch.full((R, K, V), float("-inf"), dtype=torch.float32, device=dev)
        self._ss_draft_logits[:, :, self.mask_id] = 0.0                # every slot starts as a MASK pseudo-draft
        # rows N-1..K-1 are ALWAYS MASK pseudo-drafts (k <= N-1), so they are never rewritten; only the
        # first N-1 rows change per step. Constant MASK one-hot for the "cold on rejection" case:
        self._ss_maskq = torch.full((1, N - 1, V), float("-inf"), dtype=torch.float32, device=dev)
        self._ss_maskq[:, :, self.mask_id] = 0.0
        self._ss_draft_ctr = torch.zeros(R, dtype=torch.int64, device=dev)   # per-slot draft counter (Gumbel pos)

    def _ss_sample_verify(self, input_batch, logits_3d, decode_slots, decode_idx, canvas, k, N, j):
        """Speculative rejection sampling via vLLM's stock worker RejectionSampler.

        Returns (n_acc, next_tok, new_specs, sidx, n_ok) or None (fall back to exact-greedy) if the
        stock sampler cannot be built. n_ok = drafts actually offered per request (< N-1 only when the
        confidence gate truncates the prefix). q for each real draft = the processed (temperature / top-k / top-p)
        proposal logits it was sampled from; MASK pseudo-drafts carry a one-hot q on MASK and the
        target's MASK mass is zeroed, so they always reject and the recovery is a target sample."""
        nd, blk, V = logits_3d.shape; K = blk - 1; dev = logits_3d.device
        if self._rs is None:
            try:
                from vllm.v1.worker.gpu.spec_decode.rejection_sampler import RejectionSampler as _WRS

                class _SC:  # the fields RejectionSampler reads from SpeculativeConfig
                    num_speculative_tokens = K
                    rejection_sample_method = "standard"
                    synthetic_acceptance_rates = None
                self._rs = _WRS(self._base_sampler, _SC(), dev)
                print(f"[trida-selfspec] verify = stock speculative rejection sampling (K={K}, drafts={_SS_DRAFT}: "
                      f"{'argmax of the MASK-slot logits, q one-hot' if _SS_DRAFT == 'argmax' else 'sampled from the truncated top-k/top-p proposal'}; "
                      f"temperature 0 == exact greedy)", flush=True)
            except Exception as e:  # noqa: BLE001
                self._rs_failed = True
                print(f"[trida-selfspec] WARNING stock RejectionSampler unavailable ({e!r}): exact-greedy verify", flush=True)
                return None
        rs = self._rs
        if self._ss_draft_logits is None:                                # fallback: flags differed at construction
            self._ss_alloc_q(self.diffusion_states.canvas.shape[0], dev)
        logits = self._cur_logits
        # In place ON PURPOSE: this is the [num_logits, 248k] tensor the runner just produced, and copying
        # it would cost more than everything this path saves. Safe because the step's only consumer is the
        # rejection sampler below and we return logprobs_tensors=None; if logprobs are ever enabled for the
        # diffusion path, mask a copy here instead.
        logits[:, self.mask_id] = float("-inf")                          # MASK is never an output token
        so = rs(logits, input_batch, self._ss_draft_logits)
        st = so.sampled_token_ids[decode_idx]                            # [nd, K+1]: accepted drafts, then recovery/bonus
        n_acc = (so.num_sampled[decode_idx].to(torch.int64) - 1).clamp(min=0)
        n_acc = torch.minimum(n_acc, k)                                  # pseudo-drafts cannot be accepted (p(MASK)=0)
        next_tok = st.gather(1, n_acc[:, None]).squeeze(1)
        # ---- next drafts: sample from the MASK-slot logits at slots n_acc+1.. with the request's params
        sidx = n_acc[:, None] + 1 + j[None, :]                           # [nd, N-1]
        lg = logits_3d.gather(1, sidx[:, :, None].expand(-1, -1, V)).float()   # [nd, N-1, V]
        ss = rs.sampler.sampling_states
        temp = ss.temperature.gpu[decode_slots].float()
        greedy = temp <= 0.0
        tq = torch.where(greedy, torch.ones_like(temp), temp)
        if _SS_DRAFT_TEMP > 0:                                           # proposal temperature knob (q only)
            tq = torch.where(greedy, torch.ones_like(temp), torch.full_like(temp, _SS_DRAFT_TEMP))
        proc = (lg / tq[:, None, None]).reshape(nd * (N - 1), V)
        idx3 = decode_slots.repeat_interleave(N - 1)                      # [nd*(N-1)] one expansion for all gathers
        topk = ss.top_k.gpu[idx3]
        topp = ss.top_p.gpu[idx3]
        from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p
        proc = apply_top_k_top_p(proc, topk, topp)                       # truncated proposal (in place)
        _fused_ok = (_SS_DRAFT_FUSED and _SS_GATE <= 0 and _SS_DRAFT != "argmax")
        if _fused_ok:
            # ---- fused proposal: Gumbel-max sample (or argmax at temp 0) + q write, one kernel over [nd*(N-1), V]
            all_acc = (n_acc == k)
            ss_seeds = ss.seeds.gpu[decode_slots]
            ctr = self._ss_draft_ctr[decode_slots]; self._ss_draft_ctr[decode_slots] = ctr + 1
            rows = nd * (N - 1)
            row_pos = (ctr[:, None] * (N - 1) + j[None, :]).reshape(-1)          # distinct pos per (request, slot)
            row_temp = tq.repeat_interleave(N - 1)
            row_temp = torch.where(greedy.repeat_interleave(N - 1), torch.zeros_like(row_temp), row_temp)
            new_specs_flat = torch.empty(rows, dtype=torch.int64, device=dev)
            buf = self._ss_draft_logits
            _ss_draft_gumbel_kernel[(rows,)](
                proc, proc.stride(0), row_temp, ss_seeds.repeat_interleave(N - 1), row_pos,
                all_acc.to(torch.int32).repeat_interleave(N - 1), idx3, j.repeat(nd),
                new_specs_flat, buf, buf.stride(0), buf.stride(1), self.mask_id, V, BLOCK=4096)
            new_specs = new_specs_flat.view(nd, N - 1)
            n_ok = torch.full((nd,), N - 1, dtype=torch.int64, device=dev)
        else:
            am_specs = proc.argmax(dim=-1)
            samp = torch.multinomial(torch.softmax(proc, dim=-1), 1).squeeze(1)
            g2 = greedy.repeat_interleave(N - 1)
            if _SS_DRAFT == "argmax":                                        # argmax drafts, one-hot q
                g2 = torch.ones_like(g2)
            new_specs = torch.where(g2, am_specs, samp).view(nd, N - 1)
            # q handed to the kernel next step = exactly the distribution the drafts were drawn from
            onehot = torch.full_like(proc, float("-inf")).scatter_(1, am_specs[:, None], 0.0)
            q = torch.where(g2[:, None], onehot, proc).view(nd, N - 1, V)
            all_acc = (n_acc == k)[:, None, None]
            maskq = self._ss_maskq                                            # [1, N-1, V], broadcast
            n_ok = torch.full((nd,), N - 1, dtype=torch.int64, device=dev)
            if _SS_GATE > 0:                                                  # draft confidence gate (prefix-truncating)
                pq = torch.softmax(proc, dim=-1).gather(1, new_specs.reshape(-1)[:, None]).view(nd, N - 1)
                ok = torch.cumprod((pq >= _SS_GATE).to(torch.int64), dim=1).bool()
                n_ok = ok.to(torch.int64).sum(dim=1)
                new_specs = torch.where(ok, new_specs, torch.full_like(new_specs, self.mask_id))
                q = torch.where(ok[:, :, None], q, maskq)
            # one write of the N-1 real-draft rows (rows >= N-1 are permanent MASK one-hots)
            self._ss_draft_logits[decode_slots, : N - 1] = torch.where(all_acc, q, maskq)   # cold on a rejection
        return n_acc, next_tok, new_specs, sidx, n_ok

    def _selfspec_step(self, input_batch, logits_3d, decode_slots, decode_idx, all_slots,
                       sampled, num_sampled, per_req_nlogits_np) -> SamplerOutput:
        """One AR-Trust step (greedy). Canvas [t0, spec_1..spec_k, MASK...] (blk=2N-1).
        Shift readout: logit at slot j predicts the token of slot j+1.
          - verify specs left to right: accept while argmax(logit_j) == spec_{j+1}
          - emit canvas[0 : 1+n_acc]  (t0 + accepted specs; their KV/GDN are computed)
          - next slot-0 token = argmax(logit_{n_acc})  (clean if all accepted, else the correction)
          - if all accepted: new specs = argmax(logit_{n_acc+1 .. n_acc+N-1}) (from the MASK slots)
            else: cold start (no specs)
        GDN state: persist intermediate state after slot n_acc; conv window = last W-1 of
        (old window + canvas[:n_acc+1]). All fixed-shape GPU ops, no host sync."""
        states = self.diffusion_states
        N = _SELFSPEC_N; blk = self.canvas_length; nd = logits_3d.shape[0]; dev = logits_3d.device
        canvas = states.canvas[decode_slots]                            # [nd, blk]
        k = states.sspec_k[decode_slots]                                # [nd]
        j = torch.arange(N - 1, device=dev)
        _ss_sampling = _SS_SAMPLE and _SS_SPECSHAPE and not self._rs_failed
        if _ss_sampling:
            out = self._ss_sample_verify(input_batch, logits_3d, decode_slots, decode_idx, canvas, k, N, j)
            _ss_sampling = out is not None
        if _ss_sampling:
            n_acc, next_tok, new_specs, sidx, n_ok = out
            all_acc = n_acc == k
            acc = (j[None, :] < n_acc[:, None]).to(torch.int64)
        else:
            am = logits_3d.argmax(dim=-1)                                   # [nd, blk] (exact-greedy path only)
            match = (am[:, : N - 1] == canvas[:, 1:N]) & (j[None, :] < k[:, None])
            acc = torch.cumprod(match.to(torch.int64), dim=1)
            n_acc = acc.sum(dim=1)                                          # [nd] in [0, k]
            all_acc = n_acc == k
            next_tok = am.gather(1, n_acc[:, None]).squeeze(1)              # clean or corrected
            sidx = n_acc[:, None] + 1 + j[None, :]                          # [nd, N-1] <= 2N-2
            new_specs = am.gather(1, sidx)
        if _SIGNALS_ON or _SS_CARRY_P > 0.0:
            # E1b: record the DRAFT-TIME confidence of each spec we are about to offer. Read from the
            # MASK-slot logits here, one forward before the forward that verifies them. p(argmax) via
            # logsumexp so no [nd, blk, vocab] softmax is materialized.
            with torch.no_grad():
                t2 = logits_3d.topk(2, dim=-1).values                   # [nd, blk, 2]
                lse_ = torch.logsumexp(logits_3d.float(), dim=-1)       # [nd, blk]
                pmax_ = (t2[..., 0].float() - lse_).exp()               # p(argmax) at every slot
                states.sspec_draft_p[decode_slots] = pmax_.gather(1, sidx)
        new_canvas = torch.full_like(canvas, self.mask_id)
        new_canvas[:, 0] = next_tok
        if _SS_CARRY and not _ss_sampling:
            # E2: on a rejection the drafts at old slots n_acc+2.. were produced in the PREVIOUS forward
            # from a clean prefix and never saw the rejected token, so re-offer them instead of going cold.
            # New slot i == old slot n_acc+1+i, hence the gather index n_acc+2+j (NOT sidx, which is off by
            # one for this purpose). torch.where evaluates both branches, so clamp: on all-accept rows the
            # index would run one past the canvas and read out of bounds inside a captured graph.
            cidx = (n_acc[:, None] + 2 + j[None, :]).clamp(max=blk - 1)
            carried = canvas.gather(1, cidx)                         # [nd, N-1]
            n_carry = (k - n_acc - 1).clamp(min=0)                   # survivors after the rejected slot
            keep = j[None, :] < n_carry[:, None]
            if _SS_CARRY_P > 0.0:
                # E2b: carry a survivor only if the model was confident when it DREW it. Ungated carry
                # (E2) removed 62% of cold steps and lost 24% of throughput: stale candidates were
                # offered, rejected, and the step that should have re-drafted produced one token and
                # more stale candidates. The gate keeps the survivors worth offering and lets the rest
                # go cold, which is the reset. Draft index of new slot i is n_acc+1+i.
                gidx = (n_acc[:, None] + 1 + j[None, :]).clamp(max=max(N - 2, 0))
                conf = states.sspec_draft_p[decode_slots].gather(1, gidx)
                # Truncate at the first low-confidence survivor: a MASK in the middle would break the
                # invariant that k counts REAL tokens in slots 1..k, and verification stops there anyway.
                ok = torch.cumprod((conf >= _SS_CARRY_P).to(torch.int64), dim=1).bool()
                keep = keep & ok
            carried = torch.where(keep, carried, torch.full_like(carried, self.mask_id))
            new_canvas[:, 1:N] = torch.where(all_acc[:, None], new_specs, carried)
        else:
            new_canvas[:, 1:N] = torch.where(all_acc[:, None], new_specs, torch.full_like(new_specs, self.mask_id))
        if _SS_SPECSHAPE:
            # spec-shape: slot 0 (bonus) was already emitted last step; emit accepted specs + next_tok
            # (next_tok becomes vLLM's last sampled token == next query slot 0 == new canvas[:, 0]).
            emit = canvas[:, 1:].clone()
            emit.scatter_(1, n_acc[:, None], next_tok[:, None])
            sampled[decode_idx, : blk - 1] = emit.to(sampled.dtype)
            num_sampled[decode_idx] = (1 + n_acc).to(num_sampled.dtype)
        else:
            # ---- emit t0 + accepted specs (these are the canvas inputs whose KV was just written)
            sampled[decode_idx] = canvas.to(sampled.dtype)
            num_sampled[decode_idx] = (1 + n_acc).to(num_sampled.dtype)
        # ---- persist GDN state after slot n_acc (ssm from the intermediate cache, conv window rebuilt)
        R = _TRIDA_GDN_READOUT
        if R.get("ss_win_clean") is not None:
            # Fix S: ssm state is NOT copied anywhere -- next step reads ring[cur][slot, n_acc] directly.
            states.sspec_prev_acc[decode_slots] = n_acc
            # conv window for the next step = last W-1 rows of (window, canvas[:n_acc+1]); ONE gather over all layers
            wc, pre = R["ss_win_clean"], R["ss_preconv"]
            W1 = wc.shape[2]; sl1 = decode_slots + 1
            seq = torch.cat([wc[:, sl1], pre[:, decode_slots]], dim=2)                     # [L, nd, W-1+blk, dim]
            gidx = (n_acc[:, None] + 1 + torch.arange(W1, device=dev)[None, :])           # [nd, W-1]
            gidx = gidx[None, :, :, None].expand(seq.shape[0], -1, -1, seq.shape[3])
            wc[:, sl1] = seq.gather(2, gidx).to(wc.dtype)
            states.sspec_par[decode_slots] = 1 - states.sspec_par[decode_slots]         # next read comes from the half just written
        # ---- next canvas
        states.canvas[decode_slots] = new_canvas
        states.argmax_canvas[decode_slots] = new_canvas
        if _SS_CARRY and not _ss_sampling:
            # k must equal the number of REAL tokens now sitting in slots 1..N-1, or a MASK id could be
            # "verified" and emitted. Invariant to preserve: k + 1 <= NUM_CLEAN (= N), else the kernel
            # verifies against block-end rather than token-causal readouts and losslessness breaks silently.
            # k must equal how many survivors we ACTUALLY re-offered, which the confidence gate may
            # have truncated below n_carry. Deriving it from keep covers both the gated and ungated case.
            n_kept = keep.to(torch.int64).sum(dim=1)
            new_k = torch.where(all_acc, torch.full_like(k, N - 1), n_kept)
            states.sspec_k[decode_slots] = new_k.clamp(max=N - 1)
            if _SIGNALS_ON:
                _DIAG["carry_n"] = _DIAG.get("carry_n", 0) + int((new_k * (~all_acc)).sum())
        else:
            _kfull = n_ok if _ss_sampling else torch.full_like(k, N - 1)     # gated drafts: k = offered drafts
            states.sspec_k[decode_slots] = torch.where(all_acc, _kfull, torch.zeros_like(k))
        if _SIGNALS_ON:
            # E1: does an IN-STEP signal predict acceptance? History is exhausted (lag-1 R^2 <= 0.14).
            # For each verify slot j (0..N-2) log the model's own view of its draft -- probability of the
            # DRAFTED token, entropy, and top1-top2 margin -- beside whether that slot was ultimately
            # accepted. `acc` is the cumulative-product accept mask, so acc[:, j] == 1 iff slot j survived.
            with torch.no_grad():
                lg = logits_3d[:, : N - 1, :].float()
                pr = torch.softmax(lg, dim=-1)
                top2 = pr.topk(2, dim=-1).values                      # [nd, N-1, 2]
                drafted = canvas[:, 1:N].clamp(min=0)
                p_draft = pr.gather(2, drafted[:, :, None]).squeeze(2)   # p(the token we drafted)
                ent = -(pr * (pr + 1e-9).log()).sum(-1)
                margin = top2[..., 0] - top2[..., 1]
                live = (j[None, :] < k[:, None])
                # dp = the confidence recorded when this draft was DRAWN (previous forward). This is
                # the causal signal. p_draft/ent/margin below come from the verify logits of THIS
                # forward and are therefore posterior to the accept decision, not predictors of it --
                # they are logged only as a sanity reference, never as a gate.
                dp = states.sspec_draft_p[decode_slots]
                sl_ = decode_slots.tolist(); ac_ = acc.tolist(); dp_ = dp.tolist()
                pd_ = p_draft.tolist(); en_ = ent.tolist(); mg_ = margin.tolist(); lv_ = live.tolist()
                for i in range(nd):
                    rec = _trace_rec(int(sl_[i])); sig = rec.setdefault("slot_signals", [])
                    for jj in range(N - 1):
                        if lv_[i][jj]:
                            sig.append([jj, round(dp_[i][jj], 5), round(en_[i][jj], 4),
                                        round(mg_[i][jj], 5), int(ac_[i][jj]),
                                        round(pd_[i][jj], 5)])
        if _TRACE_ON:
            ks = k.tolist(); na = n_acc.tolist(); sl = decode_slots.tolist(); em = canvas.tolist(); nt = new_canvas[:, 0].tolist()
            for i in range(nd):
                rec = _trace_rec(int(sl[i])); rec.setdefault("spec_steps", []).append([ks[i], na[i]])
                rec["blocks"].append({"ids": em[i][: 1 + na[i]], "rounds": 0, "next_seed": nt[i], "r": []})
        if _DBG_COUNT_FWD or _DBG_TRACK_REJECTS:
            _DIAG["step"] += 1; _DIAG["last"] = "decode"; _DIAG["commit"] += 1
            print(f"[FWD] req={_DIAG['req']} prefill={_DIAG['prefill']} denoise=0 commit={_DIAG['commit']} "
                  f"total={_DIAG['prefill'] + _DIAG['commit']} step_type=selfspec", flush=True)
        self._write_drafts(all_slots)
        if _DBG_TIME_PHASES and "t_step0" in _DIAG:
            torch.cuda.synchronize(); _t3 = _time.perf_counter()
            _t0 = _DIAG["t_step0"]; _t1 = _DIAG.get("t_snap1", _t0); _t2 = _DIAG.get("t_samp0", _t1)
            print(f"[TIME] snap_ms={1000*(_t1-_t0):.2f} fwd_ms={1000*(_t2-_t1):.2f} "
                  f"samp_ms={1000*(_t3-_t2):.2f} step_ms={1000*(_t3-_t0):.2f} "
                  f"gdn_ms={_DIAG.get('gdn_ms', 0.0):.2f} gdn_n={_DIAG.get('gdn_n', 0)}", flush=True)
            _DIAG["gdn_ms"] = 0.0; _DIAG["gdn_n"] = 0
        return self._build_output(input_batch, sampled, num_sampled, per_req_nlogits_np)

    def _build_output(
        self, input_batch, sampled, num_sampled, per_req_nlogits_np
    ) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        self._query_lens.np[:num_reqs] = np.diff(
            input_batch.query_start_loc_np[: num_reqs + 1]
        )
        self._num_logits.np[:num_reqs] = per_req_nlogits_np
        self._query_lens.copy_to_uva()
        self._num_logits.copy_to_uva()
        num_rejected = _compute_num_rejected(
            self._num_logits.gpu[:num_reqs],
            num_sampled,
            input_batch.query_start_loc[: num_reqs + 1],
        )
        return SamplerOutput(
            sampled_token_ids=sampled,
            logprobs_tensors=None,
            num_nans=None,
            num_sampled=num_sampled,
            num_rejected=num_rejected,
        )

    # -- main entry point ---------------------------------------------------
    def __call__(self, logits, input_batch, draft_logits=None) -> SamplerOutput:
        num_reqs = input_batch.num_reqs
        device = logits.device
        states = self.diffusion_states
        CL = self.canvas_length

        if _DBG_TIME_PHASES:
            torch.cuda.synchronize(); _DIAG["t_samp0"] = _time.perf_counter()

        if _TRACE_ON:
            _now = _time.perf_counter(); _prev = _DIAG.get("trace_t")
            _DIAG["trace_t"] = _now
            _typ = "prefill" if input_batch.num_draft_tokens == 0 else "decode"
            for sl in input_batch.idx_mapping_np[:num_reqs].tolist():
                rec = _trace_rec(int(sl))
                if _typ == "prefill":
                    rec["prefill_fwd"] += 1
                if _prev is not None:
                    rec["steps"].append([_typ, round(1000 * (_now - _prev), 3)])
        if input_batch.num_draft_tokens == 0:
            if _DBG_COUNT_FWD or _DBG_TRACK_REJECTS:
                # A prefill following a decode step = a NEW request: reset.
                # (Consecutive prefill steps = chunked prefill of one request.)
                if _DIAG["last"] != "prefill":
                    _DIAG["req"] += 1
                    _DIAG.update(prefill=0, denoise=0, commit=0, step=0)
                    _DIAG["stash"].clear(); _DIAG["blk"].clear()
                _DIAG["prefill"] += 1; _DIAG["last"] = "prefill"
                print(f"[FWD] req={_DIAG['req']} prefill={_DIAG['prefill']} "
                      f"denoise=0 commit=0 total={_DIAG['prefill']} step_type=prefill",
                      flush=True)
            return self._handle_prefill(input_batch, logits)

        slots_np = input_batch.idx_mapping_np[:num_reqs]
        per_req_nlogits_np = np.diff(input_batch.cu_num_logits_np[: num_reqs + 1])
        # Mixed-batch fix (C>1): a request finishing its prompt in the same forward as other
        # requests' canvases shows up here with nlogits=1 (its last prompt logit), NOT 0. Classify
        # rows by the scheduler's own prefill flag so the prompt row is finished (seed -> bonus,
        # init_block -> ring import next step) instead of being mistaken for a canvas.
        is_pref_np = input_batch.is_prefilling_np[:num_reqs].astype(bool)
        decode_indices_np = np.where((per_req_nlogits_np > 0) & ~is_pref_np)[0]
        prefill_indices_np = np.where((per_req_nlogits_np == 0) | is_pref_np)[0]
        decode_slots_np = slots_np[decode_indices_np]

        fin = None
        if len(prefill_indices_np) > 0:
            fin = self._finish_prefills(input_batch, prefill_indices_np, logits)

        num_decode = len(decode_indices_np)
        sampled = self._sampled[:num_reqs]
        num_sampled = self._num_sampled[:num_reqs]
        sampled.zero_()
        num_sampled.zero_()
        if _SS_SPECSHAPE and fin is not None:
            di = async_copy_to_gpu(fin[0].astype(np.int64), device=logits.device)
            sampled[di, 0] = fin[1].to(sampled.dtype); num_sampled[di] = 1
        all_slots = input_batch.idx_mapping[:num_reqs]

        if num_decode == 0:
            self._write_drafts(all_slots)
            return self._build_output(input_batch, sampled, num_sampled, per_req_nlogits_np)

        if len(prefill_indices_np) > 0:
            # MIXED step (prompt rows alongside canvas rows). The GDN self-spec ring path is bypassed
            # on such forwards (stock fallback in _trida_gdn_selfspec_core), so the canvas rows' logits
            # were computed from the wrong recurrent state and must not be consumed. The prompt rows
            # were finished above; re-propose the canvases unchanged with num_sampled=0 so vLLM rewinds
            # and re-runs them next step on the ring (lossless; costs the canvas rows one step).
            if not _DIAG.get("mixed_step_warned"):
                _DIAG["mixed_step_warned"] = True
                print(f"[trida-sampler] mixed step (nlogits={per_req_nlogits_np.tolist()[:8]} "
                      f"prefill_rows={prefill_indices_np.tolist()[:8]}): prompts finished, canvases re-proposed", flush=True)
            self._write_drafts(all_slots)
            return self._build_output(input_batch, sampled, num_sampled, per_req_nlogits_np)

        # GUARD. These four are UvaBackedTensor, i.e. HOST-MAPPED memory sized by max_num_reqs.
        # Writing past the end silently corrupts the mapping, and the GPU then reads addresses
        # outside it -- which is exactly the fault the blocking repro produced:
        #   CUDA error: operation not supported on global/shared address space
        # A plain numpy slice assignment past the end would raise, but `.gpu[:n]` on an
        # over-long view would not, so check the bound explicitly and name it.
        _uva_cap = self._decode_slots.np.shape[0]
        if _GUARDS and num_decode > _uva_cap:
            raise RuntimeError(
                f"[trida-selfspec] num_decode={num_decode} exceeds UVA buffer capacity {_uva_cap} "
                f"(= max_num_reqs). decode_slots/_decode_idx/_query_lens/_num_logits are "
                f"host-mapped and sized at construction; a batch with more decode requests than "
                f"max_num_reqs reached this path.")
        self._decode_slots.np[:num_decode] = decode_slots_np
        self._decode_idx.np[:num_decode] = decode_indices_np
        self._decode_slots.copy_to_uva()
        self._decode_idx.copy_to_uva()
        decode_slots = self._decode_slots.gpu[:num_decode]
        decode_idx = self._decode_idx.gpu[:num_decode]

        if np.any(per_req_nlogits_np[decode_indices_np] != CL) or logits.shape[0] < num_decode * CL:
            # dummy / profiling batch (vLLM's warmup runs the sampler on synthetic batches whose
            # per-request logit counts are not canvases, e.g. 2 reqs x 4 logits under
            # --max-num-seqs 16 with the spec-shape decode_query_len): nothing to sample.
            if not _DIAG.get("dummy_warned"):
                _DIAG["dummy_warned"] = True
                print(f"[trida-sampler] non-canvas decode batch (nlogits={per_req_nlogits_np.tolist()[:8]} CL={CL}): "
                      f"treated as a dummy run", flush=True)
            self._write_drafts(all_slots)
            return self._build_output(input_batch, sampled, num_sampled, per_req_nlogits_np)
        logits_3d = logits[: num_decode * CL].reshape(num_decode, CL, -1)
        if _SELFSPEC:
            self._cur_logits = logits
            return self._selfspec_step(input_batch, logits_3d, decode_slots, decode_idx,
                                       all_slots, sampled, num_sampled, per_req_nlogits_np)

        # Split this step's decode slots into DENOISE (is_encoder_phase False) vs
        # COMMIT (True). A commit step runs a causal forward over the finalized
        # block: it captures the NEXT seed from the last (shifted) logit, EMITS
        # the block tokens, and re-inits the next [seed, MASK, ...] block.
        is_enc = states.is_encoder_phase[decode_slots]     # [nd]
        denoise_mask = ~is_enc                             # [nd] bool
        commit_mask = is_enc                               # [nd] bool

        if _DBG_COUNT_FWD or _DBG_TRACK_REJECTS:
            # One engine step == one model forward. Classify + count (diag sync).
            _DIAG["step"] += 1; _DIAG["last"] = "decode"
            _is_commit = bool(commit_mask.any().item())
            _DIAG["commit" if _is_commit else "denoise"] += 1
            _tot = _DIAG["prefill"] + _DIAG["denoise"] + _DIAG["commit"]
            print(f"[FWD] req={_DIAG['req']} prefill={_DIAG['prefill']} "
                  f"denoise={_DIAG['denoise']} commit={_DIAG['commit']} total={_tot} "
                  f"step_type={'commit' if _is_commit else 'denoise'}", flush=True)

        # DE-SYNC: run BOTH the denoise round and the commit logic over ALL
        # `num_decode` rows (fixed shape) and gate their writes by the row's
        # phase mask. A row is EITHER denoise or commit (masks are complementary),
        # so committing rows are untouched by the denoise path and vice-versa.
        # This removes the two `if mask.any().item():` host syncs + `nonzero()`
        # data-dependent shapes; the denoise-step runs first (writing only
        # denoise rows), then the commit block overwrites only the commit rows.

        # ---- DENOISE rows: shift + threshold in-place on the canvas ----------
        # `active=denoise_mask` gates all writes to denoise rows inside the step.
        _denoise_shift_step(
            logits_3d,
            decode_slots,
            states.canvas,
            states.argmax_canvas,
            states.done,
            states.step,
            states.is_encoder_phase,
            mask_id=self.mask_id,
            threshold=self.confidence_threshold,
            max_steps=int(states.max_denoising_steps),
            active=denoise_mask,
        )
        # denoise steps emit nothing (num_sampled stays 0 for these rows).

        # ---- COMMIT rows: capture next seed, EMIT block, re-init next block ---
        # All fixed-shape over `num_decode` rows; every write gated by commit_mask.
        cm = commit_mask                                   # [nd] bool
        cm_col = cm.unsqueeze(1)                            # [nd,1] for [nd,CL]

        # Next block seed = argmax(last position's shifted logit)
        # (LowConfidenceShift._capture_next_seeds: full_logits[base+B-1]).
        next_seed = _next_seed_from_logits(logits_3d)      # [nd]

        # Emission (SGLang `final[start_by_bid:]`): emit ALL CL positions =
        # [seed, denoised-tail]. The carried seed IS a generated token (only
        # re-input at pos 0 to drive the shift); emitting it makes the stream
        # contiguous (matches the SGLang oracle token stream). sampled/num_sampled
        # were zeroed above, so denoise rows keep 0 — masked write preserves that.
        canvas_all = states.argmax_canvas[decode_slots]    # [nd, CL]
        cur_sampled = sampled[decode_idx]                  # [nd, CL] (zeros)
        sampled[decode_idx] = torch.where(
            cm_col, canvas_all.to(sampled.dtype), cur_sampled
        )
        cur_nsamp = num_sampled[decode_idx]                # [nd] (zeros)
        num_sampled[decode_idx] = torch.where(
            cm, torch.full_like(cur_nsamp, CL), cur_nsamp
        )

        # Carry the seed (masked) and re-init the next [seed, MASK, ...] block
        # for commit rows only. init_block is inlined here as a masked update so
        # denoise rows keep their in-progress canvas untouched.
        cur_seed = states.seed[decode_slots]
        states.seed[decode_slots] = torch.where(cm, next_seed, cur_seed)
        seed_now = states.seed[decode_slots]               # [nd] (updated)
        cur_has = states.has_seed[decode_slots]
        states.has_seed[decode_slots] = torch.where(cm, torch.ones_like(cur_has), cur_has)

        # Fresh [seed, MASK, ...] block for all rows; select it where commit.
        fresh_canvas = torch.full_like(canvas_all, self.mask_id)  # [nd, CL]
        fresh_canvas[:, 0] = seed_now
        cur_canvas = states.canvas[decode_slots]
        states.canvas[decode_slots] = torch.where(cm_col, fresh_canvas, cur_canvas)
        cur_argmax = states.argmax_canvas[decode_slots]
        states.argmax_canvas[decode_slots] = torch.where(cm_col, fresh_canvas, cur_argmax)
        # done: [True, False, False, ...] for a fresh block (pos 0 seed committed).
        fresh_done = torch.zeros_like(states.done[decode_slots])
        fresh_done[:, 0] = True
        cur_done = states.done[decode_slots]
        states.done[decode_slots] = torch.where(cm_col, fresh_done, cur_done)
        # step -> 0 for re-inited rows.
        cur_step = states.step[decode_slots]
        states.step[decode_slots] = torch.where(cm, torch.zeros_like(cur_step), cur_step)
        # is_encoder_phase -> False (denoise the fresh block) for commit rows.
        # NOTE: `fresh_block` is intentionally not touched here. The old code did
        # init_block (fresh_block=True) then immediately set it False (net False),
        # but `fresh_block` is write-only dead state (never read anywhere), so
        # leaving it unchanged is behaviorally identical.
        cur_enc2 = states.is_encoder_phase[decode_slots]
        states.is_encoder_phase[decode_slots] = torch.where(
            cm, torch.zeros_like(cur_enc2), cur_enc2
        )

        if _TRACE_ON:
            C = torch.nonzero(cm, as_tuple=False).flatten().tolist()
            if C:
                ids_l = canvas_all[C].tolist(); ns_l = next_seed[C].tolist(); sl_l = decode_slots[C].tolist()
                for j, r in enumerate(C):
                    rec = _trace_rec(int(sl_l[j])); cur_b = rec["cur"]
                    rec["blocks"].append({"ids": ids_l[j], "rounds": len(cur_b["rounds"]),
                                          "next_seed": ns_l[j], "r": cur_b["rounds"]})
                    rec["cur"] = {"rounds": []}
        if _DBG_TRACK_REJECTS:
            # Emit the finalized block's committed ids (canvas_all = pre-re-init
            # block, pos 0 = carried seed) so the full output can be rebuilt as
            # concat over blocks and joined against the causal/AR ground truth.
            for r in torch.nonzero(cm, as_tuple=False).flatten().tolist():
                slot = int(decode_slots[r].item())
                blk = _DIAG["blk"].get(slot, 0)
                print(f"[BLK] req={_DIAG['req']} slot={slot} blk={blk} "
                      f"ids={canvas_all[r].tolist()}", flush=True)
                _DIAG["blk"][slot] = blk + 1
                _DIAG["stash"].pop(slot, None)

        # Feed the (updated / re-inited) canvas back into draft_tokens.
        self._write_drafts(all_slots)

        if _DBG_TIME_PHASES and "t_step0" in _DIAG:
            torch.cuda.synchronize(); _t3 = _time.perf_counter()
            _t0 = _DIAG["t_step0"]; _t1 = _DIAG.get("t_snap1", _t0); _t2 = _DIAG.get("t_samp0", _t1)
            print(f"[TIME] snap_ms={1000*(_t1-_t0):.2f} fwd_ms={1000*(_t2-_t1):.2f} "
                  f"samp_ms={1000*(_t3-_t2):.2f} step_ms={1000*(_t3-_t0):.2f} "
                  f"gdn_ms={_DIAG.get('gdn_ms', 0.0):.2f} gdn_n={_DIAG.get('gdn_n', 0)}",
                  flush=True)
            _DIAG["gdn_ms"] = 0.0; _DIAG["gdn_n"] = 0

        return self._build_output(input_batch, sampled, num_sampled, per_req_nlogits_np)


# ---------------------------------------------------------------------------
# Model: vLLM's native Qwen3.5 GDN backbone + a clean/noisy mode switch
# ---------------------------------------------------------------------------
class Qwen3_5ForBlockDiffusion(Qwen3_5ForCausalLM):
    """The Qwen3.5 GDN backbone, reused. Weights, embeddings, MoE, and GDN
    layers are inherited unchanged from Qwen3_5ForCausalLM; only the decode
    orchestration (via the ModelState) differs.

    NOTE (plan step 3): the GDN layers currently run causally in BOTH the clean
    and noisy passes. The full-attention layers already do per-request
    causal/bidirectional via prepare_attn's `causal` tensor, but the 3/4 GDN
    layers need the block-entry snapshot/restore two-stream re-scan for the
    noisy pass. Until that lands, denoise is lossy (but runs)."""

    @staticmethod
    def get_model_state_cls():
        return Qwen3_5DiffusionModelState

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)


# ---------------------------------------------------------------------------
# ModelState: the diffusion decode orchestration
# ---------------------------------------------------------------------------
class Qwen3_5DiffusionModelState(ModelState):
    def __init__(self, vllm_config, model, encoder_cache, device):
        super().__init__(vllm_config, model, encoder_cache, device)

        diffusion_config = vllm_config.diffusion_config
        canvas_length = diffusion_config.canvas_length if diffusion_config else 32
        self.gen_config = self.model_config.try_get_generation_config() or {}
        max_denoising_steps = (
            (diffusion_config.max_denoising_steps if diffusion_config else None)
            or self.gen_config.get("max_denoising_steps", canvas_length)
        )

        # Commit gate config: confidence threshold + mask id. Read from the
        # checkpoint's config / generation_config, with defaults matching our
        # SGLang LowConfidence threshold family (thresh 0.9) and the checkpoint's
        # block_diffusion.json mask_id.
        hf_config = self.model_config.hf_config
        ct = self.gen_config.get("confidence_threshold", None)
        if ct is None:
            ct = getattr(hf_config, "confidence_threshold", None)
        self.confidence_threshold = float(ct) if ct is not None else 0.9

        mid = getattr(hf_config, "mask_id", None)
        if mid is None:
            mid = self.gen_config.get("mask_id", None)
        self.mask_id = int(mid) if mid is not None else self.model_config.get_vocab_size() - 1

        if _SS_SPECSHAPE:
            assert canvas_length == 2 * _SELFSPEC_N - 2, (
                f"self-spec spec-shape needs canvas_length == 2N-2 = {2*_SELFSPEC_N-2}, got {canvas_length}")
            canvas_length = canvas_length + 1   # states/canvas hold [bonus, drafts...] = 2N-1 slots
        self.diffusion_states = TridaDiffusionStates(
            max_num_reqs=self.max_num_reqs,
            canvas_length=canvas_length,
            vocab_size=self.model_config.get_vocab_size(),
            max_denoising_steps=max_denoising_steps,
            device=device,
        )
        self.diffusion_states.mask_id = self.mask_id  # the real mask token id
        self._req_id_to_index: dict[str, int] = {}
        # INT32 on purpose: the FlashAttention builder does `causal.to(torch.int32)`; with a bool buffer that
        # allocates a fresh tensor every step and a FULL cuda graph replays the FA4 dynamic_causal kernel
        # against the capture-time pointer (rows >= 1 read stale memory). int32 makes it a no-op on this buffer.
        self._causal_buf = torch.zeros(self.max_num_reqs, dtype=torch.int32, device=device)
        self._inputs_embeds_buf = torch.zeros(
            self.max_num_tokens, self.inputs_embeds_size, dtype=self.dtype, device=device
        )

        # --- GDN two-stream snapshot/restore (DESIGN.md "the crux") ---------
        # The 3/4 gated-delta (GDN) layers keep a RECURRENT state (conv_state,
        # ssm_state) in their `.kv_cache`, indexed per request by the mamba
        # block table's column 0. Unlike full-attn KV, a bidirectional/causal
        # mask does nothing for a recurrent scan — each denoise pass mutates the
        # state IN PLACE (inplace_final_state=True), corrupting the clean
        # block-entry state. We snapshot the clean block-entry state and restore
        # it before every re-scan (denoise) and before the commit pass, so:
        #   - block entry (first denoise of a fresh block): SNAPSHOT
        #   - subsequent denoise re-scans + the commit pass: RESTORE first
        #   - the commit (clean causal) pass then advances state past the block.
        # These are populated lazily once kv_cache is bound (first real step).
        self._gdn_layers: list[MambaBase] | None = None   # discovered GDN layers
        self._gdn_group_id: int | None = None             # mamba kv-cache group (first layer's)
        self._gdn_layer_gids: list[int] = []              # per GDN layer: its mamba group id
        # Fix B (B1+B2): ONLY the conv state needs a block-entry snapshot. The
        # denoise override never writes the ssm cache (it copies the initial
        # state and runs with output_final_state=False), so the old per-step ssm
        # snapshot/restore was a semantic no-op. And the denoise conv now reads
        # its initial state from a per-slot WORK buffer (refreshed from the
        # snapshot once per step) instead of the kv-cache row, so the cache conv
        # row is never dirtied and nothing has to be restored before the commit.
        #   _conv_snap_all: [L, R, dim, W-1] clean block-entry conv state per slot
        #   _conv_work_all: [L, R, dim, W-1] scratch the denoise conv reads/writes
        self._conv_snap_all: torch.Tensor | None = None
        self._conv_work_all: torch.Tensor | None = None
        self._conv_dim_first: bool = False
        # per-slot phase last step: True == was committing/clean, False == denoise.
        # Initialize True so a freshly-added request's FIRST denoise is treated
        # as a block entry (snapshot).
        self._prev_encoder = torch.ones(self.max_num_reqs, dtype=torch.bool, device=device)
        # per-slot: does a valid clean-entry snapshot exist? Only then may we
        # restore. Prevents restoring a zero snapshot during prompt prefill
        # (which is a clean causal pass and must advance state untouched).
        self._snap_valid = torch.zeros(self.max_num_reqs, dtype=torch.bool, device=device)
        self._last_kv_cache_config = None
        self._gdn_ready = False
        self._ss_inter = None; self._ss_preconv = None; self._ss_win_clean = None; self._ss_win_kern = None   # self-spec buffers
        # persistent per-step index buffers (fixed pointers -> CUDA-graph replay safe); [:n] views are published
        Rn = self.max_num_reqs + 8   # headroom for padded rows under FULL cuda-graph capture sizes
        self._ss_buf_slots = torch.zeros(Rn, dtype=torch.int64, device=device)
        self._ss_buf_slots_i32 = torch.zeros(Rn, dtype=torch.int32, device=device)
        self._ss_buf_slots1_i32 = torch.zeros(Rn, dtype=torch.int32, device=device)
        self._ss_buf_ssm_idx = torch.zeros(Rn, dtype=torch.int32, device=device)
        self._ss_buf_write_idx = torch.zeros(Rn, dtype=torch.int32, device=device)
        self._num_accepted = torch.ones(self.max_num_reqs, dtype=torch.int32, device=device)   # last step's num_sampled per slot
        self._dbg_mode_logged = False

    # -- GDN two-stream state snapshot/restore ------------------------------
    def _install_hid_hooks(self) -> None:
        """TRIDA_DUMP_HID: fingerprint the residual stream (hidden+residual) of the
        last canvas_length rows after every decoder layer, plus their positions."""
        if getattr(self, "_hid_hooked", False):
            return
        self._hid_hooked = True
        layers = None
        for m in self.model.modules():
            if isinstance(m, nn.ModuleList) and len(m) >= 8 and hasattr(m[0], "input_layernorm"):
                layers = m
                break
        if layers is None:
            print("[trida-hid] decoder layers not found", flush=True)
            return

        def _fp(h, tag):
            T, CL = _DIAG.get("hid_T", 0), _DIAG.get("hid_CL", 0)
            if T < CL or CL == 0:
                return
            rows = h[T - CL:T].float()
            R = _state_proj(3000, rows.shape[-1], 1, rows.device)[0]  # [4, hidden]
            proj = [[float(x) for x in (R[:2] @ r).tolist()] for r in rows]
            print(f"{tag} proj={proj!r}", flush=True)

        def _pre0(mod, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            pos = kwargs.get("positions")
            if pos is None and len(args) >= 3:
                pos = args[2]
            T, CL = _DIAG.get("hid_T", 0), _DIAG.get("hid_CL", 0)
            if pos is not None and T >= CL:
                print(f"[HIDPOS] T={T} pos={pos[T - CL:T].tolist()!r}", flush=True)
            if hs is not None:
                _fp(hs, "[HID] layer=-1")

        def _mk(li):
            def _post(mod, args, out):
                hs, res = out
                _fp(hs + res if res is not None else hs, f"[HID] layer={li}")
            return _post

        layers[0].register_forward_pre_hook(_pre0, with_kwargs=True)
        for li, lay in enumerate(layers):
            lay.register_forward_hook(_mk(li))
        print(f"[trida-hid] hooked {len(layers)} decoder layers", flush=True)

    def _install_characterize(self) -> None:
        """E0 / TRIDA_CHARACTERIZE=1: characterize the MODEL before designing anything around it.

        This is the step ES-dLLM takes first and we skipped. They measure, on a base model, how much
        the hidden state and the confidence at each position MOVE between consecutive denoising
        iterations, find that most positions barely move, and only then design skipping around that
        redundancy. We went straight to decode-level outcomes (acceptance, tokens per forward) on a
        fine-tuned checkpoint, which entangles the model, the fine-tune recipe, our self-spec loop and
        our block size -- and that entanglement is exactly why our r moved with block size.

        Quantity, theirs (Eq. 1, second term), per layer l and position i:
            d_{l,i} = || H_l,i(t) - H_l,i(t-1) ||_1  /  ( sqrt(dim) * || H_l,i(t-1) ||_2 )

        One structural difference we must not paper over: in their setting the canvas is a fixed
        sequence and position i means the same token every iteration. Our canvas SLIDES as tokens
        commit, so slot j is a different token each step. We therefore match rows by ABSOLUTE
        position id, not by slot, and only positions present in both forwards are compared.
        """
        if getattr(self, "_char_hooked", False):
            return
        self._char_hooked = True
        layers = None
        for m in self.model.modules():
            if isinstance(m, nn.ModuleList) and len(m) >= 8 and hasattr(m[0], "input_layernorm"):
                layers = m
                break
        if layers is None:
            print("[trida-char] decoder layers not found", flush=True)
            return
        prev: dict[int, dict[int, torch.Tensor]] = {}     # layer -> {abs_pos: row}
        _DIAG.setdefault("char", {})

        def _rows(h):
            T, CL = _DIAG.get("hid_T", 0), _DIAG.get("hid_CL", 0)
            if T < CL or CL == 0:
                return None
            return h[T - CL:T].detach().float()

        def _record(li, h):
            rows = _rows(h)
            pos = _DIAG.get("char_pos")
            if rows is None or pos is None or len(pos) != rows.shape[0]:
                k = ("no_rows" if rows is None else "no_pos" if pos is None else "len_mismatch")
                _DIAG["char_skip"] = _DIAG.get("char_skip", {})
                _DIAG["char_skip"][k] = _DIAG["char_skip"].get(k, 0) + 1
                return
            cur = {int(p): rows[i] for i, p in enumerate(pos)}
            old = prev.get(li)
            if old:
                shared = [q for q in cur if q in old]
                if shared:
                    a = torch.stack([cur[q] for q in shared])
                    b = torch.stack([old[q] for q in shared])
                    d = (a - b).abs().sum(-1) / (
                        (a.shape[-1] ** 0.5) * b.norm(dim=-1).clamp(min=1e-6))
                    st = _DIAG["char"].setdefault(li, [0, 0.0, 0, 0])   # n, sum, n_gt_005, n_gt_05
                    dl = d.tolist()
                    st[0] += len(dl); st[1] += float(sum(dl))
                    st[2] += sum(1 for x in dl if x > 0.05)
                    st[3] += sum(1 for x in dl if x > 0.5)
            prev[li] = cur

        def _pre0(mod, args, kwargs):
            pos = kwargs.get("positions")
            if pos is None and len(args) >= 3:
                pos = args[2]
            T, CL = _DIAG.get("hid_T", 0), _DIAG.get("hid_CL", 0)
            if _DIAG.get("char_pos") is None and pos is not None and T >= CL and CL:
                _DIAG["char_pos"] = [int(x) for x in pos[T - CL:T].tolist()]
            hs = args[0] if args else kwargs.get("hidden_states")
            if hs is not None:
                _record(-1, hs)

        def _mk(li):
            def _post(mod, args, out):
                hs, res = out
                _record(li, hs + res if res is not None else hs)
            return _post

        layers[0].register_forward_pre_hook(_pre0, with_kwargs=True)
        for li, lay in enumerate(layers):
            lay.register_forward_hook(_mk(li))
        print(f"[trida-char] E0 characterization hooked {len(layers)} layers "
              f"(normalized L1 of hidden-state drift between consecutive forwards, matched by "
              f"absolute position)", flush=True)

    def _discover_attn(self, kv_cache_config) -> None:
        """Diagnostic helper: find the full-attention layers (non-Mamba modules with
        a bound kv_cache), their kv-cache group id and page size. Lazy, once."""
        if getattr(self, "_attn_layers", None) is not None or kv_cache_config is None:
            return
        fwd_ctx = self.vllm_config.compilation_config.static_forward_context
        layers = []
        for m in fwd_ctx.values():
            if isinstance(m, MambaBase):
                continue
            kc = getattr(m, "kv_cache", None)
            if kc is None:
                continue
            t = kc[0] if isinstance(kc, (list, tuple)) else kc
            if torch.is_tensor(t) and t.dim() == 4:
                layers.append(m)
        gid, page = None, None
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            if not isinstance(group.kv_cache_spec, MambaSpec):
                gid, page = i, int(group.kv_cache_spec.block_size)
                break
        if not layers or gid is None:
            return
        self._attn_layers, self._attn_group_id, self._attn_page = layers, gid, page
        print(f"[trida-kv] attn layers={len(layers)} group={gid} page={page} "
              f"kv shape={tuple((layers[0].kv_cache[0] if isinstance(layers[0].kv_cache,(list,tuple)) else layers[0].kv_cache).shape)}", flush=True)

    def _discover_gdn(self, kv_cache_config) -> None:
        """Lazily find the GDN layers, their mamba kv-cache group, and allocate
        per-slot snapshot buffers. Runs once, after kv_cache is bound."""
        fwd_ctx = self.vllm_config.compilation_config.static_forward_context
        named = [(nm, m) for nm, m in fwd_ctx.items() if isinstance(m, MambaBase)]
        layers = [m for _, m in named]
        # Only ready once each layer's kv_cache tuple is bound to real tensors.
        if not layers or any(getattr(l, "kv_cache", None) is None for l in layers):
            return
        try:
            conv0, ssm0 = layers[0].kv_cache[0], layers[0].kv_cache[1]
        except (IndexError, TypeError):
            return
        # mamba kv-cache group PER LAYER (state row = block_table[group][:, 0]).
        # vLLM's hybrid allocator splits the 24 GDN layers across SEVERAL mamba
        # groups (here 3: layers 0,3,6,.. / 1,4,7,.. / 2,5,8,..), each with its
        # own block table and hence its own state row. Using one group's row for
        # every layer (the original bug) restored only 8/24 layers; the other 16
        # kept the denoise-dirtied conv state -> wrong commit K/V, 49.5% GSM8K.
        name_to_gid: dict[str, int] = {}
        for i, group in enumerate(kv_cache_config.kv_cache_groups):
            if isinstance(group.kv_cache_spec, MambaSpec):
                for nm in group.layer_names:
                    name_to_gid[nm] = i
        gids = [name_to_gid.get(nm) for nm, _ in named]
        if not gids or any(g is None for g in gids):
            print(f"[trida-gdn] WARNING: could not map every GDN layer to a mamba "
                  f"kv-cache group: {gids}", flush=True)
            return
        gid = gids[0]
        R = self.max_num_reqs
        self._gdn_layers = layers
        self._gdn_group_id = gid
        self._gdn_layer_gids = gids
        from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
        self._conv_dim_first = bool(is_conv_state_dim_first())
        c0 = layers[0].kv_cache[0]
        # Buffers use the kv-cache row layout VERBATIM ((W-1, dim) here), and the
        # override hands the kernel the same transposed view the stock path uses
        # (`.transpose(-1, -2)` when not dim-first). The conv kernel takes the
        # state strides as tl.constexpr; keeping the exact layout/strides of the
        # production call keeps the exact kernel specialization.
        # R+1 rows: row 0 is never used. vLLM's conv kernel treats state index 0
        # as the NULL block (reserved block id) and SKIPS the sequence, so the
        # work buffer is indexed by slot+1.
        self._conv_snap_all = torch.zeros((len(layers), R + 1, *c0.shape[1:]), dtype=c0.dtype, device=c0.device)
        self._conv_work_all = torch.zeros_like(self._conv_snap_all)
        _TRIDA_GDN_READOUT["conv_dim_first"] = self._conv_dim_first
        self._gdn_ready = True
        print(
            f"[trida-gdn] two-stream snapshot/restore ready: {len(layers)} GDN "
            f"layers, mamba groups per layer {gids}, conv{tuple(conv0.shape)} ssm{tuple(ssm0.shape)}",
            flush=True,
        )

    def _gdn_snapshot_restore(self, input_batch, block_tables) -> None:
        """Snapshot clean block-entry state / restore it before denoise+commit.

        Called from prepare_attn, BEFORE the forward. Uses the per-request
        clean(commit)/noisy(denoise) flag and its previous-step value to detect
        block entry. All in-place index_copy on GPU; no host sync."""
        if not self._gdn_ready:
            self._discover_gdn(self._last_kv_cache_config)
            if not self._gdn_ready:
                return
        n = input_batch.num_reqs
        if n == 0:
            return
        slots = input_batch.idx_mapping[:n].long()          # active slot ids
        is_enc = self.diffusion_states.is_encoder_phase[slots]   # commit? (this step)
        prev_enc = self._prev_encoder[slots]                # committing last step?

        denoise = ~is_enc                                   # [n]
        entry = denoise & prev_enc                          # [n] first denoise of a fresh block
        restore = torch.zeros_like(entry)                   # (diag only) nothing is restored anymore

        # per-group state rows; each GDN layer uses its OWN group's row.
        rows_by_gid = {g: block_tables[g][:n, 0].long() for g in set(self._gdn_layer_gids)}
        layer_rows = [rows_by_gid[g] for g in self._gdn_layer_gids]   # per layer [n]
        state_rows = layer_rows[0]  # (diag only) rows of the first layer's group
        if _DBG_DUMP_STATE:
            # Fingerprint the CURRENT cache state (pre-mutation) for every active row.
            seq = input_batch.seq_lens[:n].tolist()
            sl, ie, pe, en, rs_, sr = (slots.tolist(), is_enc.tolist(), prev_enc.tolist(),
                                       entry.tolist(), restore.tolist(), state_rows.tolist())
            for i in range(n):
                projs = []
                for li, layer in enumerate(self._gdn_layers):
                    r_li = int(layer_rows[li][i].item())
                    s = layer.kv_cache[1][r_li].float().flatten()
                    c = layer.kv_cache[0][r_li].float().flatten()
                    R = _state_proj(li, s.numel(), c.numel(), s.device)
                    projs.append([float(x) for x in (R[0] @ s).tolist()] + [float(x) for x in (R[1] @ c).tolist()])
                print(f"[STATE] slot={sl[i]} seqlen={seq[i]} is_enc={int(ie[i])} prev_enc={int(pe[i])} "
                      f"entry={int(en[i])} restore={int(rs_[i])} row={sr[i]} proj={projs!r}", flush=True)
            if os.environ.get("TRIDA_DUMP_KV") == "1":
                # Attention-KV fingerprints: (a) the just-committed block = positions
                # [seqlen-2*CL, seqlen-CL) (the current canvas occupies the last CL);
                # (b) a fixed early block at TRIDA_KV_POS0 (default 92) to detect
                # later-block writes clobbering earlier positions.
                self._discover_attn(self._last_kv_cache_config)
                if getattr(self, "_attn_layers", None):
                    CLn = int(self.diffusion_states.canvas_length); bs_ = self._attn_page
                    bt = block_tables[self._attn_group_id]
                    p0 = int(os.environ.get("TRIDA_KV_POS0", "92"))
                    for i in range(n):
                        for tag, lo in (("KV", seq[i] - 2 * CLn), ("KV0", p0)):
                            if lo < 0 or lo + CLn > seq[i] - CLn + CLn:  # need positions < seqlen
                                continue
                            poss = list(range(lo, lo + CLn)); fps = []
                            for li, lay in enumerate(self._attn_layers):
                                t = lay.kv_cache[0] if isinstance(lay.kv_cache, (list, tuple)) else lay.kv_cache
                                hd = t.shape[-1] // 2
                                ks, vs = [], []
                                for p in poss:
                                    pg = int(bt[i, p // bs_].item()); off = p % bs_
                                    ks.append(t[pg, :, off, :hd].float().flatten()); vs.append(t[pg, :, off, hd:].float().flatten())
                                K = torch.cat(ks); V = torch.cat(vs)
                                R = _state_proj(1000 + li, K.numel(), V.numel(), K.device)
                                fps.append([float(x) for x in (R[0] @ K).tolist()] + [float(x) for x in (R[1] @ V).tolist()])
                            print(f"[{tag}] slot={sl[i]} seqlen={seq[i]} pos={lo}-{lo+CLn-1} proj={fps!r}", flush=True)
                            # per-POSITION fingerprints (2-dim K + 2-dim V each) so a
                            # single wrong position (e.g. seed / slot-mapping) is visible.
                            fpp = []
                            for li, lay in enumerate(self._attn_layers):
                                t = lay.kv_cache[0] if isinstance(lay.kv_cache, (list, tuple)) else lay.kv_cache
                                hd = t.shape[-1] // 2
                                per_pos = []
                                for p in poss:
                                    pg = int(bt[i, p // bs_].item()); off = p % bs_
                                    Kp = t[pg, :, off, :hd].float().flatten(); Vp = t[pg, :, off, hd:].float().flatten()
                                    R = _state_proj(2000 + li, Kp.numel(), Vp.numel(), Kp.device)
                                    per_pos.append([float(x) for x in (R[0][:2] @ Kp).tolist()] + [float(x) for x in (R[1] @ Vp).tolist()])
                                fpp.append(per_pos)
                            print(f"[{tag}P] slot={sl[i]} seqlen={seq[i]} pos={lo}-{lo+CLn-1} proj={fpp!r}", flush=True)

        # BLOCK-ENTRY SNAPSHOT (conv only). One tiny host sync decides whether
        # any active slot enters a block this step (~1 in 3 steps at bd4); on
        # those steps copy each layer's clean cache conv row into the per-slot
        # snapshot, in the kernel's (dim, W-1) layout. No restore path exists
        # anymore: the denoise conv works on _conv_work_all (see prepare_attn /
        # _trida_gdn_forward_core), so the cache row stays clean until the
        # commit pass advances it.
        if bool(entry.any().item()):
            e_slots = slots[entry]
            for li, layer in enumerate(self._gdn_layers):
                rows = layer_rows[li][entry]
                self._conv_snap_all[li, e_slots + 1] = layer.kv_cache[0].index_select(0, rows)  # cache layout, slot+1

        self._prev_encoder[slots] = is_enc

    def get_supported_generation_tasks(self):
        return ("generate",)

    # -- request lifecycle --------------------------------------------------
    def add_request(self, req_index: int, new_req_data: Any) -> None:
        self._req_id_to_index[new_req_data.req_id] = req_index
        self.diffusion_states.add_request(req_index)
        # Fresh request: its first denoise (after prefill) is a block entry, so
        # mark last-phase as clean/encoder to trigger the snapshot then.
        self._prev_encoder[req_index] = True
        if not new_req_data.req_id.startswith("_warmup_"):
            self.diffusion_states.prompt_len[req_index] = len(new_req_data.prompt_token_ids)
        if _TRACE_ON:
            _TRACE.pop(req_index, None)
            _TRACE_META[req_index] = {"req_id": new_req_data.req_id,
                                      "prompt_len": len(new_req_data.prompt_token_ids)}

    def remove_request(self, req_id: str) -> None:
        idx = self._req_id_to_index.pop(req_id, None)
        if idx is not None:
            if _TRACE_ON:
                _trace_flush(idx)
            self.diffusion_states.remove_request(idx)

    def get_mm_embeddings(self, scheduled_encoder_inputs, input_batch, req_states):
        return None  # text-only

    # -- per-step inputs ----------------------------------------------------
    def prepare_inputs(self, input_batch, req_states) -> dict[str, Any]:
        num_tokens = input_batch.num_tokens
        num_tokens_padded = input_batch.num_tokens_after_padding
        inputs_embeds = self._inputs_embeds_buf[:num_tokens_padded]
        input_ids = input_batch.input_ids[:num_tokens]
        inputs_embeds[:num_tokens].copy_(self.model.embed_input_ids(input_ids))
        # NOTE: no self-conditioning (our backbone has no SC MLP, unlike
        #   DiffusionGemma). The denoise pass reads masked canvas tokens directly.
        return {"inputs_embeds": inputs_embeds}

    def prepare_dummy_inputs(self, num_reqs: int, num_tokens: int) -> dict[str, Any]:
        return {"inputs_embeds": self._inputs_embeds_buf[:num_tokens]}

    # -- per-step attention metadata ---------------------------------------
    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings,
                     attn_groups, kv_cache_config, for_capture=False) -> dict[str, Any]:
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens

        n = input_batch.num_reqs
        slots = input_batch.idx_mapping[:n]
        self._causal_buf[:n] = self.diffusion_states.is_encoder_phase[slots]
        if n < num_reqs:
            self._causal_buf[n:num_reqs] = False
        causal = self._causal_buf[:num_reqs]

        # GDN two-stream (DESIGN.md crux): the 3/4 gated-delta layers have no
        # causal mask — they are a recurrent scan. Snapshot the clean block-entry
        # recurrent+conv state and restore it before each denoise re-scan and
        # before the commit pass, so the noisy pass never corrupts the committed
        # clean state. Skipped for capture/dummy runs.
        self._last_kv_cache_config = kv_cache_config
        if _DBG_TIME_PHASES and not for_capture:
            torch.cuda.synchronize(); _DIAG["t_step0"] = _time.perf_counter()
        if not for_capture and not _SELFSPEC:
            self._gdn_snapshot_restore(input_batch, block_tables)
        if _SELFSPEC:
            if not self._gdn_ready:
                self._discover_gdn(kv_cache_config)
            if self._gdn_ready and n > 0:
                # per-layer state rows for THIS batch (each GDN layer's own kv-cache group)
                rows_by_gid = {g: block_tables[g][:n, 0].long() for g in set(self._gdn_layer_gids)}
                _TRIDA_GDN_READOUT["layer_rows"] = [rows_by_gid[g] for g in self._gdn_layer_gids]
                _TRIDA_GDN_READOUT["gdn_layers"] = self._gdn_layers
                Ln = len(self._gdn_layers); Rn = self.max_num_reqs; Nn = _SELFSPEC_N
                if self._ss_inter is None:
                    c0 = self._gdn_layers[0].kv_cache[0]; s0 = self._gdn_layers[0].kv_cache[1]
                    blk = int(self.diffusion_states.canvas_length)
                    # Fix S (zero-copy commit): a 2-deep RING of intermediate states [2, L, R, N, HV, V, K];
                    # step t reads its initial state straight from ring[prev][slot, prev_acc] and writes
                    # its N intermediates into ring[cur][slot, :]. Nothing is copied back to the cache.
                    # layout [L, 2R, N, ...]: half `par` (rows par*R..par*R+R-1) holds each slot's current state.
                    # +1 dummy row (index 2R) for padded requests under FULL cuda-graph replay
                    self._ss_inter = torch.zeros((Ln, 2 * Rn + 1, Nn, *s0.shape[1:]), dtype=s0.dtype, device=s0.device)
                    # conv windows: win_clean = window before slot 0 (per slot, row slot+1; row 0 = null block);
                    # win_kern = scratch the conv kernel reads/writes (refreshed from win_clean each step).
                    self._ss_win_clean = torch.zeros((Ln, Rn + 2, *c0.shape[1:]), dtype=c0.dtype, device=c0.device)   # rows: 0 null, 1..R slots, R+1 dummy
                    self._ss_win_kern = torch.zeros_like(self._ss_win_clean)
                    qkv_dim = self._gdn_layers[0].conv1d.weight.shape[0]
                    self._ss_preconv = torch.zeros((Ln, Rn + 1, blk, qkv_dim), dtype=c0.dtype, device=c0.device)   # row R = dummy
                    print(f"[trida-selfspec] N={Nn} blk={blk} ring{tuple(self._ss_inter.shape)} "
                          f"win{tuple(self._ss_win_clean.shape)} preconv{tuple(self._ss_preconv.shape)}", flush=True)
                st = self.diffusion_states
                # import cache rows (written by the stock prompt prefill) for slots starting self-spec now
                # Host-side decision (CPU mirror), no device sync. The GPU-side mask is built only
                # when an import actually happens, which is once per request.
                slots_np = input_batch.idx_mapping_np[:n]
                need_np = ~st.sspec_started_np[slots_np]
                if (not for_capture) and bool(need_np.any()):
                    need = torch.from_numpy(need_np).to(slots.device)
                    e_slots = slots[need]
                    st.sspec_par[e_slots] = 0; st.sspec_prev_acc[e_slots] = 0
                    for li, layer in enumerate(self._gdn_layers):
                        rows = _TRIDA_GDN_READOUT["layer_rows"][li][need]
                        self._ss_inter[li, e_slots, 0] = layer.kv_cache[1].index_select(0, rows).to(self._ss_inter.dtype)   # half 0
                        self._ss_win_clean[li, e_slots + 1] = layer.kv_cache[0].index_select(0, rows)
                    st.sspec_started[e_slots] = True
                    st.sspec_started_np[slots_np[need_np]] = True
                if not _SS_FUSED:
                    self._ss_win_kern.copy_(self._ss_win_clean)                   # ONE copy for all layers (two-kernel path only)
                RR = _TRIDA_GDN_READOUT
                sl = slots.long()
                if for_capture:   # dummy but valid indices (slot i, parity 0, prev_acc 0)
                    par = torch.zeros_like(sl); pacc = torch.zeros_like(sl)
                else:
                    par = st.sspec_par[slots]; pacc = st.sspec_prev_acc[slots]
                RR["ss_inter_prev"] = self._ss_inter.view(Ln, (2 * Rn + 1) * Nn, *self._ss_inter.shape[3:])
                RR["ss_inter_cur"] = self._ss_inter
                # PERSISTENT index buffers (fixed pointers), [:n_pad] views; padded rows (FULL cuda-graph
                # capture/replay) point at dummy slots so they compute harmlessly.
                n_pad = int(num_reqs) if num_reqs > n else int(n)
                if _SHAPELOG:
                    _DIAG["shape_step"] = _DIAG.get("shape_step", 0) + 1
                    try:
                        ks = self.diffusion_states.sspec_k[sl].tolist()
                    except Exception:
                        ks = "?"
                    print(f"[SHAPE] step={_DIAG['shape_step']} n={n} num_reqs={int(num_reqs)} "
                          f"n_pad={n_pad} tokens={int(input_batch.num_tokens)} "
                          f"capture={'yes' if for_capture else 'no'} k={ks}", flush=True)
                # GUARD. Self-spec dies under concurrency with an async CUDA illegal-access whose
                # traceback points at whatever API call happened to be next, which is useless for
                # diagnosis. These are the two ways this block can produce an out-of-bounds index;
                # checking them here turns a race into a named Python error at the real site.
                # Costs two int comparisons per step and no device sync.
                _cap = self._ss_buf_slots.shape[0]
                if _GUARDS and n_pad > _cap:
                    raise RuntimeError(
                        f"[trida-selfspec] n_pad={n_pad} exceeds index-buffer capacity {_cap} "
                        f"(= max_num_reqs + 8). A cuda-graph capture size larger than the headroom "
                        f"reached this path; raise the headroom or clamp capture sizes to "
                        f"max_num_reqs. n={n} num_reqs={int(num_reqs)} Rn={Rn}")
                if _GUARDS and n and int(sl.max()) >= Rn:   # int() is a device sync -- flag-gated
                    raise RuntimeError(
                        f"[trida-selfspec] slot index {int(sl.max())} >= Rn={Rn}; the conv window row "
                        f"would be slot+1={int(sl.max())+1} against {self._ss_win_clean.shape[1]} rows "
                        f"and the ring row against {self._ss_inter.shape[1]}. Buffers were allocated "
                        f"for max_num_reqs={self.max_num_reqs}.")
                if n_pad > 0:   # one launch: slots / slots+1 / ring read idx / ring write idx (+ pad rows)
                    _ss_index_prep_kernel[(triton.cdiv(n_pad, 128),)](
                        sl, par, pacc, self._ss_buf_slots, self._ss_buf_slots_i32, self._ss_buf_slots1_i32,
                        self._ss_buf_ssm_idx, self._ss_buf_write_idx, n, n_pad, Rn, Nn, BLOCK=128)
                RR["ss_ssm_idx"] = self._ss_buf_ssm_idx[:n_pad]
                RR["ss_inter_write_idx"] = self._ss_buf_write_idx[:n_pad]
                RR["ss_n"] = n_pad
                # mixed batch (prompt chunk + canvases) -> fused path must not run (rows are not all canvases)
                RR["ss_mixed"] = (not for_capture) and int(input_batch.num_tokens) != int(n) * int(self.diffusion_states.canvas_length)
                RR["ss_win_kern"] = self._ss_win_kern
                RR["ss_win_clean"] = self._ss_win_clean
                RR["ss_preconv"] = self._ss_preconv
                RR["ss_slots"] = self._ss_buf_slots[:n_pad]
                RR["ss_slots_i32"] = self._ss_buf_slots_i32[:n_pad]
                RR["ss_slots1_i32"] = self._ss_buf_slots1_i32[:n_pad]              # conv window rows (0 = null block)
                if _DBG_DUMP_STATE:
                    # [SSTATE]: per-layer fingerprint of the persisted ssm+conv rows at step start
                    # (same projection as [STATE]); compare against a fresh prefill of the same prefix.
                    seq = input_batch.seq_lens[:n].tolist(); sl = slots.tolist()
                    for i in range(n):
                        projs = []
                        for li, layer in enumerate(self._gdn_layers):
                            r_li = int(_TRIDA_GDN_READOUT["layer_rows"][li][i].item())
                            s_ = layer.kv_cache[1][r_li].float().flatten(); c_ = layer.kv_cache[0][r_li].float().flatten()
                            Rp = _state_proj(li, s_.numel(), c_.numel(), s_.device)
                            projs.append([float(x) for x in (Rp[0] @ s_).tolist()] + [float(x) for x in (Rp[1] @ c_).tolist()])
                        print(f"[SSTATE] slot={sl[i]} seqlen={seq[i]} k={int(self.diffusion_states.sspec_k[sl[i]].item())} proj={projs!r}", flush=True)
        if (_DBG_DUMP_HID or _CHARACTERIZE) and not for_capture:
            if _CHARACTERIZE:
                # E0 needs ABSOLUTE positions to match a row across consecutive forwards. Run 3894
                # installed the hooks and still recorded nothing because the layer-0 pre-hook never
                # saw `positions`; take them here, where the runner definitely has them.
                try:
                    T = int(input_batch.num_tokens); CL = int(self.diffusion_states.canvas_length)
                    pos = getattr(input_batch, "positions", None)
                    if pos is None:
                        pos = getattr(input_batch, "positions_cpu", None)
                    if pos is not None and T >= CL and CL:
                        _DIAG["char_pos"] = [int(x) for x in pos[T - CL:T].tolist()]
                    else:
                        _DIAG["char_pos"] = None
                        _DIAG["char_skip_pos"] = _DIAG.get("char_skip_pos", 0) + 1
                except Exception as e:  # never break decoding for a diagnostic
                    _DIAG["char_pos"] = None
                    _DIAG["char_err"] = repr(e)[:120]
            _DIAG["hid_T"] = int(input_batch.num_tokens)
            _DIAG["hid_CL"] = int(self.diffusion_states.canvas_length)
            if _DBG_DUMP_HID:
                self._install_hid_hooks()
            if _CHARACTERIZE:
                self._install_characterize()
        if _DBG_TIME_PHASES and not for_capture:
            torch.cuda.synchronize(); _DIAG["t_snap1"] = _time.perf_counter()

        # Set the GDN readout mode read by the patched `_forward_core`. A denoise
        # step is a spec-decode batch (num_draft_tokens>0) whose active requests
        # are ALL in the noisy phase (is_encoder_phase False → causal False).
        # Then GDN uses BLOCK-END readout; otherwise stock token-causal.
        #
        # CUDA-GRAPH NOTE: under PIECEWISE capture the GDN op (`qwen_gdn_
        # attention_core`) is a splitting_op → runs EAGER between graphed
        # segments, so this Python-flag dispatch is honored at replay. Under
        # FULL the whole forward (incl. GDN) is captured once and the branch is
        # frozen at capture time — see _install_gdn_block_readout / report. We
        # keep the runtime flag logic below unchanged; PIECEWISE is the shipped
        # cuda-graph mode precisely because it preserves this per-step dispatch.
        _TRIDA_GDN_READOUT["li"] = 0
        if _SELFSPEC and n > 0 and (for_capture or input_batch.num_draft_tokens > 0):
            # self-spec decode step (also during CUDA-graph capture, so the captured GDN kernels are ours)
            _TRIDA_GDN_READOUT["mode"] = "selfspec"
            _TRIDA_GDN_READOUT["block_size"] = int(self.diffusion_states.canvas_length)
        elif not for_capture and input_batch.num_draft_tokens > 0 and n > 0:
            all_noisy = bool((causal[:n] == 0).all().item())  # tiny C-side sync (int32 flags)
            if all_noisy:
                _TRIDA_GDN_READOUT["mode"] = "denoise"
                _TRIDA_GDN_READOUT["block_size"] = int(self.diffusion_states.canvas_length)
                # Fix B2: the denoise conv reads/writes the per-slot WORK buffer,
                # refreshed from the clean block-entry snapshot in ONE copy for all
                # 24 layers (replaces the old per-layer restore into the cache).
                if self._conv_work_all is not None:
                    self._conv_work_all.copy_(self._conv_snap_all)
                    if _DBG_NAN:
                        _TRIDA_GDN_READOUT["conv_snap_absmax"] = float(self._conv_snap_all.float().abs().max())
                    _TRIDA_GDN_READOUT["conv_work"] = self._conv_work_all
                    _TRIDA_GDN_READOUT["slots_i32"] = (slots + 1).to(torch.int32)   # slot+1: index 0 == null block, skipped by the kernel
            else:
                _TRIDA_GDN_READOUT["mode"] = "commit"
            if not self._dbg_mode_logged:
                self._dbg_mode_logged = True
                print(f"[trida-gdn] first decode step: num_draft_tokens="
                      f"{input_batch.num_draft_tokens} n={n} all_noisy={all_noisy} "
                      f"mode={_TRIDA_GDN_READOUT['mode']}", flush=True)
        else:
            _TRIDA_GDN_READOUT["mode"] = "other"

        # DIAGNOSTIC: optionally force the full-attn layers causal for denoise
        # rows (GDN mode above already decided from the real `causal`). Passing a
        # forced-True tensor here isolates the FA backend's bidirectional path.
        attn_causal = causal
        if _DBG_FORCE_CAUSAL_DENOISE:
            attn_causal = torch.ones_like(causal)

        msam = None
        if _SS_SPECSHAPE:
            # spec-decode metadata (mirrors vLLM's MambaHybridModelState): which rows are spec decodes
            # (query == drafts + 1) and how many tokens each accepted last step.
            is_pref = torch.zeros(num_reqs, dtype=torch.bool)
            is_pref[:n] = torch.from_numpy(input_batch.is_prefilling_np[:n])
            n_acc_t = None; n_draft_cpu = None
            if not for_capture and self.vllm_config.num_speculative_tokens > 0:
                n_acc_t = self._num_accepted.new_ones(num_reqs)
                n_acc_t[:n] = self._num_accepted[input_batch.idx_mapping[:n]]
                nd_np = np.full(num_reqs, -1, dtype=np.int32)
                ndr = input_batch.num_draft_tokens_per_req
                if ndr is not None:
                    is_dec = input_batch.num_scheduled_tokens[:n] == ndr[:n] + 1
                    nd_np[:n] = np.where((ndr[:n] > 0) & is_dec, ndr[:n], -1)
                n_draft_cpu = torch.from_numpy(nd_np)
            msam = _TridaSpecAttnMetadata(is_prefilling=is_pref, num_accepted_tokens=n_acc_t, num_decode_draft_tokens_cpu=n_draft_cpu)
        md = build_attn_metadata(
            attn_groups=attn_groups, num_reqs=num_reqs, num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=torch.from_numpy(input_batch.query_start_loc_np),
            max_query_len=input_batch.num_scheduled_tokens.max().item(),
            seq_lens=input_batch.seq_lens, max_seq_len=self.max_model_len,
            block_tables=block_tables, slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config, causal=attn_causal,
            model_specific_attn_metadata=msam, for_cudagraph_capture=for_capture,
        )
        if _SS_SPECSHAPE and not for_capture and n > 0 and bool(input_batch.is_prefilling_np[:n].any()):
            # Mixed step (prompt rows + canvas rows): the canvas rows run through the STOCK spec-decode
            # GDN kernel, whose per-step state slots are block_table[row, 1..num_spec]. We never allocate
            # those slots (the spec shim is init-only), so those columns hold attention block ids or stale
            # entries -> the kernel reads/writes them as mamba state rows -> illegal memory access.
            # The canvas outputs of a mixed step are discarded (re-proposed on the ring next step), so
            # point columns 1.. at the null row 0: stores to row 0 are skipped, and row 0 is a valid read.
            # The mamba block table has ONE block per request, so the stock builder's
            # block_table[spec_rows, :num_spec+1] is a [nspec, 1] tensor while the kernel indexes
            # num_spec+1 columns (and takes max_query_len = size(-1)): give it a proper [nspec, 2N-1]
            # tensor -- column 0 = the request's real state row, columns 1.. = null row 0.
            _k1 = 2 * _SELFSPEC_N - 1
            for _m in (md.values() if isinstance(md, dict) else [md]):
                _sst = getattr(_m, "spec_state_indices_tensor", None)
                if _sst is not None and _sst.dim() == 2 and _sst.shape[1] != _k1:
                    _fixed = torch.zeros((_sst.shape[0], _k1), dtype=_sst.dtype, device=_sst.device)
                    _fixed[:, 0] = _sst[:, 0]
                    _m.spec_state_indices_tensor = _fixed
        return md

    # -- commit gate --------------------------------------------------------
    def custom_sampler(self, sampler: Any):
        """Return our lean confidence-shift / threshold commit sampler.

        Structured like DiffusionGemma's custom_sampler, but returns our
        TridaDiffusionSampler (top-1 prob + threshold), and wires NO
        self-conditioning (embed_weight / normalizer / vocab-shard) because
        our Qwen3.5 backbone has no SC MLP.
        """
        return (
            TridaDiffusionSampler(
                sampler=sampler,
                diffusion_config=self.vllm_config.diffusion_config,
                vocab_size=self.model_config.get_vocab_size(),
                diffusion_states=self.diffusion_states,
                confidence_threshold=self.confidence_threshold,
                mask_id=self.mask_id,
                req_states=sampler.req_states,
            ),
            None,
        )

    num_new_sampled_tokens_per_step: int = 1 if _SS_SPECSHAPE else 0

    def postprocess_state(self, idx_mapping, num_sampled, num_computed_tokens=None) -> None:
        if _SS_SPECSHAPE and not isinstance(num_sampled, int) and idx_mapping.shape[0]:
            self._num_accepted[idx_mapping] = torch.clamp(num_sampled.to(torch.int32), min=1)


# ---------------------------------------------------------------------------
# GDN block-END readout (FLARE two-stream) — patched onto the vLLM GDN layer.
#
# The stock `_forward_core` spec path uses `fused_sigmoid_gating_delta_rule_update`
# (TOKEN-causal readout: o_t = q_t @ S_t). For the noisy/denoise pass we instead
# need BLOCK-causal readout: scan the block to its end state S_end, then every
# block token reads S_end (o_t = (l2norm(q_t)/sqrt(K)) @ S_end). This is FLARE's
# two-stream mechanism — bit-matched to the SGLang
#   eval/sglang/srt/layers/attention/block_gdn/fused_recurrent.py
# via our verbatim kernel port `block_causal_readout.py` (same l2norm eps 1e-6,
# same 1/sqrt(head_k_dim) scale, same gating g=-exp(A_log)*softplus(a+dt_bias),
# beta=sigmoid(b), same GQA repeat_interleave). The gating + l2norm here reuse
# vLLM's `fused_post_conv_prep` (identical math: L2NORM_EPS=1e-6,
# SOFTPLUS_THRESHOLD=20.0), so q/k arrive already l2-normed and we call the
# kernel with use_qk_l2norm_in_kernel=False.
#
# We only override the pure-denoise step (mode=="denoise"): all requests are
# denoising their canvas as spec tokens (num_prefills==0, num_decodes==0). Every
# other step (prompt prefill, the clean/commit causal pass, non-spec decode)
# delegates to the stock `_forward_core` unchanged. State is NOT persisted during
# denoise (output_final_state=False) — the ModelState's snapshot/restore keeps
# ssm+conv at the clean block-entry state; the commit pass (stock, token-causal)
# advances the clean state past the block.
# ---------------------------------------------------------------------------
_ORIG_GDN_FORWARD_CORE = None


def _dump_gdn_inputs(self, li: int) -> None:
    """[GDNIN]: the ssm/conv state rows this GDN layer is about to READ (post
    restore), with the metadata row indices. Same projection as [STATE] so the
    numbers compare directly against the block-entry dump."""
    from vllm.forward_context import get_forward_context
    fwd = get_forward_context()
    md_raw = fwd.attn_metadata
    if md_raw is None:
        return
    md = md_raw[self.prefix]
    pr = getattr(md, "prefill_state_indices", None); ns = getattr(md, "non_spec_state_indices_tensor", None)
    sp = getattr(md, "spec_state_indices_tensor", None)
    pr_l = pr.tolist() if pr is not None else None; ns_l = ns.tolist() if ns is not None else None
    sp_l = sp[:, 0].tolist() if sp is not None else None
    hi = md.prefill_has_initial_state.tolist() if getattr(md, "prefill_has_initial_state", None) is not None else None
    rows = pr_l if pr_l else (ns_l or [])
    projs = []
    for r in rows[:4]:
        s_ = self.kv_cache[1][r].float().flatten(); c_ = self.kv_cache[0][r].float().flatten()
        R = _state_proj(li, s_.numel(), c_.numel(), s_.device)
        projs.append([float(x) for x in (R[0] @ s_).tolist()] + [float(x) for x in (R[1] @ c_).tolist()])
    print(f"[GDNIN] layer={li} mode={_TRIDA_GDN_READOUT.get('mode')} rows_prefill={pr_l} rows_nonspec={ns_l} "
          f"rows_spec={sp_l} has_init={hi} num_prefills={md.num_prefills} num_decodes={md.num_decodes} proj={projs!r}", flush=True)


def _trida_gdn_selfspec_core(self, mixed_qkv, b, a, core_attn_out, li: int):
    """Self-spec (AR-Trust) GDN step: canvas of blk=2N-1 tokens per request.
    conv on the cache row (its dirty window is fixed up by the sampler from the saved
    pre-conv rows); packed kernel with causal_mode=2, num_clean=N (clean/spec slots
    token-causal, MASK slots block-end) and intermediate states cached per slot so the
    sampler can persist the state after the accepted prefix. Nothing persisted here."""
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from .block_causal_readout import fused_recurrent_block_causal_gated_delta_rule_packed
    R = _TRIDA_GDN_READOUT
    N = _SELFSPEC_N; blk = int(R.get("block_size") or 0)
    slots = R["ss_slots"]; n = int(R.get("ss_n", slots.shape[0]))
    T = n * blk                                  # canvas tokens (independent of padding / metadata)
    if _SS_FUSED:
        if mixed_qkv.shape[0] < T or R.get("ss_mixed"):
            if not R.get("ss_mixed_warned"):
                R["ss_mixed_warned"] = True
                print(f"[trida-selfspec] WARNING mixed batch (T={mixed_qkv.shape[0]} n={n} blk={blk}): stock fallback (v1 limitation)", flush=True)
            if _FALLBACK_LOG:
                # Every fallback entry, with the state the stock kernel is about to be handed.
                # C=2 and C=8 die here while C=4 does not, on batches that look alike by shape, so
                # the difference has to be in this state. Layer 0 only, to keep it cheap.
                if li == 0:
                    md_ = None
                    try:
                        from vllm.forward_context import get_forward_context
                        md_ = get_forward_context().attn_metadata[self.prefix]
                    except Exception:
                        pass
                    _DIAG["fb_n"] = _DIAG.get("fb_n", 0) + 1
                    try:
                        sl_ = R["ss_slots"].tolist()
                    except Exception:
                        sl_ = "?"
                    print(f"[FALLBACK] #{_DIAG['fb_n']} tokens={mixed_qkv.shape[0]} n={n} blk={blk} "
                          f"T={T} slots={sl_} "
                          f"num_prefills={getattr(md_,'num_prefills',None)} "
                          f"num_decodes={getattr(md_,'num_decodes',None)} "
                          f"has_spec_masks={getattr(md_,'spec_sequence_masks',None) is not None} "
                          f"prefill_rows={getattr(md_,'prefill_state_indices',None) is not None} "
                          f"nonspec_rows={getattr(md_,'non_spec_state_indices_tensor',None) is not None} "
                          f"spec_rows={getattr(md_,'spec_state_indices_tensor',None) is not None}",
                          flush=True)
                    # E13 showed the booleans are identical between crashing and surviving runs, so
                    # print the actual ROW VALUES the stock kernel will index with, and the cache
                    # bounds they must stay inside. A row >= cache rows, or row 0 (the reserved null
                    # block) where a real row is expected, is the kind of thing that discriminates.
                    def _vals(t):
                        try:
                            return t.flatten()[:24].tolist() if t is not None else None
                        except Exception:
                            return "?"
                    try:
                        cache_rows = int(self.kv_cache[1].shape[0])
                    except Exception:
                        cache_rows = "?"
                    print(f"[FALLBACK-ROWS] #{_DIAG['fb_n']} cache_rows={cache_rows} "
                          f"spec={_vals(getattr(md_,'spec_state_indices_tensor',None))} "
                          f"nonspec={_vals(getattr(md_,'non_spec_state_indices_tensor',None))} "
                          f"prefill={_vals(getattr(md_,'prefill_state_indices',None))} "
                          f"qsl={_vals(getattr(md_,'query_start_loc',None))} "
                          f"spec_qsl={_vals(getattr(md_,'spec_query_start_loc',None))} "
                          f"nonspec_qsl={_vals(getattr(md_,'non_spec_query_start_loc',None))}",
                          flush=True)
            return _ORIG_GDN_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)
        mixed_qkv = mixed_qkv[:T]; b = b[:T]; a = a[:T]
    else:
        md = get_forward_context().attn_metadata[self.prefix]
        if md.num_prefills == 0 or md.num_decodes > 0 or md.spec_sequence_masks is not None or md.prefill_state_indices is None:
            return _ORIG_GDN_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)
        conv_cache = self.kv_cache[0]; ssm_state = self.kv_cache[1]
        conv_state = conv_cache if is_conv_state_dim_first() else conv_cache.transpose(-1, -2)
        T = md.num_actual_tokens
        mixed_qkv = mixed_qkv[:T]; b = b[:T]; a = a[:T]
        rows = md.prefill_state_indices
    if T != n * blk:
        # mixed batch (a prompt chunk alongside canvases): not supported by the v1 self-spec
        # path; fall back to the stock kernel (state persistence would be wrong -> warn loudly).
        if not R.get("ss_mixed_warned"):
            R["ss_mixed_warned"] = True
            print(f"[trida-selfspec] WARNING mixed batch T={T} n={n} blk={blk}: stock fallback (v1 limitation)", flush=True)
        return _ORIG_GDN_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)
    if _SS_FUSED:
        # Fix S2/S3: ONE launch -- conv (from x + clean window) + split/l2norm/gating + recurrence + readout,
        # writing the readout straight into core_attn_out and the raw pre-conv rows into the preconv buffer
        # (no separate copies). Reads the window from win_clean directly (never written).
        from .block_causal_readout import fused_selfspec_gdn_layer
        fused_selfspec_gdn_layer(
            mixed_qkv, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
            conv_weight=self.conv1d.weight, conv_bias=self.conv1d.bias,
            win=R["ss_win_clean"][li], win_idx=R["ss_slots1_i32"],
            ssm_states=R["ss_inter_prev"][li], ssm_idx=R["ss_ssm_idx"],
            inter=R["ss_inter_cur"][li], inter_idx=R["ss_inter_write_idx"],
            block_size=blk, num_clean=N, cache_steps=N, layer_key=li,
            out=core_attn_out[:T], pre_out=R["ss_preconv"][li], slot_idx=R["ss_slots_i32"],
        )
        return
    # (two-kernel path) pre-conv rows of the canvas for the window rebuild
    R["ss_preconv"][li, slots] = mixed_qkv.reshape(n, blk, -1).to(R["ss_preconv"].dtype)
    conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
    win = R["ss_win_kern"][li]                                   # (R+1, W-1, dim) scratch, refreshed per step
    win_view = win if is_conv_state_dim_first() else win.transpose(-1, -2)
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1), conv_weights, self.conv1d.bias, activation=self.activation,
        conv_states=win_view, has_initial_state=md.prefill_has_initial_state,
        cache_indices=R["ss_slots1_i32"], query_start_loc=md.prefill_query_start_loc, metadata=md,
    ).transpose(0, 1)
    if mixed_qkv.stride(-1) != 1:
        mixed_qkv = mixed_qkv.contiguous()
    out, _ = fused_recurrent_block_causal_gated_delta_rule_packed(
        mixed_qkv, a, b, A_log=self.A_log, dt_bias=self.dt_bias,
        ssm_states=R["ss_inter_prev"][li], cache_indices=R["ss_ssm_idx"],     # initial state = ring[prev][slot, prev_acc]
        block_size=blk, causal_mode=2, num_clean=N, output_final_state=False,
        intermediate_states_buffer=R["ss_inter_cur"][li], intermediate_state_indices=R["ss_inter_write_idx"],
        cache_steps=N, use_qk_l2norm_in_kernel=True,
    )
    core_attn_out[:T] = out.squeeze(0)


def _trida_gdn_forward_core(self, mixed_qkv, b, a, core_attn_out):
    from vllm.forward_context import get_forward_context
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import (
        causal_conv1d_update,
    )
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

    from .block_causal_readout import (
        fused_recurrent_block_causal_gated_delta_rule,
        fused_recurrent_block_causal_gated_delta_rule_packed,
    )

    _li = int(_TRIDA_GDN_READOUT.get("li", 0)); _TRIDA_GDN_READOUT["li"] = _li + 1  # layer index this step
    if _DBG_DUMP_STATE and _TRIDA_GDN_READOUT.get("mode") in ("denoise", "commit"):
        _dump_gdn_inputs(self, _li)
    if _TRIDA_GDN_READOUT.get("mode") == "selfspec":
        return _trida_gdn_selfspec_core(self, mixed_qkv, b, a, core_attn_out, _li)
    if _TRIDA_GDN_READOUT.get("mode") != "denoise":
        return _ORIG_GDN_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)

    fwd = get_forward_context()
    md_raw = fwd.attn_metadata
    if md_raw is None:
        return _ORIG_GDN_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)
    md = md_raw[self.prefix]
    # The diffusion denoise canvas is scheduled as a fresh chunk each step, which
    # the GDN metadata builder classifies as a PREFILL (num_prefills>0, no spec).
    # Take our block-END readout fast path only for that pure-prefill denoise
    # batch; anything else (spec/decode/mixed) falls back to the stock kernel.
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn
    if (
        md.num_prefills == 0
        or md.num_decodes > 0
        or md.spec_sequence_masks is not None
        or md.prefill_state_indices is None
    ):
        return _ORIG_GDN_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)
    if not _TRIDA_GDN_READOUT.get("dbg_fired"):
        _TRIDA_GDN_READOUT["dbg_fired"] = True
        print(f"[trida-gdn] BLOCK-CAUSAL readout firing (prefill-denoise): "
              f"num_prefills={md.num_prefills} T={md.num_actual_tokens} "
              f"bs={_TRIDA_GDN_READOUT.get('block_size')}", flush=True)

    if _DBG_TIME_GDN:
        torch.cuda.synchronize(); _gdn_t0 = _time.perf_counter()

    tp = self.tp_size
    num_k_heads = self.num_k_heads // tp
    num_v_heads = self.num_v_heads // tp
    block_size = int(_TRIDA_GDN_READOUT.get("block_size") or 0)

    self_kv_cache = self.kv_cache
    conv_state = (
        self_kv_cache[0]
        if is_conv_state_dim_first()
        else self_kv_cache[0].transpose(-1, -2)
    )
    ssm_state = self_kv_cache[1]
    num_actual_tokens = md.num_actual_tokens
    mixed_qkv = mixed_qkv[:num_actual_tokens]
    b = b[:num_actual_tokens]
    a = a[:num_actual_tokens]

    conv_weights = self.conv1d.weight.view(
        self.conv1d.weight.size(0), self.conv1d.weight.size(2)
    )
    prefill_rows = md.prefill_state_indices

    # Conv (prefill/chunk layout). Mirrors the stock prefill conv call; the
    # ModelState restores conv_state after the step so this doesn't leak.
    # Fix B2: initial conv state comes from the per-slot WORK buffer (a copy of
    # the clean block-entry snapshot, refreshed each step in prepare_attn), and
    # the kernel's final-state write lands there too -- the kv-cache conv row is
    # never touched by denoise, so no restore is needed before the commit.
    work = _TRIDA_GDN_READOUT.get("conv_work")
    if work is not None:
        conv_states_arg = work[_li]                       # [R, *cache row shape]
        if not _TRIDA_GDN_READOUT.get("conv_dim_first", False):
            conv_states_arg = conv_states_arg.transpose(-1, -2)   # -> (R, dim, W-1) view, same strides pattern as the stock cache view
        cache_idx_arg = _TRIDA_GDN_READOUT["slots_i32"]   # slot+1 ids, batch order (0 = null block → skipped)
    else:  # pre-discovery fallback (never on a real step)
        conv_states_arg, cache_idx_arg = conv_state, md.non_spec_state_indices_tensor
    if _DBG_NAN:
        _xin = mixed_qkv
        print(f"[NAN] li={_li} pre-conv: x_nan={int(torch.isnan(_xin).sum())} x_shape={tuple(_xin.shape)} x_strides={_xin.stride()} "
              f"work_shape={tuple(conv_states_arg.shape)} work_strides={conv_states_arg.stride()} work_nan={int(torch.isnan(conv_states_arg).sum())} "
              f"work_absmax={float(conv_states_arg.float().abs().max())} cache_idx={cache_idx_arg.tolist()} has_init={md.prefill_has_initial_state.tolist()} "
              f"qsl={md.prefill_query_start_loc.tolist()} snap_absmax={float(_TRIDA_GDN_READOUT.get('conv_snap_absmax', -1))}", flush=True)
    _alt = None
    if _DBG_NAN and _TRIDA_GDN_READOUT.get("dbg_ab", True):
        # Round-0 A/B/C, all channel-last like production:
        #  P = production call (cache view, cache row)   [dirties the cache row; round 0 only]
        #  W = work-buffer call as wired (work view, slot)
        #  D = work-buffer call on a fresh buffer filled from the cache row right now (bypasses snapshot timing)
        def _cl(t):  # channel-last (dim, T) copy of a [T, dim] view
            return t.clone().transpose(0, 1)
        _kw = dict(activation=self.activation, has_initial_state=md.prefill_has_initial_state,
                   query_start_loc=md.prefill_query_start_loc, metadata=md)
        _row = int(md.non_spec_state_indices_tensor[0]); _slot = int(cache_idx_arg[0])
        _pre_cache_row = conv_state[_row].float().clone()
        _pre_work_row = conv_states_arg[_slot].float().clone()
        _D_buf = torch.zeros_like(_TRIDA_GDN_READOUT["conv_work"][_li])          # cache layout (R, W-1, dim)
        _D_buf[_slot] = self.kv_cache[0][_row]                                     # fill from the clean cache row now
        _D_view = _D_buf if _TRIDA_GDN_READOUT.get("conv_dim_first", False) else _D_buf.transpose(-1, -2)
        _outW = causal_conv1d_fn(_cl(mixed_qkv), conv_weights, self.conv1d.bias, conv_states=conv_states_arg, cache_indices=cache_idx_arg, **_kw).transpose(0, 1)
        _outD = causal_conv1d_fn(_cl(mixed_qkv), conv_weights, self.conv1d.bias, conv_states=_D_view, cache_indices=cache_idx_arg, **_kw).transpose(0, 1)
        _outP = causal_conv1d_fn(_cl(mixed_qkv), conv_weights, self.conv1d.bias, conv_states=conv_state, cache_indices=md.non_spec_state_indices_tensor, **_kw).transpose(0, 1)
        print(f"[AB] li={_li} |P-W|={float((_outP.float()-_outW.float()).abs().max()):.4g} |P-D|={float((_outP.float()-_outD.float()).abs().max()):.4g} "
              f"|W-D|={float((_outW.float()-_outD.float()).abs().max()):.4g} P_absmax={float(_outP.float().abs().max()):.4g} "
              f"pre|cache_row-work_row|={float((_pre_cache_row-_pre_work_row).abs().max()):.4g} "
              f"work_written={float((conv_states_arg[_slot].float()-_pre_work_row).abs().max()):.4g} "
              f"cache_written={float((conv_state[_row].float()-_pre_cache_row).abs().max()):.4g} "
              f"D_written={float((_D_view[_slot].float()-_pre_cache_row).abs().max()):.4g} "
              f"work_strides={conv_states_arg.stride()} cache_strides={conv_state.stride()} dtype={conv_state.dtype}", flush=True)
        _alt = True
        if _li == 23:
            _TRIDA_GDN_READOUT["dbg_ab"] = False
    mixed_qkv = causal_conv1d_fn(
        mixed_qkv.transpose(0, 1),
        conv_weights,
        self.conv1d.bias,
        activation=self.activation,
        conv_states=conv_states_arg,
        has_initial_state=md.prefill_has_initial_state,
        cache_indices=cache_idx_arg,
        query_start_loc=md.prefill_query_start_loc,
        metadata=md,
    ).transpose(0, 1)

    # Fix B3a: ONE fused launch per layer -- the packed kernel (the SGLang
    # reference's own denoise kernel) does split + l2norm + gating + the
    # gated-delta recurrence + the mixed block-end readout (causal_mode=2,
    # num_clean=1) and reads the clean block-entry ssm state IN PLACE from the
    # kv-cache row (no gather copy; output_final_state=False so nothing is
    # written back). Replaces fused_post_conv_prep + views + gather + kernel.
    if _GDN_PACKED:
        if mixed_qkv.stride(-1) != 1:
            mixed_qkv = mixed_qkv.contiguous()
        T = mixed_qkv.shape[0]
        out, _ = fused_recurrent_block_causal_gated_delta_rule_packed(
            mixed_qkv, a, b,
            A_log=self.A_log, dt_bias=self.dt_bias,
            ssm_states=ssm_state,
            cache_indices=prefill_rows,
            block_size=block_size if block_size > 0 else T,
            causal_mode=2, num_clean=1,
            output_final_state=False,
            use_qk_l2norm_in_kernel=True,
        )
    else:
        # legacy path (pre-B3a), kept for A/B: prep kernel + gather + recurrent kernel
        q, k, v, g, beta = fused_post_conv_prep(
            conv_output=mixed_qkv, a=a, b=b, A_log=self.A_log, dt_bias=self.dt_bias,
            num_k_heads=num_k_heads, head_k_dim=self.head_k_dim, head_v_dim=self.head_v_dim,
            apply_l2norm=True, output_g_exp=False,
        )
        T = q.shape[0]
        q = q.view(1, T, num_k_heads, self.head_k_dim)
        k = k.view(1, T, num_k_heads, self.head_k_dim)
        v = v.view(1, T, num_v_heads, self.head_v_dim)
        g = g.view(1, T, num_v_heads)
        beta = beta.view(1, T, num_v_heads)
        initial_state = ssm_state[prefill_rows].contiguous()
        out, _ = fused_recurrent_block_causal_gated_delta_rule(
            q=q, k=k, v=v, g=g, beta=beta,
            block_size=block_size if block_size > 0 else T,
            causal_mode=2, num_clean=1,
            initial_state=initial_state, output_final_state=False,
            use_qk_l2norm_in_kernel=False,
            cu_seqlens=md.prefill_query_start_loc,
        )
    if _DBG_NAN:
        print(f"[NAN] li={_li} post-conv: mixed_nan={int(torch.isnan(mixed_qkv).sum())} mixed_absmax={float(mixed_qkv.float().abs().max())} "
              f"out_nan={int(torch.isnan(out).sum())} out_absmax={float(out.float().abs().max())} ssm_row_nan={int(torch.isnan(ssm_state[prefill_rows]).sum())}", flush=True)
    core_attn_out[:num_actual_tokens] = out.squeeze(0)
    if _DBG_TIME_GDN:
        torch.cuda.synchronize()
        _DIAG["gdn_ms"] = _DIAG.get("gdn_ms", 0.0) + 1000 * (_time.perf_counter() - _gdn_t0)
        _DIAG["gdn_n"] = _DIAG.get("gdn_n", 0) + 1


def _install_spec_builder_shim() -> None:
    """Self-spec spec-shape: make the GDN / FlashAttention metadata builders treat our
    (1 + 2N-2)-token canvas as a speculative decode (reorder threshold, spec tensors, FULL
    cuda-graph capture sizes) by presenting a minimal speculative_config ONLY during their
    __init__. No real speculator is created (vllm_config.speculative_config stays None)."""
    if not _SS_SPECSHAPE:
        return
    K = 2 * _SELFSPEC_N - 2

    class _Shim:
        num_speculative_tokens = K
        parallel_drafting = False
        method = "trida_selfspec"
        disable_padded_drafter_batch = True

        def __getattr__(self, name):
            return None

    def _find_cfg(args, kwargs):
        for v in list(args) + list(kwargs.values()):
            if hasattr(v, "speculative_config") and hasattr(v, "compilation_config"):
                return v
        return None

    def _wrap(cls):
        orig = cls.__init__

        def __init__(self, *args, **kwargs):
            cfg = _find_cfg(args, kwargs); injected = False
            if cfg is not None and cfg.speculative_config is None:
                cfg.speculative_config = _Shim(); injected = True
            try:
                orig(self, *args, **kwargs)
            finally:
                if injected:
                    cfg.speculative_config = None
        cls.__init__ = __init__

    # The scheduler hard-codes 0 sampled tokens per step for diffusion models (no bonus token);
    # in spec-shape every step (incl. the prompt prefill, which emits the seed) yields 1 + accepted.
    try:
        from vllm.v1.core.sched.scheduler import Scheduler
        _orig_sched_init = Scheduler.__init__

        def _sched_init(self, *args, **kwargs):
            _orig_sched_init(self, *args, **kwargs)
            self.num_sampled_tokens_per_step = 1
        Scheduler.__init__ = _sched_init
    except Exception as e:  # noqa: BLE001
        print(f"[trida-selfspec] scheduler shim not installed: {e}", flush=True)
    # FULL cuda-graph dispatch is shape-only (get_uniform_token_count): a PROMPT of exactly
    # 2N-1 tokens for one request looks like a uniform self-spec decode and would replay the
    # decode graph (fused self-spec GDN path) over prompt tokens -> garbage (seen at N=32,
    # prompt_len 63). Treat a batch as uniform decode only if EVERY scheduled request carries
    # draft tokens (a prefill row never does).
    try:
        from vllm.v1.worker.gpu import model_runner as _mr
        _orig_uniform = _mr.get_uniform_token_count
        _flag = {"all_drafts": False}

        def _uniform(num_reqs, num_tokens, max_query_len):
            if not _flag["all_drafts"]:
                return None
            return _orig_uniform(num_reqs, num_tokens, max_query_len)
        _mr.get_uniform_token_count = _uniform

        _orig_exec = _mr.GPUModelRunner.execute_model

        def _exec(self, scheduler_output, *args, **kwargs):
            try:
                sd = scheduler_output.scheduled_spec_decode_tokens
                _flag["all_drafts"] = bool(scheduler_output.num_scheduled_tokens) and all(
                    r in sd and len(sd[r]) > 0 for r in scheduler_output.num_scheduled_tokens)
            except Exception:  # dummy runs etc.
                _flag["all_drafts"] = False
            return _orig_exec(self, scheduler_output, *args, **kwargs)
        _mr.GPUModelRunner.execute_model = _exec
    except Exception as e:  # noqa: BLE001
        print(f"[trida-selfspec] uniform-dispatch shim not installed: {e}", flush=True)

    from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
    _wrap(GDNAttentionMetadataBuilder)
    try:
        from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadataBuilder
        _wrap(FlashAttentionMetadataBuilder)
    except Exception as e:  # noqa: BLE001
        print(f"[trida-selfspec] FA builder shim not installed: {e}", flush=True)
    print(f"[trida-selfspec] spec-shape builder shim installed (num_speculative_tokens={K})", flush=True)


def _bidir_future_mask(query, key, value, n, blk, n_clean, scale):
    """Attention of each MASK slot to strictly LATER mask slots only.

    query/key/value are vLLM's flat [T, H, D] with T == n * blk. Returns (out, lse) in [n, H, blk, D]
    and [n, H, blk]; rows with no later mask get lse = -inf so the merge leaves them untouched -- that
    is every verify row, which is why the verify half stays bit-identical to the causal result.
    """
    H, D = query.shape[1], query.shape[2]
    Hk = key.shape[1]
    q = query.view(n, blk, H, D).transpose(1, 2).float()                       # [n, H, blk, D]
    k = key.view(n, blk, Hk, D).transpose(1, 2).float()
    v = value.view(n, blk, Hk, D).transpose(1, 2).float()
    if Hk != H:                                                                # GQA: broadcast kv heads
        rep = H // Hk
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)
    idx = torch.arange(blk, device=q.device)
    is_mask = idx >= n_clean
    allow = (is_mask[:, None] & is_mask[None, :] & (idx[None, :] > idx[:, None]))
    sc = (q @ k.transpose(-1, -2)) * scale                                     # [n, H, blk, blk]
    sc = sc.masked_fill(~allow[None, None], float("-inf"))
    lse = torch.logsumexp(sc, dim=-1)                                          # [n, H, blk]
    p = torch.nan_to_num(torch.softmax(sc, dim=-1), nan=0.0)                   # all -inf rows -> 0
    return p @ v, lse


def _trida_attn_forward(self, layer, query, key, value, kv_cache, attn_metadata, output,
                        output_scale=None, output_block_scale=None):
    """FlashAttentionImpl.forward + the E3 bidirectional-mask correction (no-op unless enabled)."""
    active = _SS_BIDIR and _TRIDA_GDN_READOUT.get("mode") == "selfspec"
    blk = int(_TRIDA_GDN_READOUT.get("block_size") or 0)
    if not active or blk < 3 or query.dim() != 3:
        return _ORIG_ATTN_FORWARD(self, layer, query, key, value, kv_cache, attn_metadata,
                                  output, output_scale, output_block_scale)
    _BIDIR["on"] = True; _BIDIR["lse"] = None
    try:
        ret = _ORIG_ATTN_FORWARD(self, layer, query, key, value, kv_cache, attn_metadata,
                                 output, output_scale, output_block_scale)
    finally:
        _BIDIR["on"] = False
    lse = _BIDIR["lse"]; _BIDIR["lse"] = None
    T, H, D = query.shape
    # Bail to stock behaviour on anything unexpected: a chunked prefill mixed into the batch, a
    # cascade path that consumed the lse itself, or a ragged query. Silence is wrong here, so warn once.
    if lse is None or blk <= 0 or T % blk or lse.shape != (H, T):
        if not _BIDIR["warned"]:
            _BIDIR["warned"] = True
            print(f"[TRIDA_SS_BIDIR] inactive this step (lse={None if lse is None else tuple(lse.shape)}, "
                  f"T={T}, blk={blk}, H={H}) -- falling back to causal", flush=True)
        return ret
    n = T // blk
    n_clean = (blk + 1) // 2                      # blk = 2N-1  =>  N = (blk+1)//2; masks start at N
    scale = float(getattr(self, "scale", D ** -0.5))
    with torch.no_grad():
        o2, l2 = _bidir_future_mask(query, key, value, n, blk, n_clean, scale)
        o1 = output.view(n, blk, H, D).transpose(1, 2).float()                 # [n, H, blk, D]
        l1 = lse.view(H, n, blk).permute(1, 0, 2).float()                      # [n, H, blk]
        m = torch.maximum(l1, l2)
        w1 = (l1 - m).exp().unsqueeze(-1)
        w2 = (l2 - m).exp().unsqueeze(-1)
        merged = (o1 * w1 + o2 * w2) / (w1 + w2)
        output.copy_(merged.transpose(1, 2).reshape(T, H, D).to(output.dtype))
    return ret


_ORIG_ATTN_FORWARD = None


def _install_bidir_canvas():
    """Patch FlashAttentionImpl.forward and wrap flash_attn_varlen_func to surface the softmax LSE."""
    global _ORIG_ATTN_FORWARD
    if not _SS_BIDIR or _ORIG_ATTN_FORWARD is not None:
        return
    try:
        import vllm.v1.attention.backends.flash_attn as fam
    except Exception as e:                                                     # pragma: no cover
        print(f"[TRIDA_SS_BIDIR] cannot import the FA backend ({e}); staying causal", flush=True)
        return
    _ORIG_ATTN_FORWARD = fam.FlashAttentionImpl.forward
    fam.FlashAttentionImpl.forward = _trida_attn_forward

    _orig_fa = fam.flash_attn_varlen_func
    if not getattr(_orig_fa, "_trida_wrapped", False):
        def _wrapped(*a, **kw):
            # Only intercept while our patched forward is on the stack, and never override a caller
            # that already wants the lse (the cascade path does its own merge).
            if not _BIDIR["on"] or kw.get("return_softmax_lse"):
                return _orig_fa(*a, **kw)
            kw["return_softmax_lse"] = True
            r = _orig_fa(*a, **kw)
            if isinstance(r, tuple) and len(r) == 2:
                _BIDIR["lse"] = r[1]
                return r[0]
            return r
        _wrapped._trida_wrapped = True
        fam.flash_attn_varlen_func = _wrapped
    print(f"[TRIDA_SS_BIDIR] enabled: mask slots attend bidirectionally, verify slots stay causal", flush=True)


def _install_gdn_block_readout() -> None:
    global _ORIG_GDN_FORWARD_CORE
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention,
    )
    if getattr(QwenGatedDeltaNetAttention, "_trida_patched", False):
        return
    _ORIG_GDN_FORWARD_CORE = QwenGatedDeltaNetAttention._forward_core
    QwenGatedDeltaNetAttention._forward_core = _trida_gdn_forward_core
    QwenGatedDeltaNetAttention._trida_patched = True


# ---------------------------------------------------------------------------
# Registration (out-of-tree; no vLLM fork). Import this module before serving.
# ---------------------------------------------------------------------------
def register() -> None:
    from vllm import ModelRegistry
    ModelRegistry.register_model(
        "Qwen3_5ForBlockDiffusion",
        "vllm_native_diffusion.qwen3_5_diffusion:Qwen3_5ForBlockDiffusion",
    )
    _install_gdn_block_readout()
    _install_spec_builder_shim()


if __name__ == "__main__":
    register()
    print("registered Qwen3_5ForBlockDiffusion")
