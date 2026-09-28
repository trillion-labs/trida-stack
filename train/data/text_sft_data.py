"""Single-turn SFT data for the text-only block-diffusion trainer.

The block-diffusion response-only path requires ONE contiguous supervised (assistant) span per
sequence. Agentic chat data is multi-turn (user / assistant / tool roles, plus a top-level
``tools`` schema), so we render the whole conversation as the prompt and supervise
only the **final assistant turn** — a standard "SFT on the last turn" reduction that yields a single
contiguous response span. Earlier assistant turns become conditioning (``labels = -100``).

The chat rendering is tokenizer-driven (``apply_chat_template``), so this works for any base model
whose tokenizer ships a chat template (Qwen3, Tri-7B/Llama, …).
"""

import json
import multiprocessing as mp
import random

import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from train.hf_block_diffusion import bucketed_clean_len


def _normalize_messages(messages):
    """Coerce dataset messages into the shape chat templates expect, forwarding every field the
    tokenizer's ``apply_chat_template`` consumes so rendering is fully tokenizer-driven:

      - ``role`` + ``content``
      - ``reasoning_content`` — assistant "thinking" (Qwen3 renders it inside ``<think>…</think>``);
        it stays part of the assistant turn, so it is supervised like the rest of the response.
      - ``tool_calls`` / ``tool_call_id`` / ``name`` — tool-calling / tool-result fields.

    ``content`` that is a str / list / dict is passed through untouched (the template handles
    structured content); only other scalars are stringified. Unknown keys (``mask_loss``,
    ``quality_category``, …) are dropped."""
    out = []
    for m in messages:
        role = m.get("role") or m.get("from")
        content = m.get("content")
        if content is None:
            content = m.get("value")
        if content is None:
            content = ""
        if not isinstance(content, (str, list, dict)):
            content = str(content)
        msg = {"role": role, "content": content}
        # Pass through template-consumed fields when present (reasoning_content enables Qwen3 thinking).
        for key in ("reasoning_content", "tool_calls", "tool_call_id", "name"):
            if m.get(key):
                msg[key] = m[key]
        out.append(msg)
    return out


def _to_ids(enc):
    """apply_chat_template(tokenize=True) returns a bare id list on some transformers versions and a
    BatchEncoding/dict on others (5.x). Normalize to a flat list[int]."""
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    # unwrap a possible batch dim ([[...]] -> [...])
    if enc and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return list(enc)


def _apply_template(tokenizer, messages, tools, add_generation_prompt, keep_all_reasoning=False):
    """Render + tokenize one message list. No per-call tools fallback: the tools-vs-no-tools mode
    is chosen ONCE per example in ``build_sample`` so ``full`` and ``prompt`` always render in the
    same mode (otherwise they can diverge and break the prefix/label boundary).

    ``keep_all_reasoning`` is forwarded to the chat template (a no-op on stock templates; on the
    patched Qwen3 template it keeps ``<think>`` for non-last assistant turns that carry
    ``reasoning_content``). It must be identical across the ``full`` / ``prompt`` renders of one
    sample, else the rendered reasoning diverges and the prefix/label boundary breaks."""
    enc = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=add_generation_prompt,
        keep_all_reasoning=keep_all_reasoning,
    )
    return _to_ids(enc)


def build_sample(tokenizer, example, keep_all_reasoning=False):
    """Return ``{"input_ids": LongTensor[T], "labels": LongTensor[T]}`` supervising the final
    assistant turn, or ``None`` if the example has no usable assistant response.

    ``keep_all_reasoning`` controls whether earlier (history) assistant turns keep their ``<think>``
    reasoning in the conditioning prompt (see ``_apply_template``); default off = stock behavior."""
    messages = example.get("messages") or example.get("conversations")
    if not messages:
        return None
    messages = _normalize_messages(messages)

    # index of the final assistant turn
    last_asst = None
    for i in range(len(messages) - 1, -1, -1):
        if messages[i]["role"] == "assistant":
            last_asst = i
            break
    if last_asst is None:
        return None

    # Render prompt + full in the SAME tools mode (decide once): try with tools, and only if that
    # raises, fall back to rendering BOTH without tools. Never mix modes across the two calls.
    tools = example.get("tools")

    def render(tools_arg):
        full = _apply_template(tokenizer, messages[: last_asst + 1], tools_arg, False,
                               keep_all_reasoning=keep_all_reasoning)
        prompt = _apply_template(tokenizer, messages[:last_asst], tools_arg, True,
                                 keep_all_reasoning=keep_all_reasoning)
        return full, prompt

    try:
        full, prompt = render(tools)
    except Exception:
        try:
            full, prompt = render(None)
        except Exception:
            return None

    if not full or len(prompt) >= len(full):
        return None
    # Prefix stability: the response-label boundary at len(prompt) is only valid if the prompt
    # render is an exact token prefix of the full render. Chat templates aren't guaranteed
    # prefix-stable (thinking tokens, whitespace normalization, tool interleaving); if it isn't,
    # drop the sample rather than silently supervise prompt tokens / hide response tokens.
    if full[: len(prompt)] != prompt:
        return None
    ids = torch.tensor(full, dtype=torch.long)
    labels = ids.clone()
    labels[: len(prompt)] = -100
    if (labels != -100).sum() == 0:
        return None
    return {"input_ids": ids, "labels": labels}


def build_sample_multiturn(tokenizer, example, bd_size, keep_all_reasoning=False):
    """Supervise EVERY assistant turn of a conversation in ONE sequence (for the causally-ordered
    multi-turn block-diffusion forward), instead of only the last turn.

    Returns ``None`` if unusable, else a dict of LongTensor[T]:
        input_ids, labels (all assistant turns supervised; else -100),
        resp_block (GLOBAL, conversation-ordered block id, unique per (turn, bd-block); -1 off-resp),
        turn_id   (assistant-turn ordinal 0,1,...; -1 off-resp).

    Each assistant turn's content span is located by incremental prefix renders (the standard
    multi-turn label-masking trick): ``prompt_j = template(messages[:i_j], add_generation_prompt)``
    marks the turn's content start, ``upto_j = template(messages[:i_j+1])`` its end. Turns whose
    incremental render is not an exact token prefix of the full render are skipped (become clean
    conditioning) rather than mis-supervised.
    """
    messages = example.get("messages") or example.get("conversations")
    if not messages:
        return None
    messages = _normalize_messages(messages)

    asst_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not asst_idxs:
        return None

    tools = example.get("tools")

    def render_all(tools_arg):
        full = _apply_template(tokenizer, messages, tools_arg, False,
                               keep_all_reasoning=keep_all_reasoning)
        spans = []
        for i in asst_idxs:
            prefix = _apply_template(tokenizer, messages[:i], tools_arg, True,
                                     keep_all_reasoning=keep_all_reasoning)
            upto = _apply_template(tokenizer, messages[: i + 1], tools_arg, False,
                                   keep_all_reasoning=keep_all_reasoning)
            spans.append((prefix, upto))
        return full, spans

    try:
        full, spans = render_all(tools)
    except Exception:
        try:
            full, spans = render_all(None)
        except Exception:
            return None

    if not full:
        return None
    T = len(full)
    ids = torch.tensor(full, dtype=torch.long)
    labels = torch.full((T,), -100, dtype=torch.long)
    resp_block = torch.full((T,), -1, dtype=torch.long)
    turn_id = torch.full((T,), -1, dtype=torch.long)

    gblk = 0
    n_turns = 0
    for j, (prefix, upto) in enumerate(spans):
        lp, lu = len(prefix), len(upto)
        # prefix stability: both incremental renders must be exact token prefixes of `full`
        if lu > T or lp >= lu or full[:lp] != prefix or full[:lu] != upto:
            continue
        pos = torch.arange(lp, lu)
        labels[pos] = ids[pos]
        turn_id[pos] = j
        within = torch.arange(lu - lp) // bd_size  # bd-block within this turn
        resp_block[pos] = gblk + within
        gblk += int(within.max().item()) + 1
        n_turns += 1

    if n_turns == 0 or int((labels != -100).sum()) == 0:
        return None
    return {"input_ids": ids, "labels": labels, "resp_block": resp_block, "turn_id": turn_id}


class TextSftCollator:
    """Build samples, drop over-length / over-response-length ones, bucket the batch's region-A
    length (fewer distinct flex-attention shapes), and right-pad to the bucket."""

    def __init__(self, tokenizer, max_length=16384, max_response_length=4096,
                 length_buckets=(2048, 4096, 8192, 16384), keep_all_reasoning=False):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.length_buckets = tuple(length_buckets) if length_buckets else None
        self.keep_all_reasoning = keep_all_reasoning

    def __call__(self, examples):
        kept = []
        for ex in examples:
            s = build_sample(self.tokenizer, ex, keep_all_reasoning=self.keep_all_reasoning)
            if s is None:
                continue
            if s["input_ids"].shape[0] > self.max_length:
                continue
            if int((s["labels"] != -100).sum()) > self.max_response_length:
                continue
            kept.append(s)
        if not kept:
            return None

        maxlen = max(s["input_ids"].shape[0] for s in kept)
        # bucket region-A length (block_size=1: no block rounding, just snap to a stable bucket)
        L = bucketed_clean_len(maxlen, 1, self.length_buckets)
        L = min(L, self.max_length)

        def pad(t, value):
            return F.pad(t, (0, L - t.shape[0]), value=value)

        input_ids = torch.stack([pad(s["input_ids"], self.pad_id) for s in kept])
        labels = torch.stack([pad(s["labels"], -100) for s in kept])
        attention_mask = torch.stack(
            [pad(torch.ones_like(s["input_ids"]), 0) for s in kept]
        )
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


class MultiTurnTextSftCollator:
    """Like ``TextSftCollator`` but supervises EVERY assistant turn (``build_sample_multiturn``) and
    emits per-token ``resp_block`` / ``turn_id`` for the multi-turn block-diffusion forward. One
    sample per row (no packing); the batch is bucketed + right-padded like the single-turn collator.
    ``max_response_length`` caps the TOTAL supervised tokens across all turns."""

    def __init__(self, tokenizer, bd_size=32, max_length=16384, max_response_length=4096,
                 length_buckets=(2048, 4096, 8192, 16384), keep_all_reasoning=False):
        self.tokenizer = tokenizer
        self.bd_size = bd_size
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.length_buckets = tuple(length_buckets) if length_buckets else None
        self.keep_all_reasoning = keep_all_reasoning

    def __call__(self, examples):
        kept = []
        for ex in examples:
            s = build_sample_multiturn(self.tokenizer, ex, self.bd_size,
                                       keep_all_reasoning=self.keep_all_reasoning)
            if s is None:
                continue
            if s["input_ids"].shape[0] > self.max_length:
                continue
            if int((s["labels"] != -100).sum()) > self.max_response_length:
                continue
            kept.append(s)
        if not kept:
            return None

        maxlen = max(s["input_ids"].shape[0] for s in kept)
        L = min(bucketed_clean_len(maxlen, 1, self.length_buckets), self.max_length)

        def pad(t, value):
            return F.pad(t, (0, L - t.shape[0]), value=value)

        return {
            "input_ids": torch.stack([pad(s["input_ids"], self.pad_id) for s in kept]),
            "labels": torch.stack([pad(s["labels"], -100) for s in kept]),
            "attention_mask": torch.stack([pad(torch.ones_like(s["input_ids"]), 0) for s in kept]),
            "resp_block": torch.stack([pad(s["resp_block"], -1) for s in kept]),
            "turn_id": torch.stack([pad(s["turn_id"], -1) for s in kept]),
        }


def _forward_cost(s, ar_only: bool = False):
    """Cost a sample adds to a packed row = the forward length it induces.

    Block diffusion (default): ``n_input + 2 * n_response`` — region A once plus region B, the clean
    response copy. Bounding THAT by max_length caps real GPU work per row.

    ``ar_only=True``: pure AR has NO region B (``forward_ar`` runs a single [B,L] causal pass), so the
    cost is just the token count. Charging the block-diffusion 2*n_response here would overestimate
    every AR sample and fill rows to only ~half of max_length."""
    n = s["input_ids"].shape[0]
    return n if ar_only else n + 2 * int((s["labels"] != -100).sum())


def greedy_pack(samples, max_length, max_segments=0, ar_only: bool = False):
    """First-fit: fill a row until the next sample would overflow the forward-cost budget (or the
    segment cap), then start a new row."""
    rows, cur, cost = [], [], 0
    for s in samples:
        c = _forward_cost(s, ar_only)
        over = cost + c > max_length
        over_seg = max_segments and len(cur) >= max_segments
        if cur and (over or over_seg):
            rows.append(cur)
            cur, cost = [], 0
        cur.append(s)
        cost += c
    if cur:
        rows.append(cur)
    return rows


def pack_rows_to_batch(rows, pad_id, length_buckets, max_length):
    """Stack packed rows into (input_ids, labels, attention_mask, seg_id, resp_pos). Segments are
    concatenated; ``seg_id`` is the per-row segment index (pad = -1); ``resp_pos`` is the
    response-relative index within each segment (-1 off-response). Rows pad to a bucketed length."""
    L = bucketed_clean_len(
        max(sum(s["input_ids"].shape[0] for s in row) for row in rows), 1, length_buckets
    )
    L = min(L, max_length)

    def row_tensors(row):
        ids, lab, seg, rpos = [], [], [], []
        for k, s in enumerate(row):
            n = s["input_ids"].shape[0]
            # Each packed segment must start with a non-response (prefix) token. The token-shifted
            # diffusion loss (hidden[i] -> labels[i+1]) is only cross-segment-safe because the token
            # after a segment's last response token is the next segment's masked prefix; an
            # empty-prefix segment would make one segment's last response predict the next segment's
            # first response. build_sample guarantees prompt_len>=1, so assert the invariant here.
            assert int(s["labels"][0]) == -100, "packed segment must start with a prefix (label=-100) token"
            ids.append(s["input_ids"])
            lab.append(s["labels"])
            seg.append(torch.full((n,), k, dtype=torch.long))
            rp = torch.full((n,), -1, dtype=torch.long)
            ans = (s["labels"] != -100).nonzero(as_tuple=True)[0]
            rp[ans] = torch.arange(ans.numel(), dtype=torch.long)
            rpos.append(rp)
        ids = torch.cat(ids); lab = torch.cat(lab); seg = torch.cat(seg); rpos = torch.cat(rpos)
        am = torch.ones_like(ids)
        pad = L - ids.shape[0]
        if pad > 0:
            ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
            lab = torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)])
            am = torch.cat([am, torch.zeros(pad, dtype=am.dtype)])
            seg = torch.cat([seg, torch.full((pad,), -1, dtype=seg.dtype)])
            rpos = torch.cat([rpos, torch.full((pad,), -1, dtype=rpos.dtype)])
        return ids, lab, am, seg, rpos

    out = [row_tensors(r) for r in rows]
    return {
        "input_ids": torch.stack([o[0] for o in out]),
        "labels": torch.stack([o[1] for o in out]),
        "attention_mask": torch.stack([o[2] for o in out]),
        "seg_id": torch.stack([o[3] for o in out]),
        "resp_pos": torch.stack([o[4] for o in out]),
    }


class PackedTextSftCollator:
    """Pack multiple single-turn samples per row with segment-isolated (block-diagonal) attention.
    Pulls ``pack_examples`` raw examples per call (DataLoader batch_size), builds + filters them,
    greedy-packs by forward cost, and emits packed row tensors. ``max_packed_rows`` (>0) is an
    OOM-safety drop cap on rows per step."""

    def __init__(self, tokenizer, max_length=16384, max_response_length=4096,
                 length_buckets=(2048, 4096, 8192, 16384), max_segments=0, max_packed_rows=0,
                 keep_all_reasoning=False, ar_only=False):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.length_buckets = tuple(length_buckets) if length_buckets else None
        self.max_segments = max_segments
        self.max_packed_rows = max_packed_rows
        self.keep_all_reasoning = keep_all_reasoning
        self.ar_only = ar_only

    def __call__(self, examples):
        kept = []
        for ex in examples:
            s = build_sample(self.tokenizer, ex, keep_all_reasoning=self.keep_all_reasoning)
            if s is None:
                continue
            if _forward_cost(s, self.ar_only) > self.max_length:
                continue
            if int((s["labels"] != -100).sum()) > self.max_response_length:
                continue
            kept.append(s)
        if not kept:
            return None
        rows = greedy_pack(kept, self.max_length, self.max_segments, self.ar_only)
        if self.max_packed_rows and len(rows) > self.max_packed_rows:
            rows = rows[: self.max_packed_rows]
        return pack_rows_to_batch(rows, self.pad_id, self.length_buckets, self.max_length)


def _block_align_doc(s, bd_size, pad_id):
    """Pad a multi-turn doc up to a multiple of ``bd_size`` so that, once packed, every document's
    START lands on a block boundary — required by FLARE's two-stream packed kernel
    (``cu_seqlens[:-1] % block_size == 0``). The trailing pad is non-response (label -100), off-turn
    (turn_id / resp_block -1), and stays part of THIS doc's segment (so the doc spans whole blocks)."""
    T = s["input_ids"].shape[0]
    padlen = (-T) % bd_size
    if padlen == 0:
        return s
    def cat(x, v):
        return torch.cat([x, torch.full((padlen,), v, dtype=x.dtype)])
    return {"input_ids": cat(s["input_ids"], pad_id), "labels": cat(s["labels"], -100),
            "resp_block": cat(s["resp_block"], -1), "turn_id": cat(s["turn_id"], -1)}


def pack_rows_multiturn(rows, pad_id, bd_size, length_buckets, max_length):
    """Stack packed MULTI-TURN rows -> (input_ids, labels, attention_mask, seg_id, turn_id). Each row
    is a list of block-aligned docs (whole conversations, every assistant turn supervised); ``seg_id``
    is the per-doc index (row-pad = -1); ``turn_id`` is the doc-local assistant-turn ordinal (off-resp
    / pad = -1). The bucket length is a multiple of ``bd_size`` so the whole row is block-aligned."""
    need = max(sum(s["input_ids"].shape[0] for s in row) for row in rows)
    L = min(bucketed_clean_len(need, bd_size, length_buckets), max_length)

    def row_tensors(row):
        ids, lab, seg, tid = [], [], [], []
        for k, s in enumerate(row):
            n = s["input_ids"].shape[0]
            ids.append(s["input_ids"]); lab.append(s["labels"])
            seg.append(torch.full((n,), k, dtype=torch.long)); tid.append(s["turn_id"])
        ids = torch.cat(ids); lab = torch.cat(lab); seg = torch.cat(seg); tid = torch.cat(tid)
        am = torch.ones_like(ids)
        pad = L - ids.shape[0]
        if pad > 0:  # row-end pad (block-aligned since docs + bucket are bd_size multiples)
            ids = torch.cat([ids, torch.full((pad,), pad_id, dtype=ids.dtype)])
            lab = torch.cat([lab, torch.full((pad,), -100, dtype=lab.dtype)])
            am = torch.cat([am, torch.zeros(pad, dtype=am.dtype)])
            seg = torch.cat([seg, torch.full((pad,), -1, dtype=seg.dtype)])
            tid = torch.cat([tid, torch.full((pad,), -1, dtype=tid.dtype)])
        return ids, lab, am, seg, tid

    out = [row_tensors(r) for r in rows]
    return {"input_ids": torch.stack([o[0] for o in out]),
            "labels": torch.stack([o[1] for o in out]),
            "attention_mask": torch.stack([o[2] for o in out]),
            "seg_id": torch.stack([o[3] for o in out]),
            "turn_id": torch.stack([o[4] for o in out])}


class PackedMultiTurnCollator:
    """Pack multiple WHOLE multi-turn conversations per row for the FLARE two-stream forward: every
    assistant turn supervised (``build_sample_multiturn``, interleaved reasoning), each conversation
    block-aligned + doc-isolated (``seg_id`` -> per-doc ``cu_seqlens`` in the recurrence/conv/mask).
    Mirrors ``PackedTextSftCollator`` but multi-turn + block-aligned. Forward cost is the doc length
    (the [x0;xt] forward is 2*L, budgeted by capping the row to ``max_length``)."""

    def __init__(self, tokenizer, bd_size=4, max_length=32768, max_response_length=32768,
                 length_buckets=(2048, 4096, 8192, 16384, 32768), max_segments=0, max_packed_rows=0,
                 keep_all_reasoning=True):
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.bd_size = bd_size
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.length_buckets = tuple(length_buckets) if length_buckets else None
        self.max_segments = max_segments
        self.max_packed_rows = max_packed_rows
        self.keep_all_reasoning = keep_all_reasoning

    def __call__(self, examples):
        kept = []
        for ex in examples:
            s = build_sample_multiturn(self.tokenizer, ex, self.bd_size,
                                       keep_all_reasoning=self.keep_all_reasoning)
            if s is None:
                continue
            s = _block_align_doc(s, self.bd_size, self.pad_id)
            if s["input_ids"].shape[0] > self.max_length:
                continue
            if int((s["labels"] != -100).sum()) > self.max_response_length:
                continue
            kept.append(s)
        if not kept:
            return None
        rows = greedy_pack(kept, self.max_length, self.max_segments, ar_only=True)  # cost = doc length
        if self.max_packed_rows and len(rows) > self.max_packed_rows:
            rows = rows[: self.max_packed_rows]
        return pack_rows_multiturn(rows, self.pad_id, self.bd_size, self.length_buckets, self.max_length)


class PackedRowDataset(IterableDataset):
    """Streaming first-fit packer with **carry-over** — no dropping, no ``max_packed_rows``.

    Consumes built single-turn samples from ``base`` and yields ONE packed row at a time (a list of
    segment dicts) whose total forward-cost stays ``<= max_length``. When the next sample would push
    the current row over ``max_length``, the row is emitted (the tensorizer pads it out) and that
    sample **starts the next row** — it is never dropped. The only samples skipped are ones that
    can't fit even alone (``forward_cost > max_length`` or ``response > max_response_length``), which
    would otherwise require truncation. Packing runs per DataLoader worker over its sharded stream.
    """

    def __init__(self, base, tokenizer, max_length=32768, max_response_length=32768,
                 keep_all_reasoning=False, max_segments=0, ar_only=False):
        self.base = base
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.keep_all_reasoning = keep_all_reasoning
        self.max_segments = max_segments
        # ar_only drops the block-diffusion 2*n_response term so rows fill to the real token budget.
        self.ar_only = ar_only

    def __iter__(self):
        row, cost = [], 0
        for ex in self.base:
            s = build_sample(self.tokenizer, ex, keep_all_reasoning=self.keep_all_reasoning)
            if s is None:
                continue
            c = _forward_cost(s, self.ar_only)
            if c > self.max_length or int((s["labels"] != -100).sum()) > self.max_response_length:
                continue  # can't fit alone without truncation -> skip
            over_len = cost + c > self.max_length
            over_seg = self.max_segments and len(row) >= self.max_segments
            if row and (over_len or over_seg):
                yield row              # emit the full row; `s` carries over to the next
                row, cost = [], 0
            row.append(s)
            cost += c
        if row:
            yield row


class StreamPackedCollator:
    """Tensorize a batch of pre-packed rows from ``PackedRowDataset`` (each row is already a
    ``<= max_length`` bin) via ``pack_rows_to_batch`` — no packing or dropping here. DataLoader
    ``batch_size`` = rows per micro-batch (1 for 32k)."""

    def __init__(self, tokenizer, max_length=32768, length_buckets=(2048, 4096, 8192, 16384, 32768)):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.max_length = max_length
        self.length_buckets = tuple(length_buckets) if length_buckets else None

    def __call__(self, rows):
        rows = [r for r in rows if r]
        if not rows:
            return None
        return pack_rows_to_batch(rows, self.pad_id, self.length_buckets, self.max_length)


def download_jsonl_paths(repo_id):
    """Return local paths to the dataset's JSONL data files.

    ``repo_id`` may be a **local directory** (uses its ``*.jsonl`` files directly) or an HF Hub
    dataset id (downloads the ``.jsonl`` files)."""
    import glob
    import os

    if os.path.isdir(repo_id):
        paths = sorted(glob.glob(os.path.join(repo_id, "*.jsonl")))
        if not paths:
            raise FileNotFoundError(f"no .jsonl data files in local dir {repo_id}")
        return paths

    from huggingface_hub import hf_hub_download, list_repo_files

    files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.endswith(".jsonl")]
    if not files:
        raise FileNotFoundError(f"no .jsonl data files in {repo_id}")
    return [hf_hub_download(repo_id, f, repo_type="dataset") for f in sorted(files)]


class JsonlAgenticDataset(IterableDataset):
    """Streaming JSONL reader for the agentic SFT data.

    We bypass ``datasets``/Arrow entirely — agentic JSONL records often carry heterogeneous nested
    metadata that trips Arrow's unified-schema cast, and large files would otherwise be loaded once
    per rank, so streaming avoids that RAM blow-up. Shards deterministically across (DDP rank x DataLoader
    worker) via line modulo, and applies a bounded in-memory shuffle buffer. Yields only the fields
    the collator needs (``messages``, ``tools``). Re-iterating (a new epoch) advances the RNG so the
    shuffle differs across passes.
    """

    def __init__(self, paths, rank=0, world=1, seed=42, shuffle_buffer=10000):
        self.paths = list(paths)
        self.rank = rank
        self.world = world
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        # Shared epoch counter: DataLoader re-FORKS workers each epoch, so a plain int attribute
        # would reset to its parent value every epoch (same shuffle forever). A fork-shared
        # mp.Value survives across the forked workers/epochs. Each __iter__ atomically takes the
        # next value, so every (epoch, worker) gets a distinct shuffle seed.
        self._epoch = mp.Value("i", 0)

    def _lines(self, shard, nshard):
        idx = 0
        for path in self.paths:
            with open(path) as f:
                for line in f:
                    if idx % nshard == shard:
                        yield line
                    idx += 1

    def __iter__(self):
        worker = get_worker_info()
        wid = worker.id if worker else 0
        nworkers = worker.num_workers if worker else 1
        shard = self.rank * nworkers + wid
        nshard = self.world * nworkers
        with self._epoch.get_lock():
            nonce = self._epoch.value
            self._epoch.value += 1
        rng = random.Random(self.seed + 1000 * nonce + shard)

        buf = []
        for line in self._lines(shard, nshard):
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = {"messages": ex.get("messages") or ex.get("conversations"), "tools": ex.get("tools")}
            if not item["messages"]:
                continue
            if self.shuffle_buffer <= 1:
                yield item
                continue
            buf.append(item)
            if len(buf) >= self.shuffle_buffer:
                yield buf.pop(rng.randrange(len(buf)))
        rng.shuffle(buf)
        for item in buf:
            yield item
