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
import os
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


# ---- images (Qwen3.5 VL) --------------------------------------------------------------------
IMAGE_PAD = "<|image_pad|>"
SPATIAL_MERGE = 2  # Qwen3.5 vision merger: 2x2 patches -> one LLM token


def image_pad_id(tokenizer):
    """``<|image_pad|>`` id, or None for tokenizers without it (cached: get_vocab() is costly)."""
    pid = getattr(tokenizer, "_trida_image_pad_id", -1)
    if pid == -1:
        pid = tokenizer.convert_tokens_to_ids(IMAGE_PAD) if IMAGE_PAD in tokenizer.get_vocab() else None
        tokenizer._trida_image_pad_id = pid
    return pid


def _is_image_part(part):
    # Mirrors the chat template's test, so refs come out in the order it emits <|image_pad|>.
    return isinstance(part, dict) and ("image" in part or "image_url" in part or part.get("type") == "image")


def _image_refs(messages):
    """Image references in render order. Raises on video parts (images only for now)."""
    refs = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if _is_image_part(part):
                ref = part.get("image") or part.get("image_url")
                refs.append(ref.get("url") if isinstance(ref, dict) else ref)
            elif isinstance(part, dict) and ("video" in part or part.get("type") == "video"):
                raise ValueError("video content is not supported")
    return refs


def _load_image(ref, image_root=None):
    import base64
    import io

    from PIL import Image
    if ref.startswith("data:"):
        img = Image.open(io.BytesIO(base64.b64decode(ref.split(",", 1)[1])))
    else:
        path = ref[len("file://"):] if ref.startswith("file://") else ref
        if image_root and not os.path.isabs(path):
            path = os.path.join(image_root, path)
        img = Image.open(path)
    return img.convert("RGB")


def load_image_processor(path):
    """The checkpoint's image processor on the PIL backend (no torchvision needed)."""
    try:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(path, backend="pil")
    except ImportError:  # transformers gates AutoImageProcessor itself on torchvision
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        return Qwen2VLImageProcessorPil.from_pretrained(path)


def process_images(image_processor, refs, image_root=None, max_image_pixels=1024 * 1024):
    """Load + preprocess one conversation's images. Returns (pixel_values [P, C*T*p*p],
    image_grid_thw [N, 3], tokens per image)."""
    images = [_load_image(r, image_root) for r in refs]
    size = dict(image_processor.size)
    size["longest_edge"] = min(size.get("longest_edge") or max_image_pixels, max_image_pixels)
    out = image_processor(images=images, return_tensors="pt", size=size)
    grid = out["image_grid_thw"].long()
    counts = (grid.prod(-1) // SPATIAL_MERGE**2).tolist()
    return out["pixel_values"], grid, counts


def _expand_image_pads(ids, pad_id, counts):
    """The template emits ONE <|image_pad|> per image; the model needs one per merged patch. Images
    appear in the same order in every (prefix) render, so expanding with the same ``counts`` keeps
    prefix renders token-stable."""
    out, k = [], 0
    for t in ids:
        if t == pad_id:
            out.extend([pad_id] * counts[k])
            k += 1
        else:
            out.append(t)
    return out


def mrope_position_ids(input_ids, image_grid_thw, image_token_id, spatial_merge=SPATIAL_MERGE):
    """3-D (t, h, w) RoPE positions for ONE document, ``[3, T]`` — the single-sequence form of
    ``Qwen3_5Model.get_rope_index``: text runs advance all three axes together; an image's tokens
    take (start, start + row, start + col) over its merged grid, then the position advances by
    ``max(h, w) // spatial_merge``. A text-only doc gives arange on every axis (= 1-D RoPE)."""
    ids = torch.as_tensor(input_ids)
    is_img = ids == image_token_id
    pos = torch.empty(3, ids.shape[0], dtype=torch.long)
    grids = iter(image_grid_thw.tolist() if image_grid_thw is not None else [])
    cur, i, T = 0, 0, ids.shape[0]
    while i < T:
        j = i
        while j < T and bool(is_img[j]) == bool(is_img[i]):
            j += 1
        if not is_img[i]:
            pos[:, i:j] = torch.arange(cur, cur + (j - i))
            cur += j - i
        else:
            _, gh, gw = next(grids)
            h, w = gh // spatial_merge, gw // spatial_merge
            if j - i != h * w:
                raise ValueError(f"image span of {j - i} tokens does not match grid {h}x{w}")
            pos[0, i:j] = cur
            pos[1, i:j] = cur + torch.arange(h).repeat_interleave(w)
            pos[2, i:j] = cur + torch.arange(w).repeat(h)
            cur += max(gh, gw) // spatial_merge
        i = j
    return pos


def _apply_template(tokenizer, messages, tools, add_generation_prompt, keep_all_reasoning=False,
                    image_counts=None):
    """Render + tokenize one message list. No per-call tools fallback: the tools-vs-no-tools mode
    is chosen ONCE per example in ``build_sample`` so ``full`` and ``prompt`` always render in the
    same mode (otherwise they can diverge and break the prefix/label boundary).

    ``keep_all_reasoning`` is forwarded to the chat template (a no-op on stock templates; on the
    patched Qwen3 template it keeps ``<think>`` for non-last assistant turns that carry
    ``reasoning_content``). It must be identical across the ``full`` / ``prompt`` renders of one
    sample, else the rendered reasoning diverges and the prefix/label boundary breaks.

    ``image_counts`` (tokens per image, in render order) expands each ``<|image_pad|>``; a render that
    contains image pads without counts raises (only the packed multi-turn path handles images)."""
    enc = tokenizer.apply_chat_template(
        messages, tools=tools, tokenize=True, add_generation_prompt=add_generation_prompt,
        keep_all_reasoning=keep_all_reasoning,
    )
    ids = _to_ids(enc)
    pad_id = image_pad_id(tokenizer)
    if pad_id is not None and pad_id in ids:
        if image_counts is None:
            raise ValueError("conversation has images; only --pack --multiturn supports them")
        ids = _expand_image_pads(ids, pad_id, image_counts)
    return ids


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


def build_sample_multiturn(tokenizer, example, bd_size, keep_all_reasoning=False,
                           image_processor=None, max_image_pixels=1024 * 1024):
    """Supervise EVERY assistant turn of a conversation in ONE sequence (for the causally-ordered
    multi-turn block-diffusion forward), instead of only the last turn.

    Returns ``None`` if unusable, else a dict of LongTensor[T]:
        input_ids, labels (all assistant turns supervised; else -100),
        resp_block (GLOBAL, conversation-ordered block id, unique per (turn, bd-block); -1 off-resp),
        turn_id   (assistant-turn ordinal 0,1,...; -1 off-resp),
    plus, when the conversation has images: ``pixel_values`` [P, C*T*p*p], ``image_grid_thw`` [N, 3]
    and ``mrope_pos`` [3, T] (3-D RoPE positions). Images are image parts in message ``content``
    (``{"type": "image", "image": <path | data: URI>}``; relative paths resolve against
    ``example["image_root"]``) and need ``image_processor``. ``example["max_image_pixels"]``, if set,
    overrides ``max_image_pixels`` for that conversation's images.

    Each assistant turn's content span is located by incremental prefix renders (the standard
    multi-turn label-masking trick): ``prompt_j = template(messages[:i_j], add_generation_prompt)``
    marks the turn's content start, ``upto_j = template(messages[:i_j+1])`` its end. A turn without
    reasoning renders ``<think>\\n\\n</think>``, whose ``\\n\\n`` token never matches the generation
    prompt's ``<think>\\n``; such turns are located by the turn header instead
    (``template(messages[:i_j]) + "<|im_start|>assistant\\n"``) and train with the empty think block.
    Turns that still don't render as exact token prefixes of the full render are skipped (become
    clean conditioning) rather than mis-supervised.
    """
    messages = example.get("messages") or example.get("conversations")
    if not messages:
        return None
    messages = _normalize_messages(messages)

    asst_idxs = [i for i, m in enumerate(messages) if m["role"] == "assistant"]
    if not asst_idxs:
        return None

    try:
        refs = _image_refs(messages)
    except ValueError:
        return None
    counts = None
    if refs:
        if image_processor is None:
            return None
        # a record may raise (or lower) the cap for its own images, e.g. dense diagrams
        cap = example.get("max_image_pixels") or max_image_pixels
        try:
            pixel_values, grid, counts = process_images(image_processor, refs, example.get("image_root"), cap)
        except Exception:
            return None

    tools = example.get("tools")

    def render(msgs, tools_arg, gen):
        return _apply_template(tokenizer, msgs, tools_arg, gen, keep_all_reasoning=keep_all_reasoning,
                               image_counts=counts)

    def render_all(tools_arg):
        full = render(messages, tools_arg, False)
        spans = [(render(messages[:i], tools_arg, True), render(messages[: i + 1], tools_arg, False))
                 for i in asst_idxs]
        return full, spans

    tools_arg = tools
    try:
        full, spans = render_all(tools)
    except Exception:
        tools_arg = None
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
    header = None

    gblk = 0
    n_turns = 0
    for j, (prefix, upto) in enumerate(spans):
        lp, lu = len(prefix), len(upto)
        # prefix stability: both incremental renders must be exact token prefixes of `full`
        if lu > T or full[:lu] != upto:
            continue
        if lp >= lu or full[:lp] != prefix:
            if header is None:
                header = tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
            try:
                base = render(messages[: asst_idxs[j]], tools_arg, False)
            except Exception:
                continue
            lp = len(base) + len(header)
            if lp >= lu or full[: len(base)] != base or full[len(base):lp] != header:
                continue
        pos = torch.arange(lp, lu)
        labels[pos] = ids[pos]
        turn_id[pos] = j
        within = torch.arange(lu - lp) // bd_size  # bd-block within this turn
        resp_block[pos] = gblk + within
        gblk += int(within.max().item()) + 1
        n_turns += 1

    if counts:
        labels[ids == image_pad_id(tokenizer)] = -100  # image tokens are never prediction targets
    if n_turns == 0 or int((labels != -100).sum()) == 0:
        return None
    out = {"input_ids": ids, "labels": labels, "resp_block": resp_block, "turn_id": turn_id}
    if counts:
        out.update(pixel_values=pixel_values, image_grid_thw=grid,
                   mrope_pos=mrope_position_ids(ids, grid, image_pad_id(tokenizer)))
    return out


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
    out = dict(s, input_ids=cat(s["input_ids"], pad_id), labels=cat(s["labels"], -100),
               resp_block=cat(s["resp_block"], -1), turn_id=cat(s["turn_id"], -1))
    if "mrope_pos" in s:  # the pad continues the doc's positions, as the 1-D path does
        nxt = int(s["mrope_pos"].max()) + 1
        out["mrope_pos"] = torch.cat([s["mrope_pos"], torch.arange(nxt, nxt + padlen).expand(3, -1)], dim=1)
    return out


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
    batch = {"input_ids": torch.stack([o[0] for o in out]),
             "labels": torch.stack([o[1] for o in out]),
             "attention_mask": torch.stack([o[2] for o in out]),
             "seg_id": torch.stack([o[3] for o in out]),
             "turn_id": torch.stack([o[4] for o in out])}
    docs = [s for row in rows for s in row]
    if any("mrope_pos" in s for s in docs):
        # 3-D RoPE positions [B, 3, L]: each doc starts at 0 (per-doc reset), text-only docs are
        # arange on all three axes; row-end pad is 0 (masked out). Image tensors are concatenated
        # in row-then-doc order, which is the order their <|image_pad|> runs appear in input_ids.
        mrope = torch.zeros(len(rows), 3, L, dtype=torch.long)
        for b, row in enumerate(rows):
            off = 0
            for s in row:
                n = s["input_ids"].shape[0]
                mrope[b, :, off:off + n] = s["mrope_pos"] if "mrope_pos" in s else torch.arange(n).expand(3, -1)
                off += n
        batch["mrope_pos"] = mrope
        batch["pixel_values"] = torch.cat([s["pixel_values"] for s in docs if "pixel_values" in s])
        batch["image_grid_thw"] = torch.cat([s["image_grid_thw"] for s in docs if "image_grid_thw" in s])
    return batch


def _stream_first_fit(docs, max_length, buffer_size=32, max_segments=0):
    """Buffered first-fit packing over a doc stream; yields rows (lists of docs) of total length
    ``<= max_length``. Nothing is dropped: a doc that doesn't fit the current row waits in the buffer
    for a later row. Every row starts with the OLDEST buffered doc, so a long doc can't starve, and
    the buffer never holds more than ``buffer_size`` docs because docs are pulled only to refill it.
    Each doc must already fit alone (``len(doc["input_ids"]) <= max_length``); both preconditions
    are enforced, because violating either one fails silently otherwise -- an oversized doc becomes
    a row head unconditionally and yields a row over ``max_length``, and ``buffer_size < 1`` leaves
    the buffer permanently empty, so the generator returns at once and the whole stream is lost."""
    if buffer_size < 1:
        raise ValueError(f"buffer_size must be >= 1, got {buffer_size}")
    it = iter(docs)
    buf, exhausted = [], False

    def refill():
        nonlocal exhausted
        while not exhausted and len(buf) < buffer_size:
            try:
                buf.append(next(it))
            except StopIteration:
                exhausted = True

    while True:
        refill()
        if not buf:
            return
        row = [buf.pop(0)]
        used = row[0]["input_ids"].shape[0]
        if used > max_length:
            raise ValueError(
                f"doc of {used} tokens does not fit alone in max_length={max_length}; "
                "callers must drop it before packing (see PackedMultiTurnRowDataset._docs)"
            )
        i = 0
        while True:
            while i < len(buf) and not (max_segments and len(row) >= max_segments):
                n = buf[i]["input_ids"].shape[0]
                if used + n <= max_length:
                    row.append(buf.pop(i))
                    used += n
                else:
                    i += 1
            before = len(buf)
            refill()  # docs popped into this row freed buffer slots; scan the newcomers too
            if len(buf) == before or (max_segments and len(row) >= max_segments):
                break
        yield row


class PackedMultiTurnRowDataset(IterableDataset):
    """Packed multi-turn rows for the FLARE two-stream forward (``--pack --multiturn``): builds whole
    conversations (every assistant turn supervised via ``build_sample_multiturn``), block-aligns each
    so packed doc starts land on block boundaries (doc-isolated via ``seg_id``), and yields ONE row at
    a time via buffered first-fit (``_stream_first_fit``). A conversation that doesn't fit the
    current row waits for a later one, so the only docs skipped are those that can't fit alone
    (``> max_length`` tokens, or ``> max_response_length`` supervised). DataLoader ``batch_size`` =
    rows per forward (``--micro_batch_size``). Packing runs per DataLoader worker over its sharded
    stream. With ``image_processor`` set, conversations may carry images (see
    ``build_sample_multiturn``); their tokens count toward ``max_length``."""

    def __init__(self, base, tokenizer, bd_size, max_length=32768, max_response_length=32768,
                 keep_all_reasoning=True, max_segments=0, buffer_size=32, image_processor=None,
                 max_image_pixels=1024 * 1024):
        self.base = base
        self.tokenizer = tokenizer
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.bd_size = bd_size
        self.max_length = max_length
        self.max_response_length = max_response_length
        self.keep_all_reasoning = keep_all_reasoning
        self.max_segments = max_segments
        self.buffer_size = buffer_size
        self.image_processor = image_processor
        self.max_image_pixels = max_image_pixels

    def _docs(self, counts):
        for ex in self.base:
            s = build_sample_multiturn(self.tokenizer, ex, self.bd_size,
                                       keep_all_reasoning=self.keep_all_reasoning,
                                       image_processor=self.image_processor,
                                       max_image_pixels=self.max_image_pixels)
            if s is None:
                counts["unusable"] += 1
                continue
            s = _block_align_doc(s, self.bd_size, self.pad_id)
            if s["input_ids"].shape[0] > self.max_length:
                counts["too_long"] += 1
                continue
            if int((s["labels"] != -100).sum()) > self.max_response_length:
                counts["response_too_long"] += 1
                continue
            counts["packed"] += 1
            yield s

    def __iter__(self):
        counts = {"packed": 0, "too_long": 0, "response_too_long": 0, "unusable": 0}
        rows = 0
        for row in _stream_first_fit(self._docs(counts), self.max_length, self.buffer_size,
                                     self.max_segments):
            rows += 1
            yield row
        info = get_worker_info()
        print(f"[pack] worker {info.id if info else 0}: {rows} rows from {counts['packed']} docs; "
              f"skipped too_long={counts['too_long']} response_too_long={counts['response_too_long']} "
              f"unusable={counts['unusable']}", flush=True)


class StreamPackedMultiTurnCollator:
    """Tensorize rows from ``PackedMultiTurnRowDataset`` via ``pack_rows_multiturn`` (no packing or
    dropping here)."""

    def __init__(self, tokenizer, bd_size, max_length=32768,
                 length_buckets=(2048, 4096, 8192, 16384, 32768)):
        self.pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.bd_size = bd_size
        self.max_length = max_length
        self.length_buckets = tuple(length_buckets) if length_buckets else None

    def __call__(self, rows):
        rows = [r for r in rows if r]
        if not rows:
            return None
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
                        yield path, line
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
        for path, line in self._lines(shard, nshard):
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            item = {"messages": ex.get("messages") or ex.get("conversations"), "tools": ex.get("tools"),
                    "image_root": os.path.dirname(os.path.abspath(path)),  # relative image paths
                    "max_image_pixels": ex.get("max_image_pixels")}        # per-record image cap
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
