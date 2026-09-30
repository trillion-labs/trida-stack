"""Engine: model + tokenizer + chat template + a single-sequence prompt cache.

The prompt cache is what makes a local agent loop fast: each turn re-sends the whole
conversation, but only the new messages need a prefill. Gated-delta layers cannot rewind
an arbitrary number of tokens, so the engine keeps two resumable points:

  * the live cache (everything committed at the end of the last generation), and
  * a snapshot taken right before the last ``<|im_start|>`` of the previous prompt
    (i.e. before the ``assistant`` generation header), which is where the next turn's
    re-rendered history is guaranteed to still agree with it.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Iterator, Optional

import mlx.core as mx

from .decode import DecodeStats, SamplingParams, ar_generate, selfspec_generate
from .model import DEFAULT_MODEL, TridaRuntime, load_model, mask_id_of


class IncrementalDetokenizer:
    def __init__(self, tokenizer):
        self.tok = tokenizer
        self.ids: list = []
        self.prefix = 0
        self.read = 0

    def add(self, new_ids: list) -> str:
        self.ids.extend(new_ids)
        prefix_text = self.tok.decode(self.ids[self.prefix : self.read], skip_special_tokens=False)
        new_text = self.tok.decode(self.ids[self.prefix :], skip_special_tokens=False)
        if new_text.endswith("�"):
            return ""
        delta = new_text[len(prefix_text):]
        self.prefix, self.read = self.read, len(self.ids)
        return delta


class _Slot:
    """One conversation's cache plus resumable snapshots: ``snaps['system']`` right after the
    system block (stable across every turn of an agent loop, and often 10k+ tokens of tool
    schemas) and ``snaps['last']`` before the last ``<|im_start|>`` of the previous prompt."""
    __slots__ = ("cache", "snaps", "used")

    def __init__(self, cache):
        self.cache, self.snaps, self.used = cache, {}, time.monotonic()


class Engine:
    """``cache_slots``: how many independent conversations keep a resumable cache. An agent
    harness interleaves its main loop with side requests (titles, summaries, memory review);
    with one slot every side request would evict the main conversation's cache."""

    def __init__(self, model: str = DEFAULT_MODEL, *, mode: str = "self-spec", gen_block: int = 4,
                 prefill_chunk: int = 512, prompt_cache: bool = True, fused_gdn: bool = True,
                 cache_slots: int = 3):
        from transformers import AutoTokenizer

        t = time.perf_counter()
        self.model, self.cfg, self.path = load_model(model)
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.path))
        self.mask_id = mask_id_of(self.path, self.cfg, self.tokenizer)
        self.rt = TridaRuntime(self.model, prefill_chunk=prefill_chunk, fused_gdn=fused_gdn)
        self.mode, self.n = mode, gen_block
        self.model_id = model
        self.eos_ids = self._eos_ids()
        self.im_start = self.tokenizer.convert_tokens_to_ids("<|im_start|>")
        self.use_prompt_cache = prompt_cache
        self.cache_slots = max(1, cache_slots)
        self.reset_cache()
        self.lock = threading.Lock()
        self.last_stats: Optional[dict] = None
        self.load_s = time.perf_counter() - t

    # current slot's cache / snapshot
    @property
    def cache(self):
        return self.cur.cache

    @cache.setter
    def cache(self, c):
        self.cur.cache = c

    def reset_cache(self) -> None:
        self.cur = _Slot(self.rt.make_cache())
        self.slots = [self.cur]

    def cached_tokens(self) -> list:
        return [s.cache.length for s in self.slots]
    def _eos_ids(self) -> set:
        ids = set()
        eos = self.tokenizer.eos_token_id
        if eos is not None:
            ids.add(int(eos))
        gc = self.path / "generation_config.json"
        if gc.is_file():
            e = json.loads(gc.read_text()).get("eos_token_id")
            ids.update([e] if isinstance(e, int) else (e or []))
        for t in ("<|im_end|>", "<|endoftext|>"):
            tid = self.tokenizer.convert_tokens_to_ids(t)
            if isinstance(tid, int) and tid >= 0 and tid != self.tokenizer.unk_token_id:
                ids.add(tid)
        return ids

    # -- prompts ----------------------------------------------------------------
    def render(self, messages: list, tools: Optional[list] = None, enable_thinking: bool = True,
               **template_kwargs) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=True, tokenize=False,
            enable_thinking=enable_thinking, **template_kwargs,
        )

    def encode(self, text: str) -> list:
        return self.tokenizer.encode(text, add_special_tokens=False)

    # -- prompt cache -----------------------------------------------------------
    def _position_cache(self, prompt: list) -> None:
        """Point ``self.cur`` at the slot holding the longest reusable prefix of ``prompt``
        (always < len(prompt)); otherwise take a fresh slot (evicting the least recently used)."""
        def usable(tokens):
            return 0 < len(tokens) < len(prompt) and prompt[: len(tokens)] == tokens

        if not self.use_prompt_cache:
            self.reset_cache()
            return
        best, best_slot, best_kind = 0, None, None
        for sl in self.slots:
            live = sl.cache.tokens
            if usable(live) and len(live) > best:
                best, best_slot, best_kind = len(live), sl, None
            for snap in sl.snaps.values():
                if usable(snap[0]) and len(snap[0]) > best:
                    best, best_slot, best_kind = len(snap[0]), sl, snap
        if best_slot is not None:
            self.cur = best_slot
            if best_kind is not None:
                best_slot.cache.restore(best_kind)
            # drop snapshots that are no longer a prefix of what the cache now holds
            held = best_slot.cache.tokens
            best_slot.snaps = {k: v for k, v in best_slot.snaps.items()
                               if len(v[0]) <= len(held) and held[: len(v[0])] == v[0]}
        elif any(sl.cache.length == 0 for sl in self.slots):
            self.cur = next(sl for sl in self.slots if sl.cache.length == 0)
        elif len(self.slots) < self.cache_slots:
            self.cur = _Slot(self.rt.make_cache())
            self.slots.append(self.cur)
        else:
            self.cur = min(self.slots, key=lambda s: s.used)
            self.cur.cache, self.cur.snaps = self.rt.make_cache(), {}
        self.cur.used = time.monotonic()

    def _prefill_to_snapshot(self, prompt: list) -> None:
        if not self.use_prompt_cache:
            return
        starts = [i for i, t in enumerate(prompt) if t == self.im_start]
        marks = []
        if len(starts) >= 2 and starts[0] == 0:
            marks.append(("system", starts[1]))  # end of the system block
        if starts:
            marks.append(("last", starts[-1]))  # before the generation header
        for name, cut in marks:
            if cut > self.cache.length:
                self.rt.prefill(self.cache, prompt[self.cache.length : cut])
                self.cur.snaps[name] = self.cache.snapshot()

    # -- generation -------------------------------------------------------------
    def generate_ids(self, prompt: list, *, max_tokens: int = 2048, sampling: Optional[SamplingParams] = None,
                     mode: Optional[str] = None, stats: Optional[DecodeStats] = None,
                     progress=None) -> Iterator[list]:
        """Yields lists of new token ids. Caller must hold ``self.lock``.
        ``progress(done, total)`` is called between prefill chunks (keep-alives for clients)."""
        sp = sampling or SamplingParams()
        mode = mode or self.mode
        stats = stats if stats is not None else DecodeStats()
        t = time.perf_counter()
        self._position_cache(prompt)
        reused = self.cache.length
        if progress is not None:
            done = [0]
            total = len(prompt) - reused

            def _cb(k):
                done[0] += k
                progress(done[0], total)
            self.rt.progress_cb = _cb
        try:
            self._prefill_to_snapshot(prompt)
        except BaseException:
            self.rt.progress_cb = None
            raise
        pre_s = time.perf_counter() - t
        if mode in ("causal", "ar"):
            gen = ar_generate(self.rt, self.cache, prompt, max_new_tokens=max_tokens, eos_ids=self.eos_ids,
                              sp=sp, stats=stats)
        else:
            gen = selfspec_generate(self.rt, self.cache, prompt, max_new_tokens=max_tokens,
                                    eos_ids=self.eos_ids, sp=sp, mask_id=self.mask_id, n=self.n, stats=stats)
        try:
            for i, chunk in enumerate(gen):
                if i == 0:
                    self.rt.progress_cb = None
                yield chunk
        finally:
            gen.close()
            self.rt.progress_cb = None
            stats.prefill_s += pre_s
            stats.reused_tokens = reused
            self.last_stats = stats.as_dict()

    def generate_text(self, prompt_text: str, **kw) -> Iterator[str]:
        detok = IncrementalDetokenizer(self.tokenizer)
        for ids in self.generate_ids(self.encode(prompt_text), **kw):
            d = detok.add(ids)
            if d:
                yield d

    def warmup(self) -> None:
        with self.lock:
            p = self.encode("<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n")
            for _ in self.generate_ids(p, max_tokens=8, sampling=SamplingParams(temperature=0.0)):
                pass
            self.reset_cache()
            mx.clear_cache()
