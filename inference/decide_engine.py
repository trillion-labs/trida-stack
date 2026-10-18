"""One engine for a decision checkpoint: decide, reason then decide, and generate, from one copy of the weights.

    from decide_engine import DecideEngine
    eng = DecideEngine("/path/to/checkpoint")           # in-process SGLang engine (stock SGLang >= 0.5.22)
    eng.decide({"state": ..., "questions": {...}})        # {"answers": {qid: answer}, "timing_ms": ...}
    eng.decide(request, reasoning=True)                   # writes its reasoning first, then decides with it
    eng.chat([{"role": "user", "content": "..."}])        # one assistant turn (thinking, then reply / tool call)

A decision checkpoint is a Trida checkpoint plus `jev_head.pt`: a small pointer head that scores each option of a
typed question against the question (MODEL_CONTRACT.md). It reads the last-layer hidden states at each option's end
and at `<decide>` in a causal token row (decide_request.py).

How this runs on a stock generation engine: the row is causal, so the hidden state at an option equals the last
hidden state of the row cut off there. Each question with K options becomes K+1 prefix prompts, sent in one batched
call with `return_hidden_states="last"`; the engine's prefix cache shares the work, and the head runs here. Checked
against a transformers implementation of the same head on 200 held-out decision records: accuracy 0.878 vs 0.881,
log loss 0.316 vs 0.315, mean absolute probability difference 0.1%.

Images: SGLang's Qwen-VL processor expands one `<|image_pad|>` into the image's tokens itself, so an image row's run of
image tokens is collapsed to one placeholder (option positions shift with it) and the image is resized to exactly the
size the training-time processor picked, so the engine sees the same image tokens.

The engine runs in-process because SGLang's HTTP front end (0.5.22) rejects `return_hidden_states="last"`.
Generation here is causal decoding on stock SGLang (Apache-2.0); the two-stream diffusion / self-spec decoders are in
serve.py's noncommercial backend (COMPLIANCE.md section 1b) and are not used.
"""
import json
import math
import os
import threading
import time

import torch
import torch.nn as nn

from decide_request import MAX_IMAGE_PIXELS, VISION, image_input, parse, prompt, with_reasoning

END_THINK = "</think>"
# Serving temperatures: logits are divided by T before the softmax. Fit T per checkpoint on held-out calibration
# data and put it in the checkpoint's temperature.json ({"text": T, "image": T}); these are the fallback.
DEFAULT_TEXT_T, DEFAULT_IMAGE_T = 1.0, 1.0


class PointerHead(nn.Module):
    """Scores each option against its question: logit = <k(h_option), q(h_decide)> / sqrt(head_dim)."""

    def __init__(self, d, dp=256):
        super().__init__()
        self.q, self.k = nn.Linear(d, dp), nn.Linear(d, dp)
        self.scale = 1 / math.sqrt(dp)

    def forward(self, h_decide, h_opts, owner):   # [Q, d], [sum K, d], question index of each option -> [sum K]
        return (self.k(h_opts) * self.q(h_decide)[owner]).sum(-1) * self.scale


def _confidence(qtype, p):
    k = len(p)
    if k == 1:
        return 1.0
    if qtype == "score":
        mode = max(range(k), key=p.__getitem__)
        d = sum(abs(i - (k - 1) / 2) for i in range(k)) / k
        return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / d)
    return (max(p) - 1 / k) / (1 - 1 / k)


def to_answer(q, p):
    """A question and its option probabilities -> the answer reported to the caller."""
    r = lambda x: round(float(x), 4)   # noqa: E731
    if q.qtype == "noul":
        return {"type": "noul", "noul": r(p[1])}
    if q.qtype == "choice":
        return {"type": "choice", "choice": q.keys[max(range(len(p)), key=p.__getitem__)],
                "confidence": r(_confidence("choice", p)), "probabilities": {k: r(v) for k, v in zip(q.keys, p)}}
    return {"type": "score", "score": r(sum(i * x for i, x in enumerate(p))), "legend": q.options,
            "confidence": r(_confidence("score", p)), "probabilities": {str(i): r(v) for i, v in enumerate(p)}}


def serving_temperatures(ckpt):
    """(text, image) temperatures from the checkpoint's temperature.json, else the defaults."""
    path = os.path.join(ckpt, "temperature.json")
    if not os.path.exists(path):
        return DEFAULT_TEXT_T, DEFAULT_IMAGE_T
    t = json.load(open(path))
    return float(t.get("text", DEFAULT_TEXT_T)), float(t.get("image", DEFAULT_IMAGE_T))


def load_image_processor(path):
    """The checkpoint's image processor on the PIL backend, as in training (the fast backend resizes slightly
    differently); None for a text-only checkpoint."""
    if not os.path.exists(os.path.join(path, "preprocessor_config.json")):
        return None
    try:
        from transformers import AutoImageProcessor
        return AutoImageProcessor.from_pretrained(path, backend="pil")
    except ImportError:   # transformers gates AutoImageProcessor itself on torchvision
        from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
        return Qwen2VLImageProcessorPil.from_pretrained(path)


def prefix_prompts(enc, image_pad):
    """Encoded rows -> (prompts, spans): one prompt per option end and per <decide>, each the row cut off there; spans
    give each row's (start, count) in the list. Image rows get their run of image tokens collapsed to one."""
    prompts, spans = [], []
    for r in enc.rows:
        ids, pos = list(r["ids"]), list(r["opts"]) + [r["decide"]]
        if enc.has_image:
            first = ids.index(image_pad)
            n = sum(1 for t in ids if t == image_pad)
            ids = ids[:first + 1] + ids[first + n:]
            pos = [p - (n - 1) if p > first else p for p in pos]
        spans.append((len(prompts), len(pos)))
        prompts += [ids[:p + 1] for p in pos]
    return prompts, spans


class DecideEngine:
    """Decisions, reasoned decisions and generation on one in-process SGLang engine. Thread-safe: every engine call
    takes a short lock, so concurrent requests interleave per call."""

    def __init__(self, ckpt, mem_fraction_static=0.8, context_length=16384, reasoning_temperature=0.6, **engine_kw):
        os.environ.setdefault("SGLANG_ENABLE_JIT_DEEPGEMM", "0")   # FP8 GEMMs: not needed for a bf16 model
        os.environ.setdefault("SGLANG_DEEPGEMM_PDL", "0")
        import sglang as sgl
        from transformers import AutoTokenizer
        self.ckpt, self.reasoning_temperature = ckpt, reasoning_temperature
        self.tok = AutoTokenizer.from_pretrained(ckpt)
        self.proc = load_image_processor(ckpt)
        saved = torch.load(os.path.join(ckpt, "jev_head.pt"), map_location="cpu")
        self.head = PointerHead(saved["head"]["q.weight"].shape[1], saved["head_dim"]).eval()
        self.head.load_state_dict(saved["head"])
        self.text_T, self.image_T = serving_temperatures(ckpt)
        self.image_pad = self.tok.convert_tokens_to_ids(VISION[1])
        self._stops = [self.tok.convert_tokens_to_ids(t) for t in (END_THINK, "<|im_end|>")]
        self._lock = threading.Lock()
        engine_kw.setdefault("sampling_backend", "pytorch")
        self.engine = sgl.Engine(model_path=ckpt, return_hidden_states_mode="last", mem_fraction_static=mem_fraction_static,
                                 context_length=context_length, log_level="warning", **engine_kw)

    # ---- decisions ----------------------------------------------------------------------------------

    def decide(self, request, image=None, temperature=None, reasoning=False, max_reasoning_tokens=768):
        """-> {"answers": {qid: answer}, "timing_ms"} (+ "reasoning", "reasoning_ms" when reasoning=True)."""
        t0 = time.perf_counter()
        extra = {}
        if reasoning:
            img = parse(request, image).image   # fail fast on a bad request, before generating
            text = "".join(self.think_stream(request, max_reasoning_tokens, image=img)).strip()
            extra = {"reasoning": text, "reasoning_ms": round((time.perf_counter() - t0) * 1000, 1)}
            request = with_reasoning(request, text)
        req = parse(request, image)
        enc = req.encode(self.tok, self.proc)
        T = temperature or (self.image_T if enc.has_image else self.text_T)
        z = self.logits(enc, self._sized(req.image, enc.grid) if enc.has_image else None)
        answers = {q.qid: to_answer(q, torch.softmax(zq / T, -1).tolist()) for q, zq in zip(req.questions, z)}
        return {"answers": answers, "timing_ms": round((time.perf_counter() - t0) * 1000, 1), **extra}

    def _sized(self, image, grid):
        """The image at exactly the size the processor encodes it at (grid = patches high x wide)."""
        from PIL import Image
        ps = self.proc.patch_size
        h, w = int(grid[0][1]) * ps, int(grid[0][2]) * ps
        return image if image.size == (w, h) else image.resize((w, h), Image.BICUBIC)

    @torch.no_grad()
    def logits(self, enc, image=None):
        """Encoded -> option logits per question, from "last" hidden states of the rows' prefix prompts."""
        if enc.has_image and image is None:
            raise ValueError("an image request needs its (sized) image")
        prompts, spans = prefix_prompts(enc, self.image_pad)
        kw = {"image_data": [image] * len(prompts)} if image is not None else {}
        with self._lock:
            outs = self.engine.generate(input_ids=prompts, sampling_params={"max_new_tokens": 1, "temperature": 0},
                                        return_hidden_states="last", **kw)
        h = torch.tensor([o["meta_info"]["hidden_states"] for o in outs], dtype=torch.float32)
        out = []
        for start, n in spans:
            hr = h[start:start + n]
            out.append(self.head(hr[-1:], hr[:-1], torch.zeros(n - 1, dtype=torch.long)))
        return out

    # ---- generation ---------------------------------------------------------------------------------

    def think_stream(self, request, max_new_tokens=768, image=None):
        """The model's reasoning about a request, streamed; stops at </think>."""
        if image is None:
            text = self.tok.apply_chat_template([{"role": "user", "content": prompt(request)}], tokenize=False,
                                                add_generation_prompt=True)
            kw = {}
        else:   # the image first, one placeholder, which the engine expands
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt(request)}]}]
            text = self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            _, grid, _ = image_input(image, self.proc, MAX_IMAGE_PIXELS, None)
            kw = {"image_data": self._sized(image, grid)}
        params = {"max_new_tokens": max_new_tokens, "temperature": self.reasoning_temperature, "top_p": 0.95,
                  "top_k": 20, "stop_token_ids": self._stops}
        with self._lock:
            stream = self.engine.generate(input_ids=self.tok(text).input_ids, sampling_params=params, stream=True, **kw)
        seen = ""
        while True:
            with self._lock:   # per chunk, never across a yield
                try:
                    chunk = next(stream)
                except StopIteration:
                    break
            t = chunk["text"]
            piece = t[len(seen):] if t.startswith(seen) else t   # cumulative or incremental chunks
            seen = t if t.startswith(seen) else seen + t
            piece = piece.replace(END_THINK, "")
            if piece:
                yield piece

    def chat(self, messages, tools=None, max_new_tokens=2048, temperature=0.6):
        """One assistant turn for OpenAI-style messages and tool schemas: the raw turn (thinking, then `</think>`, then
        the reply or a `<tool_call>` block), without the end-of-turn token."""
        text = self.tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
        params = {"max_new_tokens": max_new_tokens, "temperature": temperature, "top_p": 0.95, "top_k": 20,
                  "stop_token_ids": [self.tok.convert_tokens_to_ids("<|im_end|>")], "skip_special_tokens": False}
        with self._lock:
            out = self.engine.generate(input_ids=self.tok(text).input_ids, sampling_params=params)
        t = out["text"]
        for s in ("<|im_end|>", "<|endoftext|>"):
            t = t.replace(s, "")
        return t.strip()

    def shutdown(self):
        self.engine.shutdown()
