"""Typed decisions with the same request rows and answers as serve_decide.py.

The packaged decide_request.py is an exact copy of the shared serving contract, pinned
by a test. Both PyTorch zip heads and converted safetensors heads load without torch.
All model work runs on the server's MLX worker; questions branch from a shared state snapshot.
"""

from __future__ import annotations

import collections
import io
import json
import math
import pickle
import time
import zipfile
from dataclasses import replace
from pathlib import Path

import mlx.core as mx
import numpy as np

from .decide_request import MAX_STATE, RequestError, SPECIAL, VISION, parse, prompt, render, to_answer, with_reasoning
from .decode import DecodeStats, SamplingParams
from .engine import IncrementalDetokenizer
from .vision import image_key, preprocess


_STORAGE = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
            "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


def _rebuild_tensor(storage, offset, size, stride, *_):
    if (offset < 0 or len(size) != len(stride) or any(n < 0 for n in size)
            or any(s < 0 for s in stride)):
        raise ValueError("invalid tensor layout")
    end = offset + sum((n - 1) * s for n, s in zip(size, stride)) if all(size) else offset - 1
    if end >= storage.size:
        raise ValueError("tensor layout exceeds its storage")
    if math.prod(size) > storage.size:   # zero strides: a small storage viewed as a huge tensor
        raise ValueError("tensor is larger than its storage")
    if not size:
        return storage[offset].copy()
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * storage.itemsize for s in stride)).copy()


def load_torch_file(path) -> dict:
    """Read a ``torch.save`` zip file (tensors, dicts, numbers) into numpy arrays, without torch.
    Only the plain objects a state dict holds are accepted; anything else raises."""
    with zipfile.ZipFile(path) as z:
        pkl = next((n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl"), None)
        if pkl is None:
            raise ValueError(f"{path}: not a torch.save file (no data.pkl)")
        root = pkl[: -len("data.pkl")]

        class _Unpickler(pickle.Unpickler):
            def find_class(self, module, name):
                if module == "torch._utils" and name == "_rebuild_tensor_v2":
                    return _rebuild_tensor
                if module == "torch._utils" and name == "_rebuild_parameter":
                    return lambda data, *_: data
                if module == "torch" and name.endswith("Storage"):
                    return name
                if module == "collections" and name == "OrderedDict":
                    return collections.OrderedDict
                raise pickle.UnpicklingError(f"{path}: unsupported object {module}.{name}")

            def persistent_load(self, pid):
                _, storage_type, key, _location, _numel = pid
                data = z.read(f"{root}data/{key}")
                if storage_type == "BFloat16Storage":
                    return (np.frombuffer(data, np.uint16).astype(np.uint32) << 16).view(np.float32)
                if storage_type not in _STORAGE:
                    raise pickle.UnpicklingError(f"{path}: unsupported storage {storage_type}")
                return np.frombuffer(data, _STORAGE[storage_type]).copy()

        return _Unpickler(io.BytesIO(z.read(pkl))).load()


class PointerHead:
    def __init__(self, weights, hidden_size=None):
        names = {"q.weight", "q.bias", "k.weight", "k.bias"}
        if set(weights) != names:
            raise ValueError(f"decision head must contain {sorted(names)}")
        self.w = {k: v.astype(mx.float32) for k, v in weights.items()}
        shape = self.w["q.weight"].shape
        if (
            len(shape) != 2
            or min(shape) <= 0
            or (hidden_size is not None and shape[1] != hidden_size)
        ):
            raise ValueError("decision head hidden size does not match the model")
        if self.w["k.weight"].shape != shape or any(
            self.w[f"{k}.bias"].shape != (shape[0],) for k in ("q", "k")
        ):
            raise ValueError("invalid decision head shapes")
        if not all(mx.all(mx.isfinite(v)).item() for v in self.w.values()):
            raise ValueError("decision head contains non-finite weights")
        self.scale = 1 / math.sqrt(shape[0])

    def __call__(self, h_decide, h_opts):
        q = h_decide.astype(mx.float32) @ self.w["q.weight"].T + self.w["q.bias"]
        k = h_opts.astype(mx.float32) @ self.w["k.weight"].T + self.w["k.bias"]
        return mx.sum(k * q, axis=-1) * self.scale


def load_head(directory, hidden_size=None):
    directory = Path(directory)
    safe, pt = directory / "jev_head.safetensors", directory / "jev_head.pt"
    if safe.exists():
        weights = mx.load(str(safe))
    elif pt.exists():
        saved = load_torch_file(pt)
        weights = {k: mx.array(v) for k, v in saved["head"].items()}
        if weights["q.weight"].shape[0] != saved["head_dim"]:
            raise ValueError("decision head_dim does not match its weights")
    else:
        return None
    return PointerHead(weights, hidden_size)


class Decider:
    def __init__(self, engine, context_length=8192, text_temperature=None, image_temperature=None,
                 cache_mb=1024):
        self.engine, self.context_length = engine, context_length
        self.head = load_head(engine.path, engine.rt.args.hidden_size)
        self.temperatures = {"text": 1.52, "image": 1.0}
        path = engine.path / "temperature.json"
        if path.exists():
            saved = json.loads(path.read_text())
            self.temperatures.update({k: float(saved[k]) for k in self.temperatures if k in saved})
        for key, value in (("text", text_temperature), ("image", image_temperature)):
            if value is not None:
                self.temperatures[key] = float(value)
        self.temps = self.temperatures
        self.state_cache = collections.OrderedDict()
        if not math.isfinite(cache_mb) or cache_mb < 0:
            raise ValueError("decision cache size must be finite and non-negative")
        self.cache_bytes = int(cache_mb * 1024 * 1024)
        self.cache_reused_tokens = 0
        self.reused_keys, self._hit = set(), False   # keys of cached states built on a cache hit
        if any(not math.isfinite(t) or t <= 0 for t in self.temperatures.values()):
            raise ValueError("decision temperatures must be finite and positive")

    def _image(self, req):
        if req.image is None:
            return None
        eng = self.engine
        if not eng.supports_vision:
            raise RequestError("this checkpoint has no vision encoder")
        cfg = replace(
            eng.image_proc, max_pixels=min(eng.image_proc.max_pixels, req.max_image_pixels)
        )
        cfg.min_pixels = min(req.min_image_pixels or cfg.min_pixels, cfg.max_pixels)
        data = io.BytesIO()
        req.image.save(data, format="PNG")
        key = image_key(data.getvalue(), cfg)
        pixels, grid = preprocess(req.image, cfg)
        # Use the existing vision runtime; no changes to the chat image cache/config.
        if key not in eng.rt.images:
            eng.rt.images[key] = {"pixels": pixels, "grid": grid, "features": None}
        return key, grid[0] * (grid[1] // cfg.merge_size) * (grid[2] // cfg.merge_size)

    def encode(self, req, image):
        rows = req.encode_rows(self.engine.tokenizer, image[1] if image else 0)
        if image:
            pad = self.engine.tokenizer.convert_tokens_to_ids(VISION[1])
            for row in rows:
                row["ids"] = [image[0] if t == pad else t for t in row["ids"]]
        if any(len(row["ids"]) > self.context_length for row in rows):
            raise RequestError(f"decision row exceeds context length {self.context_length}")
        return rows

    @classmethod
    def load(cls, engine, text_temperature=None, image_temperature=None, **kwargs):
        decider = cls(engine, text_temperature=text_temperature,
                      image_temperature=image_temperature, **kwargs)
        return decider if decider.head is not None else None

    def _state_cache(self, tokens):
        """Reuse the longest complete cached state that prefixes this state. Worker-thread only."""
        rt = self.engine.rt
        found = None
        for key in self.state_cache:
            if len(key) <= len(tokens) and tuple(tokens[:len(key)]) == key:
                if found is None or len(key) > len(found):
                    found = key
        self.cache_reused_tokens = len(found) if found else 0
        self._hit = found is not None
        cache = self.state_cache.pop(found) if found else rt.make_cache()
        self.reused_keys.discard(found)
        if len(tokens) > cache.length:
            tail = tokens[cache.length:]
            rt.prefill(cache, tail, hidden_positions=[len(tail) - 1])
        return cache

    def _remember_state(self, cache):
        """Cache `cache` under its tokens, then evict down to the budget. Entries never reused go first, newest
        first (keeping the one just added while others are left), then reused ones, least recently used first.
        Plain LRU would evict exactly the state a cycle needs next and never hit once the cycle outgrows the
        budget; this keeps all but one of the cycle's states, and the chat-like case (each state extends a
        reused one) still behaves as LRU."""
        if not self.cache_bytes:
            return
        key = tuple(cache.tokens)
        self.state_cache[key] = cache
        if self._hit:
            self.reused_keys.add(key)
        while sum(c.nbytes() for c in self.state_cache.values()) > self.cache_bytes:
            cold = [k for k in self.state_cache if k not in self.reused_keys and k != key]
            victim = cold[-1] if cold else next(k for k in self.state_cache if k != key or len(self.state_cache) == 1)
            del self.state_cache[victim]
            self.reused_keys.discard(victim)

    def clear_cache(self):
        self.state_cache.clear()
        self.reused_keys.clear()

    def logits(self, rows):
        rt = self.engine.rt
        # User text cannot spell this special token: encode_rows escapes it.
        state_len = rows[0]["ids"].index(self.engine.tokenizer.convert_tokens_to_ids(SPECIAL[1]))
        cache = self._state_cache(rows[0]["ids"][:state_len])
        snap, out = cache.snapshot(), []
        try:
            for row in rows:
                assert row["ids"][:state_len] == rows[0]["ids"][:state_len]
                cache.restore(snap)
                h = rt.prefill(
                    cache, row["ids"][state_len:],
                    hidden_positions=[p - state_len for p in row["opts"] + [row["decide"]]],
                )
                out.append(self.head(h[-1], h[:-1]))
            mx.eval(out)
            return out
        finally:
            cache.restore(snap)
            self._remember_state(cache)

    def think(self, request, image, max_tokens, on_piece=None, mode="self-spec"):
        eng = self.engine
        content = prompt(request)
        if image:
            content = [{"type": "image"}, {"type": "text", "text": content}]
        ids = eng.encode(
            eng.render([{"role": "user", "content": content}], enable_thinking=True),
            [image] if image else None,
        )
        if len(ids) >= self.context_length:
            raise RequestError("reasoning prompt exceeds context length")
        detok, text, stats = IncrementalDetokenizer(eng.tokenizer), "", DecodeStats()
        gen = eng.generate_ids(
            ids, max_tokens=min(max_tokens, self.context_length - len(ids)),
            sampling=SamplingParams(temperature=0.6, top_p=0.95, top_k=20),
            mode=mode, stats=stats,
            stop_ids={eng.tokenizer.convert_tokens_to_ids("</think>")},
        )
        previous = eng.cur, eng.slots, eng.use_prompt_cache
        eng.use_prompt_cache = False
        try:
            for chunk in gen:
                piece = detok.add(chunk)
                text += piece
                if on_piece is not None:
                    on_piece(piece)
        finally:
            gen.close()
            eng.cur, eng.slots, eng.use_prompt_cache = previous
        return {"text": text.replace("</think>", "").strip(),
                "ended": "complete" if stats.finish_reason == "stop" else "budget",
                "stats": stats.as_dict()}

    def fit_reasoning(self, request, text, image=None):
        """Retain the conclusion when reasoning would exceed the shared state budget."""
        tok = self.engine.tokenizer
        image_tokens = image[1] + 2 if image else 0
        room = MAX_STATE - 1 - image_tokens

        def fits(candidate):
            state = render(with_reasoning(request, candidate)["state"])
            # Match the canonical encoder's escaping and tokenization.
            from .decide_request import _user
            return len(_user(tok, state)) <= room

        if fits(text):
            return text
        ids = tok.encode(text, add_special_tokens=False)
        lo, hi = 0, len(ids) + 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if fits("…" + tok.decode(ids[-mid:])):
                lo = mid
            else:
                hi = mid
        return "…" + tok.decode(ids[-lo:]) if lo else ""

    def decide(
        self,
        request,
        *,
        temperature=None,
        reasoning=False,
        max_reasoning_tokens=None,
        threshold=0.7,
        on_event=None,
        mode=None,
    ):
        """Decide on the MLX worker; direct callers must hold engine.lock on its owning thread."""
        if self.head is None:
            raise RequestError(
                "this checkpoint has no decision head (jev_head.pt or jev_head.safetensors)"
            )
        if not isinstance(reasoning, bool) and reasoning != "auto":
            raise RequestError('reasoning must be a boolean or "auto"')
        if max_reasoning_tokens is None:
            max_reasoning_tokens = 8192 if reasoning == "auto" else 768
        if (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                or not math.isfinite(threshold) or not 0 <= threshold <= 1):
            raise RequestError("threshold must be between 0 and 1")
        if mode not in (None, "causal", "self-spec"):
            raise RequestError("trida_mode must be causal or self-spec")
        if (
            not isinstance(max_reasoning_tokens, int)
            or isinstance(max_reasoning_tokens, bool)
            or max_reasoning_tokens <= 0
        ):
            raise RequestError("max_reasoning_tokens must be a positive integer")
        if temperature is not None and (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(temperature)
            or temperature <= 0
        ):
            raise RequestError("temperature must be finite and positive")
        req = parse(request)
        start = time.perf_counter()
        existing_images = set(self.engine.rt.images)
        image = self._image(req)
        emit = on_event or (lambda *_: None)
        try:
            rows = self.encode(req, image)
            t = temperature if temperature is not None else self.temperatures["image" if image else "text"]

            def score(parsed, encoded):
                return {
                    "answers": {q.qid: to_answer(q, mx.softmax(z / t).tolist())
                                for q, z in zip(parsed.questions, self.logits(encoded))},
                    "timing_ms": round((time.perf_counter() - start) * 1000, 1),
                }

            if reasoning is False:
                result = score(req, rows)
                emit("direct", result)
                emit("result", result)
                return result

            direct = score(req, rows)
            result = {**direct, "direct": direct, "escalated": reasoning is True,
                      "threshold": threshold}
            if reasoning == "auto":
                qid, top = min(((qid, top_probability(a)) for qid, a in direct["answers"].items()),
                               key=lambda item: item[1])
                result["lowest"] = {"question": qid, "top_probability": top}
                result["escalated"] = top < threshold
            if result["escalated"]:
                result["reason"] = (
                    f"reasoned because top probability was {top:.4g} (question {qid}), below {threshold}"
                    if reasoning == "auto" else "reasoning requested"
                )
            emit("direct", {**direct, **({"reason": result["reason"]} if result["escalated"] else {})})
            if not result["escalated"]:
                emit("result", result)
                return result
            thinking_start = time.perf_counter()
            thought = self.think(request, image, max_reasoning_tokens,
                                 on_piece=lambda piece: emit("piece", piece),
                                 mode=mode or "self-spec")
            result.update(reasoning=thought["text"], reasoning_stats=thought["stats"],
                          reasoning_ms=round((time.perf_counter() - thinking_start) * 1000, 1))
            if thought["ended"] != "complete":
                result["cut"] = "reasoning did not finish; using the direct answer"
            else:
                text = self.fit_reasoning(request, thought["text"], image)
                if text != thought["text"]:
                    result["reasoning_trimmed"] = True
                reasoned = parse(with_reasoning(request, text), image=req.image)
                result["reasoned"] = score(reasoned, self.encode(reasoned, image))
                result["answers"] = result["reasoned"]["answers"]
            result["timing_ms"] = round((time.perf_counter() - start) * 1000, 1)
            emit("result", result)
            return result
        finally:
            if image and image[0] not in existing_images:
                self.engine.rt.images.pop(image[0], None)


def top_probability(answer):
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())
