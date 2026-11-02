"""Typed decisions (Jev) on MLX: the same Trida checkpoint that generates text also answers typed
questions with probabilities, through its pointer head (``jev_head.pt``).

A request is ``{"state": any JSON, "questions": {qid: question}}`` (optionally with an image), the
format the demos and trida-stack's ``inference/jev.py`` use. A question is ``noul`` (yes/no),
``choice`` (named options) or ``score`` (ordered levels). Each question becomes one causal row

    <state> [image] state text <q> instructions <opt> o1 </opt> ... <decide>

and the pointer head scores every ``</opt>`` hidden state against the ``<decide>`` one. Rows share
their state, so the state is prefilled once and each question's branch runs from a snapshot of it.

Reasoned decisions: the model first writes its reasoning (generation, self-spec by default, stopping
at ``</think>``), the reasoning is added to the state, and the head decides again. ``decide_or_reason``
does that only when the direct answer is unsure (top probability below 0.7, the Zoro-1 model card's
rule), which is how the demos escalate.

Everything here runs on the server's MLX worker thread.
"""
from __future__ import annotations

import collections
import io
import json
import math
import pickle
import re
import sys
import time
import zipfile
from pathlib import Path
from typing import Callable, Optional

import mlx.core as mx
import numpy as np

from . import jev
from .decode import DecodeStats, SamplingParams

_CONTEXT = jev.training_context(7552)          # the vision run's state limit, as the public demo uses
MAX_STATE, MAX_BRANCH = _CONTEXT["max_state"], _CONTEXT["max_branch"]   # 7552, 8192
MAX_IMAGE_PIXELS = 448 * 448                    # images are decided on at the training size
MAX_OPTIONS = 255
QTYPES = ("noul", "choice", "score")
THRESHOLD = 0.7                                 # Zoro-1 card: reason when the top probability is below 0.7
DEFAULT_TEXT_TEMPERATURE = 1.52                 # Zoro-1 card (revision ab05e6c): fitted text temperature
DEFAULT_IMAGE_TEMPERATURE = 1.0                 # none fitted for Zoro-1: raw head probabilities
REASONING_SAMPLING = dict(temperature=0.6, top_p=0.95, top_k=20)   # the card's reasoned-action setting
INSTRUCTION = ("Think it through step by step: work out anything that needs calculating or checking, then say which "
               "option you choose for each question.")
# Tag-shaped special tokens that caller text must not spell (besides <|name|>, which jev.user_tokens rewrites).
TAG_TOKENS = ["<think>", "</think>", "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>",
              "<tts_pad>", "<tts_text_bos>", "<tts_text_eod>", "<tts_text_bos_single>"]
_TAG_RE = re.compile("|".join(re.escape(t) for t in sorted(TAG_TOKENS, key=len, reverse=True)))


class DecisionError(ValueError):
    """A request the decision head cannot take; the message says what is wrong."""


def escape(text: str) -> str:
    """``<|name|>`` -> ``<¦name¦>`` and ``<think>`` -> ``<¦think¦>`` (etc.), so text can't spell a special token."""
    return _TAG_RE.sub(lambda m: "<¦" + m.group(0)[1:-1] + "¦>", jev._SPECIAL_RE.sub(r"<¦\1¦>", text))


# ---- jev_head.pt without torch --------------------------------------------------------------------

_STORAGE = {"FloatStorage": np.float32, "DoubleStorage": np.float64, "HalfStorage": np.float16,
            "LongStorage": np.int64, "IntStorage": np.int32, "BoolStorage": np.bool_}


def _rebuild_tensor(storage, offset, size, stride, *_):
    if not size:
        return storage[offset].copy()
    return np.lib.stride_tricks.as_strided(storage[offset:], shape=tuple(size),
                                           strides=tuple(s * storage.itemsize for s in stride)).copy()


def load_torch_file(path) -> dict:
    """Read a ``torch.save`` zip file (tensors, dicts, numbers) into numpy arrays, without torch.
    Only the plain objects a state dict holds are accepted; anything else raises."""
    with zipfile.ZipFile(path) as z:
        pkl = next(n for n in z.namelist() if n.endswith("/data.pkl") or n == "data.pkl")
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
    """kev's pointer head: logits[k] = <Wk h_opt[k] + bk, Wq h_decide + bq> / sqrt(dp)."""

    def __init__(self, wq, bq, wk, bk):
        self.wq, self.bq = mx.array(np.asarray(wq, np.float32)), mx.array(np.asarray(bq, np.float32))
        self.wk, self.bk = mx.array(np.asarray(wk, np.float32)), mx.array(np.asarray(bk, np.float32))
        self.dim = int(self.wq.shape[0])
        self.scale = 1 / math.sqrt(self.dim)

    def __call__(self, h_decide: mx.array, h_opts: mx.array) -> mx.array:   # [D], [K, D] -> [K]
        q = h_decide @ self.wq.T + self.bq
        k = h_opts @ self.wk.T + self.bk
        return (k * q).sum(-1) * self.scale


def load_head(model_dir) -> Optional[PointerHead]:
    """The checkpoint's pointer head from ``jev_head.safetensors`` or ``jev_head.pt``, or None."""
    d = Path(model_dir)
    if (d / "jev_head.safetensors").is_file():
        w = {k: np.array(v) for k, v in mx.load(str(d / "jev_head.safetensors")).items()}
    elif (d / "jev_head.pt").is_file():
        saved = load_torch_file(d / "jev_head.pt")
        w = saved.get("head", saved) if isinstance(saved, dict) else None
        if not isinstance(w, dict):
            raise DecisionError(f"{d / 'jev_head.pt'}: expected {{'head': state_dict, 'head_dim': dp}}")
    else:
        return None
    try:
        return PointerHead(w["q.weight"], w["q.bias"], w["k.weight"], w["k.bias"])
    except KeyError as e:
        raise DecisionError(f"{d}: the pointer head has no {e} weight") from None


def temperatures(model_dir, text: Optional[float] = None, image: Optional[float] = None) -> dict:
    """Serving temperatures (decisions divide the head's logits by them): explicit values win, then the
    checkpoint's temperature.json, then the Zoro-1 card's text value (1.52) and 1.0 for images."""
    t = {"text": DEFAULT_TEXT_TEMPERATURE, "image": DEFAULT_IMAGE_TEMPERATURE, "source": "default (Zoro-1 card)"}
    f = Path(model_dir) / "temperature.json"
    if f.is_file():
        d = json.loads(f.read_text())
        t.update({k: float(d[k]) for k in ("text", "image") if k in d}, source=str(f))
    if text is not None:
        t["text"], t["source"] = float(text), "explicit"
    if image is not None:
        t["image"] = float(image)
    for k in ("text", "image"):
        if not (t[k] > 0 and math.isfinite(t[k])):
            raise DecisionError(f"the {k} temperature must be a number > 0, got {t[k]!r}")
    return t


# ---- requests -------------------------------------------------------------------------------------

def _question(qid, q) -> dict:
    where = f"question {qid!r}"
    if not isinstance(q, dict) or q.get("type") not in QTYPES:
        raise DecisionError(f"{where}: needs a type ({', '.join(QTYPES)}) and criteria")
    t, c = q["type"], q.get("criteria")
    if t == "noul":
        c = c or {}
        if not isinstance(c, dict) or set(c) - {"true", "false"}:
            raise DecisionError(f"{where}: noul criteria are optional and only have 'true' / 'false'")
        opts = [jev.option_text("no", c.get("false")), jev.option_text("yes", c.get("true"))]
    elif t == "choice":
        if not isinstance(c, dict) or not c:
            raise DecisionError(f"{where}: choice criteria must be an object {{option key: description}}")
        opts = [jev.option_text(k, v) for k, v in c.items()]
    else:
        if not isinstance(c, list) or not c:
            raise DecisionError(f"{where}: score criteria must be a list of levels, lowest first")
        opts = [jev.render(x) for x in c]
    if len(opts) > MAX_OPTIONS:
        raise DecisionError(f"{where}: {len(opts)} options; at most {MAX_OPTIONS}")
    return {"qid": qid, "qtype": t, "instr": jev.render(q.get("instructions")), "options": opts,
            "keys": jev.question_keys(t, c), "levels": c if t == "score" else None}


def parse(request: dict) -> tuple:
    """-> (rendered state text, [question dicts]). DecisionError on a malformed request."""
    if not isinstance(request, dict) or "state" not in request:
        raise DecisionError("a decision request is {'state': ..., 'questions': {qid: question}}")
    qs = request.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise DecisionError("'questions' must be a non-empty object {question id: question}")
    return jev.render(request["state"]), [_question(qid, q) for qid, q in qs.items()]


def reasoning_prompt(request: dict) -> str:
    """A decision request as one user message: the state, then each question with its options."""
    parts = [jev.render(request["state"])]
    for qid, q in request["questions"].items():
        lines = [f"Question ({qid}): {jev.render(q.get('instructions')) or qid}"]
        if q["type"] == "noul":
            lines.append("Answer yes or no.")
        elif q["type"] == "choice":
            lines += ["Options:"] + [f"- {k}: {jev.render(v)}" if v not in (None, "") else f"- {k}"
                                     for k, v in q["criteria"].items()]
        else:
            lines += ["Levels, lowest first:"] + [f"{i}. {jev.render(v)}" for i, v in enumerate(q["criteria"])]
        parts.append("\n".join(lines))
    return "\n\n".join(parts + [INSTRUCTION])


def with_reasoning(request: dict, text: str) -> dict:
    """The request with the model's reasoning added to its state (as ``reasoning``)."""
    state = request["state"]
    state = {**state, "reasoning": text} if isinstance(state, dict) else {"situation": state, "reasoning": text}
    return {**request, "state": state}


# ---- answers --------------------------------------------------------------------------------------

def _confidence(qtype, p):
    K = len(p)
    if K == 1:
        return 1.0
    if qtype == "score":
        mode = max(range(K), key=p.__getitem__)
        D = sum(abs(i - (K - 1) / 2) for i in range(K)) / K
        return max(0.0, 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / D)
    return (max(p) - 1 / K) / (1 - 1 / K)


def to_answer(q: dict, p: list) -> dict:
    r = lambda x: round(float(x), 4)  # noqa: E731
    if q["qtype"] == "noul":
        return {"type": "noul", "noul": r(p[1])}
    if q["qtype"] == "choice":
        return {"type": "choice", "choice": q["keys"][max(range(len(p)), key=p.__getitem__)],
                "confidence": r(_confidence("choice", p)), "probabilities": {k: r(v) for k, v in zip(q["keys"], p)}}
    return {"type": "score", "score": r(sum(i * x for i, x in enumerate(p))), "legend": q["levels"],
            "confidence": r(_confidence("score", p)), "probabilities": {str(i): r(v) for i, v in enumerate(p)}}


def top_probability(answer: dict) -> float:
    """The highest option probability (for noul max(p, 1 - p)): what escalation compares to the threshold."""
    if answer["type"] == "noul":
        return max(answer["noul"], 1 - answer["noul"])
    return max(answer["probabilities"].values())


def softmax(z, temperature: float) -> list:
    z = np.asarray(z, np.float64) / temperature
    z = np.exp(z - z.max())
    return (z / z.sum()).tolist()


# ---- the decider ----------------------------------------------------------------------------------

class Decider:
    """Typed and reasoned decisions with the engine's model. Call from the MLX worker thread."""

    def __init__(self, engine, head: PointerHead, temps: dict):
        self.engine, self.head, self.temps = engine, head, temps
        tok = engine.tokenizer
        self.image_pad = tok.convert_tokens_to_ids(jev.VISION[1])
        self.think_end = tok.convert_tokens_to_ids("</think>")

    @classmethod
    def load(cls, engine, text_temperature=None, image_temperature=None) -> Optional["Decider"]:
        head = load_head(engine.path)
        if head is None:
            return None
        hidden = engine.rt.args.hidden_size
        if head.wq.shape[1] != hidden:
            raise DecisionError(f"jev head expects hidden size {head.wq.shape[1]}, the model has {hidden}")
        return cls(engine, head, temperatures(engine.path, text_temperature, image_temperature))

    # rows ------------------------------------------------------------------------------------------
    def encode(self, request: dict, image=None) -> tuple:
        """-> (questions, rows, state_len). Rows are jev.encode_rows' (image pad ids replaced by the
        engine's image key). The state must fit (never cut), each question its row."""
        state, qs = parse(request)
        tok = self.engine.tokenizer
        image_tokens, key = 0, None
        if image is not None:
            if not self.engine.supports_vision:
                raise DecisionError("this checkpoint has no vision encoder; send the request without an image")
            key, image_tokens = self.engine.add_image(image, max_pixels=request.get("max_image_pixels")
                                                      or MAX_IMAGE_PIXELS)
        img = image_tokens + 2 if image_tokens else 0
        text = jev.user_tokens(tok, escape(state))
        room = MAX_STATE - 1 - img
        if len(text) > room:
            raise DecisionError(f"the state is {len(text)} tokens; at most {room} fit (state limit {MAX_STATE})")
        record = {"state": escape(state), "questions": [
            {"instr": escape(q["instr"]), "options": [escape(o) for o in q["options"]], "label": None,
             "qtype": q["qtype"]} for q in qs]}
        rows = jev.encode_rows(tok, record, MAX_STATE, sys.maxsize, image_tokens=image_tokens)
        state_len = 1 + img + len(text)
        for q, row in zip(qs, rows):
            if len(row["ids"]) - state_len > MAX_BRANCH - state_len:
                raise DecisionError(f"question {q['qid']!r} is too long for its row; shorten it")
            if key is not None:
                row["ids"] = [key if t == self.image_pad else t for t in row["ids"]]
        return qs, rows, state_len

    def logits(self, rows: list, state_len: int) -> list:
        """Option logits per row: the shared state is prefilled once, each branch runs from its snapshot."""
        rt = self.engine.rt
        cache = rt.make_cache()
        rt.prefill(cache, rows[0]["ids"][:state_len])
        snap = cache.snapshot()
        out = []
        for r in rows:
            assert r["ids"][:state_len] == rows[0]["ids"][:state_len]
            cache.restore(snap)
            pos = [i - state_len for i in r["opts"]] + [r["decide"] - state_len]
            h = rt.prefill(cache, r["ids"][state_len:], hidden_positions=pos)   # [K + 1, D]
            out.append(self.head(h[-1], h[:-1]))
        mx.eval(out)
        return [np.array(z) for z in out]

    def decide(self, request: dict, image=None, temperature: Optional[float] = None) -> dict:
        """-> {"answers": {qid: answer}, "timing_ms", "temperature"}."""
        t0 = time.perf_counter()
        qs, rows, state_len = self.encode(request, image)
        T = float(temperature) if temperature else self.temps["image" if image is not None else "text"]
        answers = {q["qid"]: to_answer(q, softmax(z, T)) for q, z in zip(qs, self.logits(rows, state_len))}
        return {"answers": answers, "timing_ms": round((time.perf_counter() - t0) * 1000, 1), "temperature": T}

    # reasoning ---------------------------------------------------------------------------------------
    def think(self, request: dict, image=None, max_tokens: int = 8192, on_piece: Optional[Callable] = None,
              mode: Optional[str] = None, sampling: Optional[SamplingParams] = None) -> dict:
        """The model's reasoning about a request (generation, thinking on, stops at ``</think>``).
        -> {"text", "ended": "complete" | "budget", "stats"}. ``on_piece(text)`` streams it."""
        from .engine import IncrementalDetokenizer
        eng = self.engine
        content = escape(reasoning_prompt(request))
        images = None
        if image is not None:
            images = [eng.add_image(image, max_pixels=request.get("max_image_pixels") or MAX_IMAGE_PIXELS)]
            content = [{"type": "image"}, {"type": "text", "text": content}]
        prompt = eng.encode(eng.render([{"role": "user", "content": content}], enable_thinking=True), images)
        stats = DecodeStats()
        detok = IncrementalDetokenizer(eng.tokenizer)
        text = ""
        sp = sampling or SamplingParams(**REASONING_SAMPLING)
        for ids in eng.generate_ids(prompt, max_tokens=max_tokens, sampling=sp, mode=mode, stats=stats,
                                    stop_ids={self.think_end}):
            d = detok.add(ids)
            if d:
                text += d
                if on_piece is not None:
                    on_piece(d)
        ended = "complete" if stats.finish_reason == "stop" else "budget"
        return {"text": text.replace("</think>", "").strip(), "ended": ended, "stats": stats.as_dict()}

    def _state_room(self, request: dict, image) -> tuple:
        """(tokens the request's state takes, tokens that fit) as encode() counts them."""
        state, _ = parse(request)
        img = 0
        if image is not None and self.engine.supports_vision:
            _, n = self.engine.add_image(image, max_pixels=request.get("max_image_pixels") or MAX_IMAGE_PIXELS)
            img = n + 2
        return len(jev.user_tokens(self.engine.tokenizer, escape(state))), MAX_STATE - 1 - img

    def fit_reasoning(self, request: dict, text: str, image=None) -> str:
        """``text`` itself when the state with it fits, else its longest tail (the conclusion is at the end)
        that does, prefixed with an ellipsis. Long reasoning (up to 8192 tokens) can exceed MAX_STATE."""
        used, room = self._state_room(with_reasoning(request, text), image)
        if used <= room:
            return text
        tok = self.engine.tokenizer
        ids = tok.encode(text, add_special_tokens=False)
        lo, hi = 0, len(ids)  # number of tail tokens kept: lo fits, hi does not
        while hi - lo > 1:
            mid = (lo + hi) // 2
            cand = "…" + tok.decode(ids[len(ids) - mid:])
            if self._state_room(with_reasoning(request, cand), image)[0] <= room:
                lo = mid
            else:
                hi = mid
        return "…" + tok.decode(ids[len(ids) - lo:]) if lo else ""

    def decide_or_reason(self, request: dict, image=None, *, reasoning: str = "auto", threshold: float = THRESHOLD,
                         max_reasoning_tokens: int = 8192, temperature: Optional[float] = None,
                         on_event: Optional[Callable] = None, mode: Optional[str] = None) -> dict:
        """Direct answer first; with reasoning="auto" reason once and decide again when any question's top
        probability is below ``threshold`` ("always": always reason; "never": direct only). Cut-off
        reasoning is never decided on. ``on_event(kind, payload)`` streams "direct", "piece", "result".
        -> {"answers" (the answer to use), "direct", "escalated", "reason"?, "reasoning"?, "reasoned"?, "cut"?}."""
        if reasoning not in ("auto", "always", "never"):
            raise DecisionError("reasoning must be 'auto', 'always' or 'never'")
        emit = on_event or (lambda *_: None)
        direct = self.decide(request, image, temperature)
        qid, top = min(((q, top_probability(a)) for q, a in direct["answers"].items()), key=lambda t: t[1])
        result = {"answers": direct["answers"], "direct": direct, "escalated": False,
                  "lowest": {"question": qid, "top_probability": round(top, 4)}, "threshold": threshold}
        if reasoning == "never" or (reasoning == "auto" and top >= threshold):
            emit("direct", direct)
            emit("result", result)
            return result
        why = (f"reasoned because top probability was {top:.4g} (question {qid}), below {threshold}"
               if reasoning == "auto" else "reasoning requested")
        result.update(escalated=True, reason=why)
        emit("direct", {**direct, "reason": why})
        r = self.think(request, image, max_reasoning_tokens, on_piece=lambda p: emit("piece", p), mode=mode)
        result.update(reasoning=r["text"], reasoning_stats=r["stats"])
        if r["ended"] != "complete":
            looped = r["stats"].get("finish_reason") == "repetition"
            result["cut"] = ("the reasoning went in circles (loop guard) before finishing; using the direct answer"
                             if looped else f"the reasoning hit the {max_reasoning_tokens}-token budget before "
                             f"finishing; using the direct answer")
            emit("result", result)
            return result
        text = self.fit_reasoning(request, r["text"], image)
        if text != r["text"]:
            result["reasoning_trimmed"] = (f"the reasoning did not fit the decision state ({MAX_STATE} tokens); "
                                           f"its last {len(self.engine.tokenizer.encode(text, add_special_tokens=False))} "
                                           f"tokens were kept")
        reasoned = self.decide(with_reasoning(request, text), image, temperature)
        result.update(reasoned=reasoned, answers=reasoned["answers"])
        emit("result", result)
        return result
