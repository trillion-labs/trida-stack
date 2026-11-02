"""Decision requests: what a Trida decision checkpoint sees, validated and encoded into token rows.

    from decide_request import parse, RequestError
    req = parse({"state": ..., "questions": {...}}, image=None)   # RequestError on anything malformed
    enc = req.encode(tokenizer, image_processor)                 # token rows (+ image tensors)

A request is `{"state": any JSON, "questions": {qid: question}}` plus, optionally:
- `image`: a PIL image, a path, or a base64 / data-URL string (the `image` argument wins over the key);
- `min_image_pixels` / `max_image_pixels`: the image's size bounds (default: no minimum, 448x448 maximum).

A question is `{"type": "noul" | "choice" | "score", "instructions"?, "criteria"}`:
- `noul`: yes/no; optional criteria `{"true": ..., "false": ...}`;
- `choice`: `{key: description}`;
- `score`: a list of levels, lowest first.

Each question needs 1-255 options and becomes one causal token row:

    <state> [<|vision_start|> <|image_pad|> x n <|vision_end|>] state text <q> instructions <opt> o1 </opt> ... <decide>

The model's decision head reads the last-layer hidden states at each option's `</opt>` and at `<decide>`
(decide_engine.py). This is the format decision checkpoints are trained on; the rendering and row encoding follow
kev (https://github.com/jaredpalmer/kev, Apache-2.0), see NOTICE. `prompt` / `with_reasoning` at the end turn a
request into a chat prompt for reason-then-decide.
"""
import base64
import io
import os
import re
from dataclasses import dataclass, field

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]  # state q opt /opt decide
VISION = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]
MAX_STATE, MAX_BRANCH = 7552, 8192   # state tokens, and state + one question, as in training
MAX_IMAGE_PIXELS = 448 * 448
MAX_OPTIONS = 255
QTYPES = ("noul", "choice", "score")
KEYS = {"state", "questions", "image", "min_image_pixels", "max_image_pixels"}
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class RequestError(ValueError):
    """A request the model cannot take. The message says what is wrong and where; it is shown to whoever sent it."""


# ---- rendering ------------------------------------------------------------------------------------

def render(v, indent=0):
    """Flatten str | object | array into the text the model sees. Field names are kept as labels."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
                     for k, x in v.items())


def option_text(name, desc):
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def _json_like(v, where):
    if v is None or isinstance(v, (str, int, float, bool)):
        return
    if isinstance(v, list):
        for i, x in enumerate(v):
            _json_like(x, f"{where}[{i}]")
    elif isinstance(v, dict):
        for k, x in v.items():
            if not isinstance(k, str):
                raise RequestError(f"{where}: keys must be strings, got {k!r}")
            _json_like(x, f"{where}.{k}")
    else:
        raise RequestError(f"{where}: {type(v).__name__} is not JSON (use strings, numbers, booleans, lists, objects)")


# ---- the parsed request ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Question:
    qid: str
    qtype: str
    instr: str          # rendered instructions
    options: list       # rendered option texts, in order
    keys: list          # the key each option's probability is reported under


@dataclass
class Encoded:
    """Token rows, one per question: {"ids", "decide": index of <decide>, "opts": index of each </opt>}, plus the
    image's pixel values and grid when the request has an image."""
    rows: list
    pixel_values: object = None
    grid: object = None

    @property
    def has_image(self):
        return self.pixel_values is not None


@dataclass
class Request:
    state: str                     # rendered state text
    questions: list                # [Question], in request order
    image: object = None           # PIL image (RGB) or None
    min_image_pixels: int | None = None
    max_image_pixels: int = MAX_IMAGE_PIXELS
    raw: dict = field(default_factory=dict, repr=False)

    def encode(self, tokenizer, image_processor=None):
        """-> Encoded. The state is cut to fit MAX_STATE (the image counts as its token count + 2); a question that
        does not fit its row raises RequestError."""
        pixel_values = grid = None
        image = []
        if self.image is not None:
            if image_processor is None:
                raise RequestError("the request has an image but this checkpoint has no image processor (not a VL model)")
            pixel_values, grid, n = image_input(self.image, image_processor, self.max_image_pixels, self.min_image_pixels)
            vs, pad, ve = (tokenizer.convert_tokens_to_ids(t) for t in VISION)
            image = [vs] + [pad] * n + [ve]
        s_id, q_id, o_id, c_id, d_id = (tokenizer.convert_tokens_to_ids(t) for t in SPECIAL)
        state = [s_id] + image + _user(tokenizer, self.state)[: max(0, MAX_STATE - 1 - len(image))]
        rows = []
        for q in self.questions:
            instr = [q_id] + _user(tokenizer, q.instr)
            spans = [[o_id] + _user(tokenizer, o) + [c_id] for o in q.options]
            branch = instr + [t for sp in spans for t in sp] + [d_id]
            if len(branch) > MAX_BRANCH - len(state):
                raise RequestError(f"question {q.qid!r} is too long: {len(branch)} tokens of instructions and options "
                                   f"with a {len(state)}-token state (at most {MAX_BRANCH} per question); shorten it")
            ends, cursor = [], len(state) + len(instr)
            for sp in spans:
                cursor += len(sp)
                ends.append(cursor - 1)
            rows.append({"ids": state + branch, "decide": len(state) + len(branch) - 1, "opts": ends})
        return Encoded(rows, pixel_values, grid)


def _user(tok, text):
    """Caller text can never produce delimiter/control tokens: `<|name|>` is rewritten to `<¦name¦>` first."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def image_input(image, proc, max_pixels, min_pixels):
    """An in-memory image -> (pixel_values, grid [1, 3], number of image tokens), at the training resolution cap."""
    size = dict(proc.size)
    size["longest_edge"] = min(size.get("longest_edge") or max_pixels, max_pixels)
    if min_pixels:
        size["shortest_edge"] = min(min_pixels, size["longest_edge"])
    out = proc(images=[image], return_tensors="pt", size=size)
    grid = out["image_grid_thw"].long()
    return out["pixel_values"], grid, int(grid.prod(-1)[0]) // proc.merge_size ** 2


def _load_image(ref, allow_paths=True):
    """A PIL image, a path (only when allow_paths) or a base64 string / data URL -> RGB image. Over HTTP allow_paths
    is False: a client must not make the server open its own files, nor learn which paths exist."""
    from PIL import Image
    if ref is None:
        return None
    try:
        if isinstance(ref, Image.Image):
            return ref.convert("RGB")
        if isinstance(ref, (str, os.PathLike)):
            s = str(ref)
            path = s[len("file://"):] if s.startswith("file://") else s
            if allow_paths and not s.startswith("data:") and os.path.exists(path):
                return Image.open(path).convert("RGB")
            try:   # a data URL or bare base64
                data = base64.b64decode(s.split(",", 1)[-1] if s.startswith("data:") else s, validate=True)
            except ValueError:
                if not allow_paths:
                    raise RequestError("image: give a base64 string or a data URL") from None
                raise RequestError(f"image: no file at {path[:200]!r}, and it is not base64 either") from None
            return Image.open(io.BytesIO(data)).convert("RGB")
    except RequestError:
        raise
    except Exception as e:   # noqa: BLE001 -- any decode failure is the caller's bad image
        raise RequestError(f"image: could not read it ({type(e).__name__}: {e})") from e
    raise RequestError(f"image: give a PIL image, a path or a base64 string, not {type(ref).__name__}")


def _question(qid, q):
    where = f"question {qid!r}"
    if not isinstance(q, dict):
        raise RequestError(f"{where}: must be an object with a type and criteria")
    t = q.get("type")
    if t not in QTYPES:
        raise RequestError(f"{where}: type must be one of {', '.join(QTYPES)}, got {t!r}")
    _json_like(q.get("instructions"), f"{where}.instructions")
    c = q.get("criteria")
    if t == "noul":
        c = c or {}
        if not isinstance(c, dict) or set(c) - {"true", "false"}:
            raise RequestError(f"{where}: noul criteria are optional and may only have 'true' and 'false'")
        _json_like(c, f"{where}.criteria")
        opts, keys = [option_text("no", c.get("false")), option_text("yes", c.get("true"))], ["false", "true"]
    elif t == "choice":
        if not isinstance(c, dict) or not c:
            raise RequestError(f"{where}: choice criteria must be an object {{option key: description}}")
        _json_like(c, f"{where}.criteria")
        opts, keys = [option_text(k, v) for k, v in c.items()], list(c)
    else:
        if not isinstance(c, list) or not c:
            raise RequestError(f"{where}: score criteria must be a list of levels, lowest first")
        _json_like(c, f"{where}.criteria")
        opts, keys = [render(x) for x in c], [str(i) for i in range(len(c))]
    if len(opts) > MAX_OPTIONS:
        raise RequestError(f"{where}: {len(opts)} options; at most {MAX_OPTIONS}")
    return Question(qid, t, render(q.get("instructions")), opts, keys)


def _pixels(req, key, default):
    v = req.get(key)
    if v is None:
        return default
    if not isinstance(v, int) or isinstance(v, bool) or v <= 0:
        raise RequestError(f"{key}: must be a positive integer number of pixels, got {v!r}")
    return v


def parse(request, image=None, allow_paths=True):
    """dict -> Request, or RequestError naming the first problem found."""
    if not isinstance(request, dict):
        raise RequestError(f"a request must be an object with 'state' and 'questions', got {type(request).__name__}")
    unknown = set(request) - KEYS
    if unknown:
        raise RequestError(f"unknown request keys {sorted(unknown)}; allowed: {sorted(KEYS)}")
    if "state" not in request:
        raise RequestError("the request has no 'state'")
    _json_like(request["state"], "state")
    qs = request.get("questions")
    if not isinstance(qs, dict) or not qs:
        raise RequestError("'questions' must be a non-empty object {question id: question}")
    questions = [_question(qid, q) for qid, q in qs.items()]
    max_px = _pixels(request, "max_image_pixels", MAX_IMAGE_PIXELS)
    min_px = _pixels(request, "min_image_pixels", None)
    img = _load_image(image if image is not None else request.get("image"), allow_paths)
    return Request(render(request["state"]), questions, img, min_px, max_px, request)


# ---- reason, then decide --------------------------------------------------------------------------

REASONING_INSTRUCTION = ("Think it through step by step: work out anything that needs calculating or checking, then "
                         "say which option you choose for each question.")


def prompt(request):
    """A decision request -> one user message for the model to reason about: the state, then each question."""
    parts = [render(request["state"])]
    for qid, q in request["questions"].items():
        lines = [f"Question ({qid}): {render(q.get('instructions')) or qid}"]
        if q["type"] == "noul":
            lines.append("Answer yes or no.")
        elif q["type"] == "choice":
            lines += ["Options:"] + [f"- {k}: {render(v)}" if v not in (None, "") else f"- {k}"
                                     for k, v in q["criteria"].items()]
        else:
            lines += ["Levels, lowest first:"] + [f"{i}. {render(v)}" for i, v in enumerate(q["criteria"])]
        parts.append("\n".join(lines))
    return "\n\n".join(parts + [REASONING_INSTRUCTION])


def with_reasoning(request, text):
    """The request with the model's reasoning added to its state (as `reasoning`)."""
    state = request["state"]
    state = {**state, "reasoning": text} if isinstance(state, dict) else {"situation": state, "reasoning": text}
    return {**request, "state": state}
