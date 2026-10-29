# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
"""Jev request encoding: the token rows a Trida checkpoint's typed-decision (Jev) head reads.

A Jev request is a ``state`` plus typed questions (``noul`` yes/no, ``choice``, ``score``); the model returns one
probability per option. Each question is a causal row ``<state> … <q> instructions <opt> o1 </opt> … <decide>``
(reused Qwen special tokens as delimiters). The pointer head (``inference/zoro-1/jev_head.py``) scores every ``</opt>`` hidden
state against the ``<decide>`` hidden state.

Derived from kev (https://github.com/jaredpalmer/kev, Copyright Jared Palmer and the kev contributors, Apache-2.0):
the request mapping is kev's ``kev/api.py`` / ``kev.data.materialize`` and the encoder is kev's ``kev/model.py``
encode. Modified: a state may open with an image (``image_tokens``), and the training stack's tokenizer handling. The code is
the training stack's encoder unchanged, so rows built here are byte-identical to the rows that encoder builds;
``inference/zoro-1/test_jev_golden.py`` checks that against golden fixtures.

Standard library only (no torch), so a serving process or demo can import it without training dependencies.
"""
import re

# ---- request mapping (kev/api.py) ---------------------------------------------------------------

def render(v, indent=0):
    """Flatten str | object | array into text the model sees. Field names are kept as labels."""
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


def question_keys(qtype, criteria):
    """Keys a question's probabilities are reported under, in option order."""
    if qtype == "choice":
        return list(criteria)
    if qtype == "noul":
        return ["false", "true"]
    return [str(i) for i in range(len(criteria))]


def materialize(req):
    """Labelled request -> internal record {"state", "questions": [{instr, options, label, qtype, keys, target?}]}
    (kev.data.materialize over kev.api.to_record)."""
    qs = []
    for qid, q in req["questions"].items():
        keys = question_keys(q["type"], q.get("criteria"))
        if q["type"] == "noul":
            c = q.get("criteria") or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
        elif q["type"] == "choice":
            opts = [option_text(k, v) for k, v in q["criteria"].items()]
        else:
            opts = [render(x) for x in q["criteria"]]
        y = q["label"]
        rq = {"instr": render(q.get("instructions")), "options": opts, "qid": qid, "qtype": q["type"], "keys": keys,
              "label": keys.index(y) if q["type"] == "choice" else int(y)}
        if q.get("target") is not None:
            t = [float(q["target"].get(k, 0.0)) for k in keys]
            if sum(t) <= 0:
                raise ValueError(f"target for {qid} puts no mass on any option")
            rq["target"] = [x / sum(t) for x in t]
        qs.append(rq)
    return {"state": render(req["state"]), "questions": qs}


# ---- encoder (kev/model.py) -----------------------------------------------------------------------

SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]  # state q opt /opt decide
VISION = ["<|vision_start|>", "<|image_pad|>", "<|vision_end|>"]   # an image in the state (Qwen3.5 VL tokens)
MAX_STATE, MAX_BRANCH = 384, 1024
_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


class ContextOverflow(ValueError):
    pass


def training_context(max_state=MAX_STATE):
    """Encoder limits with the state limit lifted to max_state; the row limit grows by the same amount."""
    if max_state < MAX_STATE:
        raise ValueError(f"max_state must be >= {MAX_STATE}")
    return {"max_state": max_state, "max_branch": MAX_BRANCH + max_state - MAX_STATE}


def user_tokens(tok, text):
    """Caller text can never produce delimiter/control tokens: `<|name|>` is rewritten to `<¦name¦>` first."""
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", text), add_special_tokens=False).input_ids


def encode_rows(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH, image_tokens=0):
    """One record -> its question rows, each the causal row [<state> state] + [<q> instr <opt>o</opt>... <decide>]
    (kev.model.encode + rows_of, non-isolated options). Row k: {"ids", "decide": index of <decide>, "opts": index of
    each </opt>, "label", "target"?}. The state is cut to its first max_state tokens (kev's non-strict encode); a
    branch over its row budget raises ContextOverflow.
    image_tokens > 0 (a record with an image; not in kev): the state opens with the image,
    <state> <|vision_start|> <|image_pad|> x image_tokens <|vision_end|> text, and the text is cut so the whole
    state stays within max_state (the image counts as image_tokens + 2)."""
    s_id, q_id, o_id, c_id, d_id = (tok.convert_tokens_to_ids(t) for t in SPECIAL)
    image = []
    if image_tokens:
        vs, pad, ve = (tok.convert_tokens_to_ids(t) for t in VISION)
        image = [vs] + [pad] * image_tokens + [ve]
    state = [s_id] + image + user_tokens(tok, rec["state"])[: max(0, max_state - 1 - len(image))]
    rows = []
    for q in rec["questions"]:
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        branch = instr + [t for sp in spans for t in sp] + [d_id]
        if len(branch) > max_branch - len(state):
            raise ContextOverflow(f"branch too long: {len(branch)} tokens with a {len(state)}-token state")
        ends, cursor = [], len(state) + len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)
        row = {"ids": state + branch, "decide": len(state) + len(branch) - 1, "opts": ends, "label": q["label"],
               "qtype": q["qtype"]}
        if q.get("target") is not None:
            row["target"] = q["target"]
        rows.append(row)
    return rows
