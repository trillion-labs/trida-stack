"""Self-routing (``trida_mode: "auto"``). PROOF OF CONCEPT: shows how one checkpoint's three heads (AR,
block diffusion, and the Jev decision head) can work together in a single engine for faster inference. It is
not a tuned or production router.

Before decoding, one typed decision (``noul``) asks the model whether the reply needs careful generation.
P(yes) >= threshold -> ``causal`` (exact AR); otherwise ``diffusion`` (parallel block diffusion: fast, not
lossless). The policy is a single threshold so a client can trade quality for speed per request.

The decision runs through ``Decider`` on its own state cache: the state is a Jev-format row, not the chat
prompt, so a routed request prefills its context twice (once as the decision state, once to generate); the
Decider's prefix cache keeps the first of those incremental across turns (see ``route_state``). Sharing the
prefill outright needs the routing question trained as a suffix of the chat prompt.
"""
from __future__ import annotations

import sys
import time
import traceback

from .decide_request import MAX_STATE, render

ROUTE_QID = "careful"
ROUTE_QUESTION = {
    "type": "noul",
    "instructions": (
        "Would a fast, rough draft of the assistant's next reply likely be wrong or sloppy? Yes if the reply needs "
        "careful step-by-step reasoning, exact arithmetic, code, structured output such as a tool call or JSON, or "
        "precise facts. No for short, simple or casual replies."
    ),
    "criteria": {"true": "needs careful generation", "false": "a fast draft is fine"},
}
ROUTE_THRESHOLD = 0.5
CONVERSATION = "conversation"
STATE_BUDGET = MAX_STATE - 512  # tokens for the rendered state (the decision row adds its own delimiters)


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict) and p.get("type") == "text":
                parts.append(str(p.get("text", "")))
            elif isinstance(p, dict) and p.get("type") in ("image_url", "image"):
                parts.append("[image]")
        return " ".join(parts)
    return "" if content is None else str(content)


def _function_name(obj):
    """``obj["function"]["name"]`` of a tool or tool call, or None when any level is missing or not an object."""
    f = obj.get("function") if isinstance(obj, dict) else None
    return f.get("name") if isinstance(f, dict) else None


def _turn(m) -> str:
    text = _text(m.get("content"))
    calls = [n for n in (_function_name(c) for c in m.get("tool_calls") or []) if n]
    if calls:
        text = (text + " " if text else "") + "[calls " + ", ".join(calls) + "]"
    return f"{m.get('role', 'user')}: {text}"


def route_state(messages=None, prompt=None, tools=None) -> dict:
    """The decision state for a chat (``messages``, ``tools``) or completion (``prompt``) request. Tools come first
    and the conversation oldest turn first, so the state of a later turn starts with the state of an earlier one
    and the Decider's prefix cache prefills only what is new. ``fit_state`` keeps it within the decision context."""
    if prompt is not None:
        return {"request": _text(prompt)}
    state = {}
    names = [n for n in (_function_name(t) for t in tools or []) if n]
    if names:
        state["tools available"] = ", ".join(names)
    state[CONVERSATION] = [_turn(m) for m in messages or [] if isinstance(m, dict)]
    return state


def fit_state(state: dict, count_tokens, budget: int = STATE_BUDGET) -> dict:
    """Drop the oldest turns until the rendered state fits ``budget`` tokens (left alone, the decision row would
    cut its end: the newest turns). Turns are dropped up to a cut point at a multiple of budget/4 tokens from the
    start of the conversation, so the first kept turn stays the same while the conversation grows by up to a
    quarter of the budget, and the prefix cache keeps hitting between cuts."""
    turns = state.get(CONVERSATION) or []
    total = count_tokens(render(state))
    if total <= budget or len(turns) < 2:
        return state
    offsets = [0]  # where each turn starts, in tokens from the start of the conversation (estimated per turn)
    for t in turns:
        offsets.append(offsets[-1] + count_tokens(render([t])))
    step = max(1, budget // 4)
    start, q = 0, 1
    while start < len(turns) - 1 and total > budget:
        start = next((i for i in range(start, len(turns)) if offsets[i] >= q * step), len(turns) - 1)
        total = count_tokens(render({**state, CONVERSATION: turns[start:]}))  # exact, not the estimate
        q += 1
    return {**state, CONVERSATION: turns[start:]}


def route(decider, state: dict, threshold: float = ROUTE_THRESHOLD) -> dict:
    """Ask the routing question about ``state``; returns the route record reported with the response:
    ``{"mode", "p_careful", "threshold", "ms"}`` (+ ``"fallback"``: why it decoded causally without deciding).
    Any failure to decide falls back to ``causal``: the request is still served, exactly. Worker thread only."""
    t = time.perf_counter()
    try:
        tok = decider.engine.tokenizer
        state = fit_state(state, lambda s: len(tok.encode(s, add_special_tokens=False)))
        res = decider.decide({"state": state, "questions": {ROUTE_QID: ROUTE_QUESTION}}, allow_paths=False)
        p = float(res["answers"][ROUTE_QID]["noul"])
    except Exception as e:  # noqa: BLE001 - routing is an optimization; never fail the request over it
        traceback.print_exc(file=sys.stderr)
        return {"mode": "causal", "p_careful": None, "threshold": threshold,
                "fallback": f"{type(e).__name__}: {e}", "ms": round((time.perf_counter() - t) * 1000, 1)}
    return {"mode": "causal" if p >= threshold else "diffusion", "p_careful": p, "threshold": threshold,
            "ms": round((time.perf_counter() - t) * 1000, 1)}
