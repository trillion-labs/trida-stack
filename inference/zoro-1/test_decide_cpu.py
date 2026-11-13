#!/usr/bin/env python3
"""CPU-only tests for the decision path (decide_request, decide_engine helpers, serve_decide parsing): no GPU, no
SGLang, no model download. A tiny character-level tokenizer stands in for the real one.

    pytest -q inference/zoro-1/test_decide_cpu.py
"""
import pytest
import torch

from decide_engine import PointerHead, prefix_prompts, to_answer
from decide_request import MAX_STATE, SPECIAL, VISION, RequestError, fit_reasoning, parse, prompt, with_reasoning
from serve_decide import TurnSplitter, split_turn


class CharTok:
    """Special tokens get ids 1..8, every other character its code point + 100."""
    ids = {t: i + 1 for i, t in enumerate(SPECIAL + VISION)}

    def convert_tokens_to_ids(self, t):
        return self.ids[t]

    def __call__(self, text, add_special_tokens=False):
        return type("Out", (), {"input_ids": [ord(c) + 100 for c in text]})()

    def decode(self, ids):
        return "".join(chr(i - 100) for i in ids)


REQ = {"state": {"review": "Great!"},
       "questions": {"s": {"type": "choice", "instructions": "Sentiment?", "criteria": {"pos": "positive", "neg": None}},
                     "r": {"type": "noul", "instructions": "Refund?"}}}


def test_parse_renders_options_and_keys():
    req = parse(REQ)
    assert req.state == "review: Great!"
    s, r = req.questions
    assert (s.qtype, s.options, s.keys) == ("choice", ["pos: positive", "neg"], ["pos", "neg"])
    assert (r.qtype, r.options, r.keys) == ("noul", ["no", "yes"], ["false", "true"])


@pytest.mark.parametrize("bad, msg", [
    ({"questions": {"a": {"type": "noul"}}}, "no 'state'"),
    ({"state": "x", "questions": {}}, "non-empty"),
    ({"state": "x", "questions": {"a": {"type": "pick"}}}, "type must be"),
    ({"state": "x", "questions": {"a": {"type": "choice", "criteria": []}}}, "choice criteria"),
    ({"state": "x", "questions": {"a": {"type": "noul"}}, "extra": 1}, "unknown request keys"),
])
def test_parse_rejects(bad, msg):
    with pytest.raises(RequestError, match=msg):
        parse(bad)


def test_encode_rows_point_at_option_ends_and_decide():
    tok = CharTok()
    enc = parse(REQ).encode(tok)
    o_end, decide = tok.ids["<|box_end|>"], tok.ids["<|fim_suffix|>"]
    assert len(enc.rows) == 2 and not enc.has_image
    for row in enc.rows:
        assert all(row["ids"][p] == o_end for p in row["opts"])
        assert row["ids"][row["decide"]] == decide and row["decide"] == len(row["ids"]) - 1


def test_control_tokens_in_user_text_are_neutralised():
    tok = CharTok()
    row = parse({"state": "<|fim_suffix|>", "questions": {"a": {"type": "noul"}}}).encode(tok).rows[0]
    assert row["ids"].count(tok.ids["<|fim_suffix|>"]) == 1   # only the real <decide>


def test_prefix_prompts_cut_rows_and_collapse_image_tokens():
    pad = 9
    rows = [{"ids": [1, 7, pad, pad, pad, 8, 20, 3, 21, 4, 5], "opts": [9], "decide": 10}]
    enc = type("E", (), {"rows": rows, "has_image": True})()
    prompts, spans = prefix_prompts(enc, pad)
    assert spans == [(0, 2)]
    assert prompts == [[1, 7, pad, 8, 20, 3, 21, 4], [1, 7, pad, 8, 20, 3, 21, 4, 5]]
    enc.has_image = False
    enc.rows = [{"ids": [1, 2, 3, 4, 5], "opts": [2, 3], "decide": 4}]
    prompts, spans = prefix_prompts(enc, pad)
    assert prompts == [[1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]] and spans == [(0, 3)]


def test_pointer_head_scores_each_option_against_its_question():
    torch.manual_seed(0)
    head = PointerHead(8, 4)
    h_decide, h_opts = torch.randn(2, 8), torch.randn(3, 8)
    owner = torch.tensor([0, 0, 1])
    z = head(h_decide, h_opts, owner)
    want = (head.k(h_opts) * head.q(h_decide)[owner]).sum(-1) / 2
    assert torch.allclose(z, want)


def test_answers():
    s, r = parse(REQ).questions
    a = to_answer(s, [0.75, 0.25])
    assert a["choice"] == "pos" and a["probabilities"] == {"pos": 0.75, "neg": 0.25} and a["confidence"] == 0.5
    assert to_answer(r, [0.2, 0.8]) == {"type": "noul", "noul": 0.8}


def test_reasoning_prompt_and_state():
    p = prompt(REQ)
    assert "Question (s): Sentiment?" in p and "- pos: positive" in p and "Answer yes or no." in p
    assert with_reasoning(REQ, "because")["state"] == {"review": "Great!", "reasoning": "because"}
    assert with_reasoning({**REQ, "state": "x"}, "y")["state"] == {"situation": "x", "reasoning": "y"}


def test_split_turn_parses_tool_calls():
    raw = ("<think>\nlook it up\n</think>\n\nChecking.\n<tool_call>\n<function=get_order>\n<parameter=order_id>\n"
           "#W1\n</parameter>\n<parameter=ids>\n[\"a\", \"b\"]\n</parameter>\n</function>\n</tool_call>")
    reasoning, text, calls = split_turn(raw)
    assert reasoning == "look it up" and text == "Checking."
    assert calls[0]["function"]["name"] == "get_order"
    assert calls[0]["function"]["arguments"] == '{"order_id": "#W1", "ids": ["a", "b"]}'


def test_split_turn_without_closed_thinking():
    assert split_turn("still thinking", thinking=True, truncated=True) == ("still thinking", "", [])
    assert split_turn("The answer is 42.", thinking=True) == ("", "The answer is 42.", [])
    assert split_turn("plain answer", thinking=False) == ("", "plain answer", [])


def _stream(raw, thinking=True, size=1, truncated=False, tools=None):
    """Feed a raw turn in `size`-character chunks; -> {field: text}, tool calls."""
    sp, got = TurnSplitter(thinking, tools), {}
    for i in range(0, len(raw), size):
        for f, t in sp.feed(raw[i:i + size]):
            got[f] = got.get(f, "") + t
    rest, calls = sp.finish(truncated)
    for f, t in rest:
        got[f] = got.get(f, "") + t
    return got, calls


@pytest.mark.parametrize("size", [1, 3, 1000])
def test_turn_splitter_matches_split_turn(size):
    raw = ("weigh it\n</think>\n\nChecking.\n<tool_call>\n<function=get_order>\n<parameter=order_id>\n#W1\n"
           "</parameter>\n</function>\n</tool_call>")
    got, calls = _stream(raw, size=size)
    reasoning, text, want = split_turn(raw)
    assert got["reasoning_content"].strip() == reasoning and got["content"].strip() == text
    assert [c["function"] for c in calls] == [c["function"] for c in want]


def test_turn_splitter_plain_and_no_thinking():
    got, calls = _stream("a < b </thi", size=2)   # a tag prefix that never completes is still text
    # thinking never closed and not cut off: streamed live as reasoning, then sent as the answer (as split_turn reads it)
    assert got == {"reasoning_content": "a < b </thi", "content": "a < b </thi"} and not calls
    got, calls = _stream("a < b </thi", size=2, truncated=True)
    assert got == {"reasoning_content": "a < b </thi"} and not calls
    got, calls = _stream("391", thinking=False)
    assert got == {"content": "391"} and not calls


TOOLS = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {
    "n": {"type": "integer"}, "ok": {"type": "boolean"}, "zip": {"type": "string"}, "note": {"type": ["string", "null"]}}}}}]


def _args(raw, tools=None):
    import json
    _, _, calls = split_turn(raw, thinking=False, tools=tools)
    return json.loads(calls[0]["function"]["arguments"])


def test_tool_call_values_are_typed_by_the_schema():
    raw = ("<tool_call>\n<function=f>\n<parameter=n>\n3\n</parameter>\n<parameter=ok>\ntrue\n</parameter>\n"
           "<parameter=zip>\n02134\n</parameter>\n<parameter=note>\n123\n</parameter>\n<parameter=x>\nnull\n"
           "</parameter>\n</function>\n</tool_call>")
    assert _args(raw, TOOLS) == {"n": 3, "ok": True, "zip": "02134", "note": "123", "x": None}
    # no schema: every value that is JSON is read as JSON, anything else stays text
    assert _args(raw) == {"n": 3, "ok": True, "zip": "02134", "note": 123, "x": None}
    sp = TurnSplitter(thinking=False, tools=TOOLS)
    sp.feed(raw)
    assert sp.finish()[1][0]["function"] == split_turn(raw, thinking=False, tools=TOOLS)[2][0]["function"]


def test_requests_never_name_files(tmp_path):
    pytest.importorskip("PIL")
    from PIL import Image
    from decide_request import open_image
    png = tmp_path / "a.png"
    Image.new("RGB", (8, 8), "red").save(png)
    # the parser never touches the filesystem: an existing image file is refused like a missing one, and the error
    # names no path, so it says nothing about which files exist
    for path in (str(png), "file://" + str(png), "/nonexistent/file.png", "/etc/hosts"):
        with pytest.raises(RequestError) as e:
            parse({**REQ, "image": path})
        assert str(tmp_path) not in str(e.value) and "/etc" not in str(e.value) and "nonexistent" not in str(e.value)
        with pytest.raises(RequestError):
            parse(REQ, image=path)
    # Python callers open the file themselves and pass the image
    assert parse(REQ, image=open_image(png)).image.size == (8, 8)
    assert open_image("file://" + str(png)).size == (8, 8)


def test_decide_endpoint_parses_without_paths():
    pytest.importorskip("httpx")
    testclient = pytest.importorskip("fastapi.testclient")
    from serve_decide import create_app

    class Engine:
        ckpt = "/ckpt/m"

        async def adecide(self, request, image=None, temperature=None, reasoning=False,
                          max_reasoning_tokens=768):
            parse(request, image)
            return {"answers": {}}

    pytest.importorskip("PIL")
    client = testclient.TestClient(create_app(Engine()))
    r = client.post("/v1/decide", json={"request": {**REQ, "image": "/etc/hosts"}})
    assert r.status_code == 400 and "/etc" not in r.text


def test_decide_rows_match_the_jev_encoder():
    # the server must score the rows decision checkpoints were trained on; inference/zoro-1/jev.py is pinned to the training
    # encoder by golden fixtures, so this ties decide_request's own encoder to it
    import jev
    from decide_request import MAX_STATE
    for req in (REQ, {**REQ, "state": "x" * 9000},
                {"state": ["a", {"b": 2}], "questions": {"m": {"type": "score", "instructions": "how much?",
                                                             "criteria": ["low", "mid", "high"]}}}):
        first = {"choice": lambda q: next(iter(q["criteria"])), "noul": lambda q: True, "score": lambda q: 0}
        labelled = {**req, "questions": {k: {**q, "label": first[q["type"]](q)} for k, q in req["questions"].items()}}
        want = jev.encode_rows(CharTok(), jev.materialize(labelled), **jev.training_context(MAX_STATE))
        got = parse(req).encode(CharTok()).rows
        assert got == [{k: r[k] for k in ("ids", "decide", "opts")} for r in want]


TURNS = [
    ("weigh it\n</think>\n\nChecking.\n<tool_call>\n<function=f>\n<parameter=n>\n3\n</parameter>\n</function>\n"
     "</tool_call>", True, False),
    ("The answer is 42.", True, False),                                    # thinking never closed
    ("still thinking", True, True),                                        # ... because max_tokens cut it
    ("ok\n</think>\n\nText <tool_call>\n<function=f>\nbroken", True, False),   # malformed tool call
    ("A <tool_call>\n<function=f>\n</function>\n</tool_call> B", False, False),  # text after a call
    ("r</think>\n\nhi", True, False),
    ("\n\n391", False, False),
]


@pytest.mark.parametrize("raw, thinking, truncated", TURNS)
@pytest.mark.parametrize("size", [1, 2, 3, 1000])
def test_stream_matches_split_turn(raw, thinking, truncated, size):
    """Streamed and non-streamed chat return the same turn. Only trailing whitespace may differ: a stream cannot know
    the text before a tag or the end was the last of its field."""
    reasoning, text, calls = split_turn(raw, thinking, truncated, TOOLS)
    got, got_calls = _stream(raw, thinking, size, truncated, TOOLS)
    assert got.get("content", "").rstrip() == text
    if reasoning:
        assert got["reasoning_content"].rstrip() == reasoning
    assert [c["function"] for c in got_calls] == [c["function"] for c in calls]


class ChatEngine:
    """Stands in for DecideEngine's chat: replays one raw turn, and records the sampling arguments it was given."""
    ckpt = "/ckpt/trida-decide"

    def __init__(self, raw, finish="stop"):
        self.raw, self.finish, self.kw = raw, finish, None

    async def achat(self, messages, **kw):
        self.kw = kw
        return {"text": self.raw, "prompt_tokens": 7, "completion_tokens": 5, "finish_reason": self.finish}

    async def achat_stream(self, messages, **kw):
        self.kw = kw
        for i in range(0, len(self.raw), 3):
            yield self.raw[i:i + 3]
        yield {"prompt_tokens": 7, "completion_tokens": 5, "finish_reason": self.finish}


def _client(engine):
    pytest.importorskip("httpx")
    testclient = pytest.importorskip("fastapi.testclient")
    from serve_decide import create_app
    return testclient.TestClient(create_app(engine))


MSG = {"messages": [{"role": "user", "content": "hi"}]}


def _events(r):
    import json
    lines = [x[len("data: "):] for x in r.text.split("\n") if x.startswith("data: ")]
    assert lines[-1] == "[DONE]"
    return [json.loads(x) for x in lines[:-1]]


def test_chat_response_shape_and_usage():
    raw = "look\n</think>\n\n<tool_call>\n<function=f>\n<parameter=n>\n3\n</parameter>\n</function>\n</tool_call>"
    r = _client(ChatEngine(raw)).post("/v1/chat/completions", json={**MSG, "tools": TOOLS})
    assert r.status_code == 200
    choice = r.json()["choices"][0]
    assert choice["finish_reason"] == "tool_calls"
    assert choice["message"]["content"] is None and choice["message"]["reasoning_content"] == "look"
    assert choice["message"]["tool_calls"][0]["function"] == {"name": "f", "arguments": '{"n": 3}'}
    assert r.json()["usage"] == {"prompt_tokens": 7, "completion_tokens": 5, "total_tokens": 12}
    r = _client(ChatEngine("cut off", "length")).post("/v1/chat/completions", json=MSG)
    assert r.json()["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize("raw, finish", [("look\n</think>\n\nIt is 3.", "stop"), ("The answer is 42.", "stop"),
                                         ("still thinking", "length"), (TURNS[0][0], "stop"), (TURNS[3][0], "stop")])
def test_chat_stream_matches_non_stream(raw, finish):
    client = _client(ChatEngine(raw, finish))
    whole = client.post("/v1/chat/completions", json={**MSG, "tools": TOOLS}).json()["choices"][0]
    events = _events(client.post("/v1/chat/completions", json={**MSG, "tools": TOOLS, "stream": True}))
    deltas = [e["choices"][0]["delta"] for e in events]
    assert deltas[0] == {"role": "assistant"}
    last = events[-1]
    assert last["choices"][0]["finish_reason"] == whole["finish_reason"] and last["usage"]["total_tokens"] == 12
    order = [k for d in deltas[1:-1] for k in d]                 # reasoning, then content, then tool calls
    assert order == sorted(order, key=["reasoning_content", "content", "tool_calls"].index)
    text = "".join(d.get("content", "") for d in deltas).rstrip()
    assert text == (whole["message"]["content"] or "")
    calls = [c["function"] for d in deltas for c in d.get("tool_calls", [])]
    assert calls == [c["function"] for c in whole["message"].get("tool_calls", [])]


def test_chat_sampling_values():
    engine = ChatEngine("x")
    client = _client(engine)
    body = {**MSG, "temperature": None, "top_p": None, "max_tokens": None, "top_k": None, "presence_penalty": None}
    assert client.post("/v1/chat/completions", json=body).status_code == 200      # null means "use the default"
    assert engine.kw["temperature"] == 0.6 and engine.kw["max_new_tokens"] == 2048 and engine.kw["top_k"] == 20
    assert client.post("/v1/chat/completions", json={**MSG, "temperature": 0, "top_k": 5}).status_code == 200
    assert engine.kw["temperature"] == 0.0 and engine.kw["top_k"] == 5            # 0 is greedy, not "unset"
    for bad in ({"temperature": "hot"}, {"top_k": 2.5}, {"max_tokens": True}):
        r = client.post("/v1/chat/completions", json={**MSG, **bad})
        assert r.status_code == 400 and next(iter(bad)) in r.text
    assert client.post("/v1/chat/completions", json={"messages": []}).status_code == 400


def test_mlx_copy_of_the_request_contract_is_identical():
    # inference/zoro-1/mlx ships its own copy (it installs without this repo's torch stack); the MLX suite also pins
    # it, but CI runs only this job, so an edit to one copy alone fails here.
    from pathlib import Path
    here = Path(__file__).resolve().parent
    assert (here / "mlx/trida_mlx/decide_request.py").read_bytes() == (here / "decide_request.py").read_bytes(), \
        "copy inference/zoro-1/decide_request.py to inference/zoro-1/mlx/trida_mlx/decide_request.py"

def test_decide_forwards_reasoning_budget():
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    class Engine:
        ckpt = "zoro-1"

        async def adecide(self, request, image, temperature, reasoning, **kwargs):
            self.received = (request, reasoning, kwargs)
            return {"answers": {}}

    engine = Engine()
    client = TestClient(create_app(engine))
    response = client.post("/v1/decide", json={"request": REQ, "reasoning": True,
                                             "max_reasoning_tokens": 2048})
    assert response.status_code == 200
    assert engine.received == (REQ, True, {"max_reasoning_tokens": 2048})
    for budget in [1.5, True]:
        assert client.post("/v1/decide", json={"request": REQ,
                           "max_reasoning_tokens": budget}).status_code == 400


def _reasoning_engine(finish="stop", context_length=16384, text="Check it."):
    from decide_engine import DecideEngine

    class Tok(CharTok):
        def apply_chat_template(self, *args, **kwargs):
            return "prompt"

    class Backend:
        calls = 0

        async def async_generate(self, **kwargs):
            self.calls += 1

            async def chunks():
                yield {"text": text, "meta_info": {}}
                yield {"text": text + "</think>" if finish == "stop" else text,
                       "meta_info": {"finish_reason": {"type": finish}, "completion_tokens": 3}}
            return chunks()

    engine = object.__new__(DecideEngine)
    engine.ckpt, engine.tok, engine.proc = "zoro-1", Tok(), None
    engine.context_length, engine.reasoning_temperature, engine._stops = context_length, 0.6, []
    engine.decision_max_state, engine.decision_max_row, engine.reasoning_token_limit = 7552, 8192, 4096
    engine.text_T, engine.image_T, engine.engine = 1.0, 1.0, Backend()

    async def logits(enc, image):
        engine.scored = enc
        return [torch.tensor([0.0, 1.0]) for row in enc.rows]

    engine.alogits = logits
    return engine


@pytest.mark.parametrize("finish", ["stop", "length"])
def test_decide_exposes_real_reasoning_finish_status(finish):
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    engine = _reasoning_engine(finish)
    response = TestClient(create_app(engine)).post("/v1/decide", json={
        "request": REQ, "reasoning": True, "max_reasoning_tokens": 100})
    assert response.status_code == 200
    body = response.json()
    assert body["reasoning"] == "Check it."
    assert body["reasoning_stats"]["finish_reason"] == finish
    if finish == "length":
        assert body["cut"] == "length" and body["answers_source"] == "direct"
        assert engine.scored.rows == parse(REQ).encode(engine.tok).rows
    else:
        assert "cut" not in body and body["answers_source"] == "reasoned"
    assert set(body["answers"]) == set(REQ["questions"])


@pytest.mark.parametrize("image_tokens", [0, 400])
def test_fit_reasoning_preserves_conclusion_without_encoder_truncation(image_tokens):
    tok = CharTok()
    request = {**REQ, "state": "x" * 7000}
    text = "step " * 300 + "conclusion: yes"
    fitted = fit_reasoning(request, text, tok, image_tokens)
    assert fitted.startswith("…") and fitted.endswith("conclusion: yes")
    reasoned = parse(with_reasoning(request, fitted))
    room = MAX_STATE - 1 - (image_tokens + 2 if image_tokens else 0)
    assert len(tok(reasoned.state).input_ids) <= room
    # Inspect what actually reaches the decision rows, not just the fitted string.
    row = reasoned.encode_rows(tok, image_tokens)[0]["ids"]
    q = row.index(tok.ids[SPECIAL[1]])
    assert tok.decode(row[q - len("conclusion: yes"):q]) == "conclusion: yes"


@pytest.mark.parametrize("state_size,expected_cut", [(MAX_STATE - 1, "state_limit"), (7000, None)])
def test_cuda_reports_reasoning_trim_on_invoice_length_states(state_size, expected_cut):
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    text = "step " * 300 + "conclusion: yes"
    engine = _reasoning_engine(text=text)
    request = {**REQ, "state": "x" * state_size}
    body = TestClient(create_app(engine)).post("/v1/decide", json={
        "request": request, "reasoning": True, "max_reasoning_tokens": 2048}).json()
    assert body["reasoning_stats"]["finish_reason"] == "stop"
    assert body["reasoning_trimmed"] is True
    assert body["reasoning"] == text
    if expected_cut:
        assert body["cut"] == expected_cut and body["answers_source"] == "direct"
        assert engine.scored.rows == parse(request).encode(engine.tok).rows
    else:
        assert "cut" not in body and body["answers_source"] == "reasoned"
        row = engine.scored.rows[0]["ids"]
        q = row.index(engine.tok.ids[SPECIAL[1]])
        assert engine.tok.decode(row[q - len("conclusion: yes"):q]) == "conclusion: yes"


def test_cuda_empty_reasoning_returns_explicit_direct_fallback():
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    body = TestClient(create_app(_reasoning_engine(text="  "))).post("/v1/decide", json={
        "request": REQ, "reasoning": True}).json()
    assert body["cut"] == "empty" and body["answers_source"] == "direct"
    assert body["answers"]


@pytest.mark.parametrize("budget", [0, -1, 4097, 64000])
def test_decide_rejects_excessive_reasoning_before_generation(budget):
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    engine = _reasoning_engine()
    response = TestClient(create_app(engine)).post("/v1/decide", json={
        "request": REQ, "reasoning": True, "max_reasoning_tokens": budget})
    assert response.status_code == 400 and engine.engine.calls == 0


def test_reasoning_budget_reserves_prompt_space():
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    engine = _reasoning_engine(context_length=10)
    client = TestClient(create_app(engine))
    assert client.post("/v1/decide", json={"request": REQ, "reasoning": True,
                                        "max_reasoning_tokens": 5}).status_code == 400
    assert engine.engine.calls == 0
    assert client.post("/v1/decide", json={"request": REQ, "reasoning": True,
                                        "max_reasoning_tokens": 4}).status_code == 200


def test_extended_limits_keep_full_state_and_reasoning_with_strict_metadata():
    from fastapi.testclient import TestClient
    from serve_decide import create_app

    text = "because " * 1000
    engine = _reasoning_engine(context_length=32768, text=text)
    engine.decision_max_state, engine.decision_max_row = 15744, 16384
    engine.reasoning_token_limit = 8192
    request = {**REQ, "state": "x" * 4438}
    client = TestClient(create_app(engine))
    response = client.post("/v1/decide", json={"request": request, "reasoning": True,
                                             "max_reasoning_tokens": 8192})
    assert response.status_code == 200
    body = response.json()
    assert body["answers_source"] == "reasoned" and not body["reasoning_trimmed"]
    assert "cut" not in body and body["reasoning"] == text.strip()
    expected = parse(with_reasoning(request, text.strip())).state
    row = engine.scored.rows[0]["ids"]
    q = row.index(engine.tok.ids[SPECIAL[1]])
    assert engine.tok.decode(row[1:q]) == expected
    assert len(row) <= 16384
    assert client.post("/v1/decide", json={"request": request, "reasoning": True,
                                         "max_reasoning_tokens": 8193}).status_code == 400


@pytest.mark.parametrize("state,row,context,budget", [
    (16384, 8192, 32768, 8192), (16384, 16384, 8192, 8192),
    (0, 8192, 32768, 8192), (7552, 8192, 32768, 8193),
    (True, 8192, 32768, 8192), (16384, 16384, 32768, 8192),
])
def test_invalid_server_limits_are_rejected_before_loading_model(state, row, context, budget):
    from decide_request import validate_decision_limits
    with pytest.raises(RequestError):
        validate_decision_limits(state, row, context, budget)


def test_extended_encoder_still_rejects_rows_over_its_limit():
    request = {"state": "x" * 12000, "questions": {
        "q": {"type": "noul", "instructions": "q" * 5000}}}
    with pytest.raises(RequestError, match="16384 per question"):
        parse(request).encode(CharTok(), max_state=16384, max_branch=16384)


@pytest.mark.parametrize("image_tokens", [0, 32])
def test_reasoning_fit_reserves_longest_question_row(image_tokens):
    tok = CharTok()
    request = {"state": "x" * 3000, "questions": {
        "short": {"type": "noul"},
        "long": {"type": "noul", "instructions": "q" * 1200}}}
    parse(request).encode_rows(tok, image_tokens)
    text = "because " * 2000 + "therefore yes"
    fitted = fit_reasoning(request, text, tok, image_tokens)
    assert fitted and fitted != text and fitted.endswith("therefore yes")
    rows = parse(with_reasoning(request, fitted)).encode_rows(tok, image_tokens)
    assert all(len(row["ids"]) <= 8192 for row in rows)


def test_long_question_reasoning_returns_trimmed_answer_over_http():
    from fastapi.testclient import TestClient
    from serve_decide import create_app
    engine = _reasoning_engine(text="because " * 2000 + "therefore yes")
    request = {"state": "x" * 3000, "questions": {
        "q": {"type": "noul", "instructions": "q" * 1200}}}
    response = TestClient(create_app(engine)).post("/v1/decide", json={
        "request": request, "reasoning": True})
    assert response.status_code == 200
    body = response.json()
    assert body["reasoning_trimmed"] and body["answers_source"] == "reasoned"
    assert all(len(row["ids"]) <= 8192 for row in engine.scored.rows)
