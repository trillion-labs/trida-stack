#!/usr/bin/env python3
"""CPU-only tests for the decision path (decide_request, decide_engine helpers, serve_decide parsing): no GPU, no
SGLang, no model download. A tiny character-level tokenizer stands in for the real one.

    pytest -q inference/test_decide_cpu.py
"""
import pytest
import torch

from decide_engine import PointerHead, prefix_prompts, to_answer
from decide_request import SPECIAL, VISION, RequestError, parse, prompt, with_reasoning
from serve_decide import split_turn


class CharTok:
    """Special tokens get ids 1..8, every other character its code point + 100."""
    ids = {t: i + 1 for i, t in enumerate(SPECIAL + VISION)}

    def convert_tokens_to_ids(self, t):
        return self.ids[t]

    def __call__(self, text, add_special_tokens=False):
        return type("Out", (), {"input_ids": [ord(c) + 100 for c in text]})()


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
