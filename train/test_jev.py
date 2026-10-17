"""CPU tests for Jev decision training data (train/jev.py). No GPU, no network. Parity with kev itself (encoding and
forward, on real kev data and a tiny model) is checked by train/tools/jev_parity_{kev,trida}.py."""
import math
import random

import torch
import torch.nn.functional as F

from train.jev import (
    MIN_PACK,
    SPECIAL,
    JevRowDataset,
    PointerHead,
    augment,
    encode_rows,
    jev_loss,
    materialize,
    pack_jev,
    training_context,
)


class CharTok:
    """Stand-in tokenizer: one id per character, the delimiter tokens get their own ids."""

    def convert_tokens_to_ids(self, t):
        return 1000 + SPECIAL.index(t)

    def __call__(self, text, add_special_tokens=False):
        return type("Enc", (), {"input_ids": [ord(c) % 997 for c in text]})()


REQ = {"state": "refund please", "questions": {
    "team": {"type": "choice", "instructions": "which team", "criteria": {"billing": "money", "tech": None, "sales": "buy"},
             "label": "billing"},
    "urgent": {"type": "noul", "instructions": "urgent?", "label": True},
    "mood": {"type": "score", "instructions": "mood", "criteria": ["calm", "angry"], "label": 1}}}


def test_encode_rows_layout():
    tok = CharTok()
    rows = encode_rows(tok, materialize(REQ), **training_context(384))
    assert len(rows) == 3
    for row, q in zip(rows, materialize(REQ)["questions"]):
        assert row["ids"][0] == 1000                                   # <state>
        assert row["ids"][row["decide"]] == 1004 and row["decide"] == len(row["ids"]) - 1   # <decide> ends the row
        assert all(row["ids"][o] == 1003 for o in row["opts"])         # every option index is its </opt>
        assert len(row["opts"]) == len(q["options"]) and row["label"] == q["label"]
    assert rows[0]["label"] == 0 and rows[1]["label"] == 1 and rows[2]["label"] == 1   # billing / true / level 1


def test_pack_and_loss_match_manual():
    tok = CharTok()
    v1 = encode_rows(tok, materialize(REQ), **training_context(384))
    v2 = v1[:1]
    b = pack_jev([v1, v2], tok)
    assert b["input_ids"].shape[1] == sum(len(r["ids"]) for r in v1 + v2)
    assert b["seg_id"][0].unique().tolist() == [0, 1, 2, 3]           # one document per question row
    z = torch.randn(len(b["opts"]))
    loss, acc = jev_loss(z, b)
    qs = b["questions"]
    ce = [-F.log_softmax(z[q["start"]:q["start"] + q["k"]], -1)[q["label"]] for q in qs]
    want = (torch.stack(ce[:3]).mean() + ce[3]) / 2                   # mean over variants of mean over questions
    assert torch.allclose(loss, want)
    assert 0 <= acc <= 1


def test_soft_target_loss():
    tok = CharTok()
    req = {"state": "?", "questions": {"q": {"type": "choice", "instructions": "x", "criteria": {"a": None, "b": None},
                                             "label": "a", "target": {"a": 0.5, "b": 0.5}}}}
    b = pack_jev([encode_rows(tok, materialize(req), **training_context(384))], tok)
    loss, _ = jev_loss(torch.zeros(2), b)
    assert math.isclose(loss.item(), math.log(2), rel_tol=1e-6)


def test_augment_keeps_label_option():
    rng = random.Random(0)
    for _ in range(200):
        out = augment(REQ, rng)["questions"]["team"]
        assert out["label"] in out["criteria"]


def test_dataset_shards_cover_records_once_and_respect_token_cap():
    tok = CharTok()
    records = [dict(REQ, state=f"record {i}") for i in range(40)]
    for rank in range(4):
        it = iter(JevRowDataset(records, tok, rank=rank, world=4, seed=1, max_tokens=400, p_none_pair=0.0))
        assert all(next(it)["input_ids"].shape[1] <= 400 for _ in range(4))
    ds = JevRowDataset(records, tok, rank=0, world=1, seed=1, max_tokens=100000, p_none_pair=0.0)
    first_epoch = next(iter(ds))
    assert len({q["variant"] for q in first_epoch["questions"]}) == 40   # one batch holds every record once


def test_pointer_head_shapes():
    head = PointerHead(16, dp=8)
    z = head(torch.randn(3, 16), torch.randn(7, 16), torch.tensor([0, 0, 1, 1, 1, 2, 2]))
    assert z.shape == (7,)


def test_oversize_variant_gets_its_own_micro_batch():
    tok = CharTok()
    rec = dict(REQ, state="x" * 600)                                  # ~700 row tokens, over the 400 budget
    it = iter(JevRowDataset([rec], tok, max_tokens=400, max_variant_tokens=2000, p_none_pair=0.0))
    assert next(it)["input_ids"].shape[1] > 400
    ds = JevRowDataset([rec, REQ], tok, max_tokens=400, max_variant_tokens=500, p_none_pair=0.0)
    assert all(next(iter(ds))["input_ids"].shape[1] <= 400 for _ in range(2)) and ds.skipped >= 1


def test_short_pack_is_padded_with_a_dummy_row():
    tok = CharTok()
    req = {"state": "?", "questions": {"q": {"type": "noul", "instructions": "x", "label": True}}}
    rows = encode_rows(tok, materialize(req), **training_context(384))
    b = pack_jev([rows], tok)
    n = len(rows[0]["ids"])
    assert n < MIN_PACK and b["input_ids"].shape[1] == MIN_PACK
    assert b["seg_id"][0, n:].unique().tolist() == [1] and len(b["questions"]) == 1
