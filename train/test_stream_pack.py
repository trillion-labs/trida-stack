"""CPU tests for packed multi-turn rows (buffered first-fit, --pack --multiturn). No GPU, no network."""
import random
from types import SimpleNamespace

import torch

from train.data.text_sft_data import (
    StreamPackedMultiTurnCollator,
    _block_align_doc,
    _stream_first_fit,
    greedy_pack,
)

BD = 8
PAD = 0


def make_doc(n, tag):
    """A built multi-turn doc of n tokens: token ids carry the tag, the second half is supervised."""
    ids = torch.full((n,), tag, dtype=torch.long)
    labels = ids.clone()
    labels[: n // 2] = -100
    turn = torch.where(labels != -100, 0, -1)
    return _block_align_doc({"input_ids": ids, "labels": labels, "resp_block": turn.clone(),
                             "turn_id": turn}, BD, PAD)


def skewed_docs(count, max_length, seed=0):
    rng = random.Random(seed)
    sizes = [rng.choice([rng.randint(200, 3000), rng.randint(8000, max_length)]) for _ in range(count)]
    return [make_doc(n, tag) for tag, n in enumerate(sizes, start=1)]


def test_every_doc_packed_exactly_once_and_rows_fit():
    max_length = 32768
    docs = skewed_docs(500, max_length)
    rows = list(_stream_first_fit(iter(docs), max_length, buffer_size=32))
    tags = [int(d["input_ids"][0]) for row in rows for d in row]
    assert sorted(tags) == list(range(1, len(docs) + 1))
    assert all(sum(d["input_ids"].shape[0] for d in row) <= max_length for row in rows)


def test_rows_start_with_oldest_buffered_doc():
    # Row heads come out in stream order, so a doc that never fits next to others can't starve.
    docs = skewed_docs(300, 16384, seed=1)
    heads = [int(row[0]["input_ids"][0]) for row in _stream_first_fit(iter(docs), 16384, buffer_size=16)]
    assert heads == sorted(heads)


def test_buffer_bounded_by_buffer_size():
    pulled = []

    def stream():
        for d in skewed_docs(200, 16384, seed=2):
            pulled.append(d)
            yield d

    emitted = 0
    for row in _stream_first_fit(stream(), 16384, buffer_size=8):
        emitted += len(row)
        assert len(pulled) - emitted <= 8


def test_max_segments_respected():
    docs = [make_doc(100, t) for t in range(1, 101)]
    rows = list(_stream_first_fit(iter(docs), 32768, buffer_size=32, max_segments=5))
    assert max(len(r) for r in rows) <= 5
    assert sum(len(r) for r in rows) == 100


def test_fills_rows_better_than_drop_overflow_packing():
    # The previous collator path: greedy_pack 11 docs at a time, keep only the first row.
    max_length = 65536
    docs = skewed_docs(660, max_length, seed=3)
    old_kept = sum(len(greedy_pack(docs[i:i + 11], max_length, ar_only=True)[0])
                   for i in range(0, len(docs), 11))
    new_rows = list(_stream_first_fit(iter(docs), max_length, buffer_size=32))
    assert sum(len(r) for r in new_rows) == len(docs) > old_kept
    fill = sum(d["input_ids"].shape[0] for r in new_rows for d in r) / (len(new_rows) * max_length)
    assert fill > 0.8


def test_collator_rows_are_block_aligned_and_doc_isolated():
    tok = SimpleNamespace(pad_token_id=PAD, eos_token_id=PAD)
    collate = StreamPackedMultiTurnCollator(tok, bd_size=BD, max_length=4096,
                                            length_buckets=(1024, 2048, 4096))
    docs = [make_doc(n, t) for t, n in enumerate([301, 77, 1000, 13, 512], start=1)]
    rows = list(_stream_first_fit(iter(docs), 4096, buffer_size=4))
    batch = collate(rows[:1])
    assert batch["input_ids"].shape[1] % BD == 0
    seg = batch["seg_id"][0]
    starts = [0] + [i for i in range(1, seg.shape[0]) if seg[i] != seg[i - 1] and seg[i] >= 0]
    assert all(s % BD == 0 for s in starts)
    for k, d in enumerate(rows[0]):
        assert torch.equal(batch["input_ids"][0][seg == k], d["input_ids"])
    assert (batch["labels"][0][batch["attention_mask"][0] == 0] == -100).all()
