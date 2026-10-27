"""CPU tests for image+text (VL) training data: 3-D RoPE positions, image-pad expansion, packing.
No GPU, no network. Tests marked with TRIDA_TEST_TOKENIZER (a local Qwen3.5 checkpoint dir) also
exercise the real chat template; they skip when it isn't set."""
import os
import random

import pytest
import torch

from train.data.text_sft_data import (
    StreamPackedMultiTurnCollator,
    _block_align_doc,
    _expand_image_pads,
    _image_refs,
    _stream_first_fit,
    build_sample_multiturn,
    mrope_position_ids,
)

IMG = 248056
BD = 8


def random_layout(rng, n_images):
    """Token ids with text runs around image runs + the (t, h, w) patch grid of each image."""
    ids, grids = [], []
    for _ in range(n_images):
        ids += [rng.randint(0, 1000) for _ in range(rng.randint(1, 12))]
        gh, gw = 2 * rng.randint(1, 6), 2 * rng.randint(1, 6)
        grids.append([1, gh, gw])
        ids += [IMG] * (gh * gw // 4)
    ids += [rng.randint(0, 1000) for _ in range(rng.randint(1, 12))]
    return torch.tensor(ids), torch.tensor(grids, dtype=torch.long).reshape(-1, 3)


def test_mrope_matches_hf_get_rope_index():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    cfg = Qwen3_5Config(
        text_config=dict(hidden_size=64, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                         num_key_value_heads=1, head_dim=32, vocab_size=248320,
                         layer_types=["linear_attention", "full_attention"]),
        vision_config=dict(depth=1, hidden_size=32, intermediate_size=32, num_heads=2, out_hidden_size=64),
        image_token_id=IMG)
    model = Qwen3_5Model(cfg)
    rng = random.Random(0)
    for n_images in (0, 1, 2, 3):
        for _ in range(5):
            ids, grids = random_layout(rng, n_images)
            ref, _ = model.get_rope_index(ids[None], (ids == IMG).int()[None],
                                          image_grid_thw=grids if n_images else None)
            assert torch.equal(mrope_position_ids(ids, grids, IMG), ref[:, 0])


def test_text_only_positions_are_plain_arange():
    pos = mrope_position_ids(torch.arange(10), None, IMG)
    assert torch.equal(pos, torch.arange(10).expand(3, -1))


def test_expand_image_pads():
    assert _expand_image_pads([1, IMG, 2, IMG, 3], IMG, [3, 2]) == [1, IMG, IMG, IMG, 2, IMG, IMG, 3]
    # a prefix render holding only the first image expands identically
    assert _expand_image_pads([1, IMG, 2], IMG, [3, 2]) == [1, IMG, IMG, IMG, 2]


def test_image_refs_in_render_order_and_video_rejected():
    msgs = [{"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "x"},
                                         {"image_url": {"url": "data:b"}}]},
            {"role": "assistant", "content": "ok"},
            {"role": "user", "content": [{"type": "image", "image": "c.png"}]}]
    assert _image_refs(msgs) == ["a.png", "data:b", "c.png"]
    with pytest.raises(ValueError):
        _image_refs([{"role": "user", "content": [{"type": "video", "video": "v.mp4"}]}])


def make_doc(rng, n_images, tag):
    ids, grids = random_layout(rng, n_images)
    labels = torch.full_like(ids, -100)
    labels[-3:] = ids[-3:]
    turn = torch.where(labels != -100, 0, -1)
    s = {"input_ids": ids, "labels": labels, "resp_block": turn.clone(), "turn_id": turn}
    if n_images:
        s.update(mrope_pos=mrope_position_ids(ids, grids, IMG), image_grid_thw=grids,
                 pixel_values=torch.full((int(grids.prod(-1).sum()), 4), float(tag)))
    return _block_align_doc(s, BD, 0)


def test_packing_keeps_images_in_doc_order_and_resets_positions():
    rng = random.Random(1)
    docs = [make_doc(rng, n, tag) for tag, n in enumerate([2, 0, 1, 3, 0, 1], start=1)]
    rows = list(_stream_first_fit(iter(docs), 4096, buffer_size=8))
    tok = type("T", (), {"pad_token_id": 0, "eos_token_id": 0})()
    batch = StreamPackedMultiTurnCollator(tok, bd_size=BD, max_length=4096, length_buckets=(4096,))(rows)
    order = [s for row in rows for s in row]
    assert torch.equal(batch["image_grid_thw"], torch.cat([s["image_grid_thw"] for s in order if "image_grid_thw" in s]))
    assert torch.equal(batch["pixel_values"], torch.cat([s["pixel_values"] for s in order if "pixel_values" in s]))
    for b, row in enumerate(rows):
        seg = batch["seg_id"][b]
        for k, s in enumerate(row):
            pos = batch["mrope_pos"][b][:, seg == k]
            want = s["mrope_pos"] if "mrope_pos" in s else torch.arange(s["input_ids"].shape[0]).expand(3, -1)
            assert torch.equal(pos, want) and int(pos[:, 0].max()) == 0  # each doc restarts at 0
    assert int((batch["input_ids"] == IMG).sum()) == int(batch["image_grid_thw"].prod(-1).sum()) // 4


def test_block_align_continues_positions():
    rng = random.Random(2)
    ids, grids = random_layout(rng, 1)
    T = ids.shape[0]
    s = {"input_ids": ids, "labels": ids.clone(), "resp_block": torch.zeros_like(ids),
         "turn_id": torch.zeros_like(ids), "mrope_pos": mrope_position_ids(ids, grids, IMG)}
    out = _block_align_doc(s, 64, 0)
    pad = out["input_ids"].shape[0] - T
    assert out["input_ids"].shape[0] % 64 == 0 and out["mrope_pos"].shape[1] == T + pad
    nxt = int(s["mrope_pos"].max()) + 1
    assert torch.equal(out["mrope_pos"][:, T:], torch.arange(nxt, nxt + pad).expand(3, -1))


TOKENIZER = os.environ.get("TRIDA_TEST_TOKENIZER")
needs_tokenizer = pytest.mark.skipif(not TOKENIZER, reason="set TRIDA_TEST_TOKENIZER to a local Qwen3.5 checkpoint")


@needs_tokenizer
def test_image_conversation_builds_prefix_stable(tmp_path):
    from PIL import Image
    from transformers import AutoTokenizer

    from train.data.text_sft_data import load_image_processor
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ip = load_image_processor(TOKENIZER)
    Image.new("RGB", (320, 256), (1, 2, 3)).save(tmp_path / "a.png")
    Image.new("RGB", (640, 480), (4, 5, 6)).save(tmp_path / "b.png")
    ex = {"image_root": str(tmp_path), "messages": [
        {"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "What color?"}]},
        {"role": "assistant", "content": "<think>\nLook.\n</think>\n\nGreen."},
        {"role": "user", "content": [{"type": "image", "image": "b.png"}, {"type": "text", "text": "And this?"}]},
        {"role": "assistant", "content": "Red."}]}
    s = build_sample_multiturn(tok, ex, BD, keep_all_reasoning=True, image_processor=ip)
    assert s is not None
    pad = tok.convert_tokens_to_ids("<|image_pad|>")
    img = s["input_ids"] == pad
    assert int(img.sum()) == int(s["image_grid_thw"].prod(-1).sum()) // 4 == 80 + 300
    assert not (s["labels"][img] != -100).any()
    assert len(set(s["turn_id"][s["turn_id"] >= 0].tolist())) == 2  # both turns, incl. the no-think one
    assert torch.equal(s["mrope_pos"], mrope_position_ids(s["input_ids"], s["image_grid_thw"], pad))
    # a per-record cap overrides the default: a 2000x1500 image is ~1k tokens at 1 MP, ~2.9k at 4 MP
    Image.new("RGB", (2000, 1500), (7, 8, 9)).save(tmp_path / "big.png")
    big = {"image_root": str(tmp_path), "messages": [
        {"role": "user", "content": [{"type": "image", "image": "big.png"}, {"type": "text", "text": "?"}]},
        {"role": "assistant", "content": "ok"}]}
    n_default = int((build_sample_multiturn(tok, big, BD, True, image_processor=ip)["input_ids"] == pad).sum())
    n_4mp = int((build_sample_multiturn(tok, dict(big, max_image_pixels=4 * 1024 * 1024), BD, True,
                                        image_processor=ip)["input_ids"] == pad).sum())
    assert n_default <= 1024 < n_4mp <= 4096


@needs_tokenizer
def test_no_think_turn_is_supervised():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    ex = {"messages": [{"role": "user", "content": "2+2?"}, {"role": "assistant", "content": "4"}]}
    s = build_sample_multiturn(tok, ex, BD, keep_all_reasoning=True)
    assert tok.decode(s["input_ids"][s["labels"] != -100]) == "<think>\n\n</think>\n\n4<|im_end|>\n"
