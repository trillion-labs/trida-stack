#!/usr/bin/env python3
"""1-GPU smoke for image+text training on the FLARE two-stream path.

    python train/tools/vlm_smoke.py --text <flat text ckpt> --vl <same ckpt + vision tower>

(build ``--vl`` with ``train/tools/compose_vl_checkpoint.py``). Checks, in order:
  1. text-only regression: the VL checkpoint gives the SAME loss as the flat checkpoint on the same
     text-only packed batch and seed (the vision tower must change nothing for text);
  2. image batches: finite loss, non-zero projector (merger) grads, no grads on the frozen encoder;
  3. a few optimizer steps on synthetic colored-image Q&A lower the loss.
"""
import argparse
import os
import random
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.data.text_sft_data import (  # noqa: E402
    PackedMultiTurnRowDataset,
    StreamPackedMultiTurnCollator,
    load_image_processor,
)
from train.hf_block_diffusion_hybrid import HFBlockDiffusionHybrid  # noqa: E402

BD = 8
COLORS = {"red": (220, 30, 30), "green": (30, 200, 60), "blue": (40, 70, 230), "yellow": (240, 220, 30)}
TEXT_CONVS = [
    {"messages": [{"role": "user", "content": "Name a prime number below 10."},
                  {"role": "assistant", "content": "<think>\nPrimes below 10: 2, 3, 5, 7.\n</think>\n\n7."}]},
    {"messages": [{"role": "user", "content": "Capital of France?"}, {"role": "assistant", "content": "Paris."},
                  {"role": "user", "content": "And of Italy?"}, {"role": "assistant", "content": "Rome."}]},
]


def build(model_id):
    return HFBlockDiffusionHybrid(model_id=model_id, bd_size=BD, ar_loss_weight=0.1, full_mask=True,
                                  loss_weighting="uniform", grad_checkpoint=True,
                                  linear_block_mode="bidirectional").cuda()


def batches(model, convs, image_processor=None, max_length=4096):
    ds = PackedMultiTurnRowDataset(convs, model.tokenizer, bd_size=BD, max_length=max_length,
                                   max_response_length=max_length, keep_all_reasoning=True,
                                   image_processor=image_processor)
    collate = StreamPackedMultiTurnCollator(model.tokenizer, bd_size=BD, max_length=max_length,
                                            length_buckets=(1024, 2048, 4096))
    return [{k: v.cuda() for k, v in collate([row]).items()} for row in ds]


def run(model, batch, seed=0):
    torch.manual_seed(seed)
    images = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "mrope_pos") if k in batch}
    return model(batch["input_ids"], batch["labels"], batch["attention_mask"], seg_id=batch["seg_id"],
                 turn_id=batch["turn_id"], **images)


def image_convs(root, n):
    rng = random.Random(0)
    from PIL import Image
    convs = []
    for i in range(n):
        name, rgb = rng.choice(list(COLORS.items()))
        Image.new("RGB", (rng.choice([224, 256, 320]), rng.choice([224, 256])), rgb).save(f"{root}/{i}.png")
        convs.append({"image_root": root, "messages": [
            {"role": "user", "content": [{"type": "image", "image": f"{i}.png"},
                                         {"type": "text", "text": "What color is this image? One word."}]},
            {"role": "assistant", "content": f"{name.capitalize()}."}]})
    return convs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text", required=True)
    ap.add_argument("--vl", required=True)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    flat = build(a.text)
    text_batch = batches(flat, TEXT_CONVS)[0]
    with torch.no_grad():
        ref, _ = run(flat, text_batch)
    del flat
    torch.cuda.empty_cache()

    vl = build(a.vl)
    assert vl.is_vl
    with torch.no_grad():
        got, _ = run(vl, text_batch)
    print(f"[1] text-only loss: flat {ref.item():.6f}  vl {got.item():.6f}  |diff| {abs(ref - got).item():.2e}")
    assert torch.equal(ref, got), "VL checkpoint changed the text-only loss"
    # a text-only batch must still run the vision tower (FSDP lockstep): projector grads exist and are 0
    vl.train()
    loss, _ = run(vl, text_batch)
    loss.backward()
    mg = [p.grad for n, p in vl.visual.named_parameters() if n.startswith("merger.")]
    print(f"    text-only backward: projector grads present {sum(g is not None for g in mg)}/{len(mg)}, "
          f"all zero {all(g is not None and not g.any() for g in mg)}")
    assert all(g is not None and not g.any() for g in mg), "text-only batch skipped the vision tower"
    vl.zero_grad(set_to_none=True)

    ip = load_image_processor(a.vl)
    root = tempfile.mkdtemp()
    img_batches = batches(vl, image_convs(root, 96), ip, max_length=2048)
    print(f"    {len(img_batches)} image rows, {int(img_batches[0]['image_grid_thw'].shape[0])} images in row 0")

    vl.train()
    loss, logs = run(vl, img_batches[0])
    loss.backward()
    merger = [p.grad for n, p in vl.visual.named_parameters() if n.startswith("merger.")]
    encoder = [p.grad for n, p in vl.visual.named_parameters() if not n.startswith("merger.")]
    print(f"[2] image loss {loss.item():.4f} | merger grad norm {torch.stack([g.norm() for g in merger]).norm():.3e} "
          f"| encoder grads {sum(g is not None for g in encoder)}/{len(encoder)}")
    assert torch.isfinite(loss) and all(g is not None and g.abs().sum() > 0 for g in merger)
    assert all(g is None for g in encoder), "frozen vision encoder received gradients"

    groups = vl.param_groups(projector_lr_scale=0.1)
    opt = torch.optim.AdamW(groups, lr=1e-5)
    for g in opt.param_groups:
        g["lr"] = 1e-5 * g["lr_scale"]
    print(f"    optimizer groups: {[(len(g['params']), g['lr']) for g in opt.param_groups]}")
    opt.zero_grad(set_to_none=True)
    hist = []
    for step in range(a.steps):
        loss, _ = run(vl, img_batches[step % len(img_batches)], seed=step)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(vl.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        hist.append(loss.item())
    first, last = sum(hist[:5]) / 5, sum(hist[-5:]) / 5
    print(f"[3] loss over {a.steps} steps: first-5 avg {first:.4f} -> last-5 avg {last:.4f}")
    assert last < first, "loss did not decrease on synthetic image data"
    print("VLM smoke: all checks passed")


if __name__ == "__main__":
    main()
