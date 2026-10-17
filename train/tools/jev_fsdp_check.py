#!/usr/bin/env python3
"""2-GPU FSDP check for joint SFT + Jev training on a VL checkpoint: each optimizer step is one SFT micro-step without
gradient sync (rank 0 an image row, rank 1 a text-only row) followed by the Jev micro-step that syncs, as train.py
does with --jev_data. The vision projector only gets gradient from the SFT micro-step, so it must still take part in
the syncing Jev micro-step, or that gradient is never reduce-scattered and the projector never trains.

Checks every step that the projector's local shard moved on both ranks (it has a gradient) and that both ranks agree on
the step's loss reduction. --no_touch removes the Jev step's vision-tower call to show the failure.

    torchrun --nproc_per_node=2 train/tools/jev_fsdp_check.py --vl <ckpt>-vl [--no_touch]
"""
import argparse
import datetime
import os
import sys
import tempfile

import torch
import torch.distributed as dist
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from train.data.text_sft_data import load_image_processor  # noqa: E402
from train.hf_block_diffusion_hybrid import HFBlockDiffusionHybrid  # noqa: E402
from train.jev import encode_rows, materialize, pack_jev, training_context  # noqa: E402
from train.tools.vlm_smoke import BD, TEXT_CONVS, batches, image_convs, run  # noqa: E402

REQ = {"state": "Customer writes: I was charged twice for my October invoice and want the duplicate refunded.",
       "questions": {
           "team": {"type": "choice", "instructions": "Which team should handle this?",
                    "criteria": {"billing": "payments and refunds", "tech": "product bugs", "sales": "new purchases"},
                    "label": "billing"},
           "urgent": {"type": "noul", "instructions": "Is this urgent?", "label": False}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vl", required=True)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--no_touch", action="store_true", help="drop the Jev step's vision-tower call (shows the bug)")
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120),
                            device_id=torch.device(f"cuda:{rank}"))
    model = HFBlockDiffusionHybrid(model_id=a.vl, bd_size=BD, ar_loss_weight=0.1, full_mask=True,
                                   loss_weighting="uniform", grad_checkpoint=True,
                                   linear_block_mode="bidirectional", jev_head_dim=256).cuda()
    if a.no_touch:
        model._touch_vision_tower = lambda x: x
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.tm.layers:
        fully_shard(layer, mp_policy=mp)
    for block in model.visual.blocks:
        fully_shard(block, mp_policy=mp)
    fully_shard(model.visual.merger, mp_policy=mp)
    fully_shard(model.jev_head, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    for m in model.modules():   # as train.py: create every unit's unsharded params once
        if m is not model and hasattr(m, "unshard"):
            m.unshard()
            m.reshard()
    if rank == 0:
        rows = batches(model, image_convs(tempfile.mkdtemp(), 24), load_image_processor(a.vl), max_length=2048)
    else:
        rows = batches(model, TEXT_CONVS)
    tok = model.tokenizer
    jb = pack_jev([encode_rows(tok, materialize(REQ), **training_context())], tok)
    jb = {k: (v.cuda() if torch.is_tensor(v) else v) for k, v in jb.items()}
    opt = torch.optim.AdamW(model.param_groups(projector_lr_scale=1.0, jev_head_lr_scale=10.0), lr=1e-5)
    proj = [p for p in model.visual.merger.parameters()]
    model.train()
    ok = True
    for step in range(a.steps):
        before = [p.to_local().detach().clone() for p in proj]
        model.set_requires_gradient_sync(False)   # SFT micro-step: accumulate, no reduce-scatter
        b = rows[step % len(rows)]
        loss, _ = run(model, b, seed=step)
        loss.backward()
        model.set_requires_gradient_sync(True)    # Jev micro-step: the one that syncs
        jloss, _ = model(None, None, None, jev=jb)
        jloss.backward()
        has_grad = all(p.grad is not None for p in proj)
        opt.step()
        opt.zero_grad(set_to_none=True)
        moved = sum(float((p.to_local().detach() - q).abs().sum()) for p, q in zip(proj, before))
        flag = torch.tensor([float(has_grad and moved > 0)], device="cuda")
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        ok &= bool(flag.item())
        print(f"rank {rank} step {step}: {'images' if 'pixel_values' in b else 'text-only'} SFT row, "
              f"sft {loss.item():.4f} jev {jloss.item():.4f} | projector grad {has_grad} moved {moved:.3e}", flush=True)
    dist.barrier()
    if rank == 0:
        print("Jev FSDP check:", "PASS — the projector trains in joint SFT + Jev steps" if ok
              else "FAIL — the projector's SFT gradient was never applied")
    dist.destroy_process_group()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
