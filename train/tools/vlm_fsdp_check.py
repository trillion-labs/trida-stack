#!/usr/bin/env python3
"""2-GPU FSDP regression check for image+text training: one rank gets a row WITH images, the other a
text-only row, for several steps. Before the fix this deadlocked (the text-only rank skipped the
vision tower's collectives -> NCCL watchdog timeout in a backward reduce-scatter).

    torchrun --nproc_per_node=2 train/tools/vlm_fsdp_check.py --vl <ckpt>-vl
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
from train.tools.vlm_smoke import TEXT_CONVS, batches, build, image_convs, run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--vl", required=True)
    ap.add_argument("--steps", type=int, default=4)
    a = ap.parse_args()
    rank = int(os.environ["RANK"])
    torch.cuda.set_device(rank)
    # short timeout so a regression fails fast instead of hanging for 10 minutes
    dist.init_process_group("nccl", timeout=datetime.timedelta(seconds=120),
                            device_id=torch.device(f"cuda:{rank}"))
    model = build(a.vl)
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.tm.layers:
        fully_shard(layer, mp_policy=mp)
    for block in model.visual.blocks:
        fully_shard(block, mp_policy=mp)
    fully_shard(model.visual.merger, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    if rank == 0:
        rows = batches(model, image_convs(tempfile.mkdtemp(), 24), load_image_processor(a.vl), max_length=2048)
    else:
        rows = batches(model, TEXT_CONVS)
    opt = torch.optim.AdamW(model.param_groups(projector_lr_scale=0.1), lr=1e-6)
    model.train()
    for step in range(a.steps):
        b = rows[step % len(rows)]
        loss, _ = run(model, b, seed=step)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        print(f"rank {rank} step {step}: {'images' if 'pixel_values' in b else 'text-only'} row, loss {loss.item():.4f}",
              flush=True)
    dist.barrier()
    if rank == 0:
        print("VL FSDP check: mixed image/text-only ranks stayed in lockstep")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
