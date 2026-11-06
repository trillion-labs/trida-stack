"""Distributed initialization and FSDP model sharding."""

import os
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard


def rank():        return int(os.environ.get("RANK", 0))
def local_rank():  return int(os.environ.get("LOCAL_RANK", 0))
def world_size():  return int(os.environ.get("WORLD_SIZE", 1))
def is_master():   return rank() == 0
def log(msg):
    if is_master():
        print(msg, flush=True)


def setup_distributed(args):
    """Bind this rank to its GPU, start NCCL, and seed training."""
    # Bind the process group to this rank's GPU (set_device FIRST, then pass device_id). Without an
    # explicit device_id, NCCL collectives that run outside the main compute stream — notably DCP's
    # gather_object in save_train_state — pick the wrong CUDA context and die with "NCCL Error 1:
    # unhandled cuda error" (and emit the "barrier(): using the device under current context" warn).
    torch.cuda.set_device(local_rank())
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank()}"))
    torch.manual_seed(args.seed)
    dev = torch.cuda.current_device()

    return dev


def shard_model(model, args):
    """Shard child units before the root and initialize their FSDP state."""
    # ---- FSDP2: shard each decoder layer (+ vision block) + the root; bf16 params, fp32 reduce ----
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for layer in model.tm.layers:
        fully_shard(layer, mp_policy=mp, reshard_after_forward=not args.fsdp_keep_params)
    if model.is_vl:
        for block in model.visual.blocks:
            fully_shard(block, mp_policy=mp, reshard_after_forward=not args.fsdp_keep_params)
        fully_shard(model.visual.merger, mp_policy=mp)
    if getattr(model, "jev_head", None) is not None:
        # Own unit: the head only has gradients on the (syncing) Jev micro-step, while the root's other params
        # accumulate fp32 grads over the SFT micro-steps; FSDP2 reduce-scatters a unit at one gradient dtype.
        fully_shard(model.jev_head, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    # FSDP2 creates a unit's unsharded params on its first all-gather, and its end-of-backward hook visits every
    # unit — also one a micro-step never ran (the Jev head on SFT steps, the vision projector on text-only rows),
    # which then fails before that unit's first use. Gather + release each unit once so they all exist.
    for m in model.modules():
        if isinstance(m, FSDPModule) and m is not model:
            m.unshard()
            m.reshard()
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    log(f"[model] {args.model_id} sharded across {world_size()} ranks | "
        f"{'fp32 master' if args.fp32_master else 'bf16'} weights | "
        f"trainable {n_train / 1e9:.2f}B of {n_all / 1e9:.2f}B params")
