"""Optimizer construction, cosine learning rate, and optimizer steps."""

import math
import torch
from train.distributed import log


def get_lr(it, max_lr, warmup, total, min_ratio):
    if it < warmup:
        return max_lr * (it + 1) / max(1, warmup)
    if it >= total:
        return max_lr * min_ratio
    r = (it - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * r))
    return max_lr * (min_ratio + (1.0 - min_ratio) * coeff)


def build_optimizer(args, model):
    """Create AdamW using the model's projector and Decision learning-rate groups."""
    # Discriminative LR: the vision projector trains at --projector_lr_scale x the language-model
    # LR (the vision encoder is frozen and not optimized). Text-only models get one group.
    groups = model.param_groups(projector_lr_scale=args.projector_lr_scale, decision_head_lr_scale=args.decision_head_lr_scale)
    optim = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.adam_beta2), weight_decay=args.weight_decay)
    if len(groups) > 1:
        log(f"[optim] language model lr {args.lr:g}"
            + "".join(f" | {name} lr {args.lr * scale:g} (x{scale:g})" for name, scale in
                      (("projector", args.projector_lr_scale), ("decision head", args.decision_head_lr_scale))
                      if (name != "projector" or model.is_vl) and (name != "decision head" or args.decision_data)))

    return optim


def apply_optimizer_step(model, optim, global_step, args):
    """Schedule LR, clip gradients, update parameters, then clear gradients."""
    lr = get_lr(global_step, args.lr, args.warmup, args.max_steps, args.lr_min_ratio)
    for g in optim.param_groups:
        g["lr"] = lr * g.get("lr_scale", 1.0)
    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optim.step()
    optim.zero_grad(set_to_none=True)
    return lr, gnorm
