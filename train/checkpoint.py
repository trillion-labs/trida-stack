"""Save model weights and distributed optimizer state; resume training."""

import json
import os
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_optimizer_state_dict, set_optimizer_state_dict,
)
from train.distributed import is_master, log


def _set_config_dtype(path, dtype):
    """With --fp32_master the module holds fp32 masters, so save_pretrained records "float32" in
    config.json although the weights are written in the compute dtype. Make the config match the
    weights (the field is ``dtype`` in transformers 5, ``torch_dtype`` in 4.x)."""
    name = str(dtype).replace("torch.", "")

    def fix(d):
        for k, v in d.items():
            if k in ("dtype", "torch_dtype") and isinstance(v, str):
                d[k] = name
            elif isinstance(v, dict):
                fix(v)

    with open(path) as f:
        cfg = json.load(f)
    fix(cfg)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)


def save_hf(model, args, tag):
    """Gather the full (unsharded) HF weights to rank 0 and write a from_pretrained-loadable dir."""
    sd = get_model_state_dict(
        model, options=StateDictOptions(full_state_dict=True, cpu_offload=True)
    )
    if args.fp32_master:
        sd = {k: v.to(model.compute_dtype) if v.is_floating_point() else v for k, v in sd.items()}
    if is_master():
        out = os.path.join(args.save_dir, tag)
        # strip the "model." prefix so keys match the underlying HF model
        hf_sd = {k[len("model."):]: v for k, v in sd.items() if k.startswith("model.")}
        model.model.save_pretrained(out, state_dict=hf_sd)
        if args.fp32_master:
            _set_config_dtype(os.path.join(out, "config.json"), model.compute_dtype)
        model.tokenizer.save_pretrained(out)
        if getattr(model, "image_processor", None) is not None:
            model.image_processor.save_pretrained(out)  # preprocessor_config.json for VL checkpoints
        if getattr(model, "decision_head", None) is not None:   # the decision pointer head (export: train/tools/export_kev_checkpoint.py)
            torch.save({"head": {k[len("jev_head."):]: v for k, v in sd.items() if k.startswith("jev_head.")},
                        "head_dim": model.decision_head.q.out_features}, os.path.join(out, "jev_head.pt"))
        with open(os.path.join(out, "block_diffusion.json"), "w") as f:
            json.dump({"bd_size": model.bd_size, "mask_id": model.mask_id,
                       "ar_loss_weight": model.ar_loss_weight}, f, indent=2)
        log(f"[save] wrote {out}")
    dist.barrier()


def save_train_state(model, optim, global_step, args, tag):
    """DCP-save the optimizer state (sharded, FSDP2-aware) + global_step next to the HF weights, so a
    run can be resumed exactly. Model weights already live in the HF dir (save_hf)."""
    # NOTE: every early-return / skip here MUST be decided identically on all ranks. dcp.save and
    # get_optimizer_state_dict are COLLECTIVES: a per-rank try/except around them diverges the ranks
    # (the failing rank races ahead to the barrier while the others sit in the collective), which
    # turns a clean error into a 10-minute NCCL watchdog timeout + SIGABRT. Hence a uniform flag.
    if not args.save_optim_state:
        return
    out = os.path.join(args.save_dir, tag)
    # save_hf() just gathered the full state dict, so allocated memory peaks right here. NCCL's comm
    # buffers live OUTSIDE the torch caching allocator, so the DCP gather can die with "NCCL Error 1:
    # unhandled cuda error" (a disguised CUDA OOM). Release the cache first to leave NCCL room.
    torch.cuda.empty_cache()
    osd = get_optimizer_state_dict(model, optim)  # collective on all ranks
    dcp.save({"optim": osd}, checkpoint_id=os.path.join(out, "train_state"))
    if is_master():
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "train_state.json"), "w") as f:
            json.dump({"global_step": global_step}, f)
    dist.barrier()


def load_train_state(model, optim, resume_dir):
    """Restore optimizer state + global_step from a checkpoint dir. Returns the global_step to
    continue from, or 0 (warm restart, fresh optimizer) if the dir has only HF weights."""
    state_dir = os.path.join(resume_dir, "train_state")
    meta = os.path.join(resume_dir, "train_state.json")
    if not (os.path.isdir(state_dir) and os.path.exists(meta)):
        log(f"[resume] {resume_dir} has no optimizer state -> WARM restart (fresh optimizer, step 0)")
        return 0
    osd = get_optimizer_state_dict(model, optim)  # template with the right structure
    dcp.load({"optim": osd}, checkpoint_id=state_dir)
    set_optimizer_state_dict(model, optim, optim_state_dict=osd)
    with open(meta) as f:
        gs = int(json.load(f)["global_step"])
    log(f"[resume] restored optimizer + resumed at global_step={gs} from {resume_dir}")
    return gs


def save_checkpoint(model, optim, global_step, args):
    """All ranks save weights first, then optional optimizer state in the same order."""
    tag = f"step_{global_step}"
    save_hf(model, args, tag)
    save_train_state(model, optim, global_step, args, tag)
