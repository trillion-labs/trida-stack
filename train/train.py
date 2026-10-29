"""Native-torch FSDP2 training harness for HF block-diffusion (any HF causal LM).

Launch with torchrun:

    torchrun --nnodes=1 --nproc_per_node=8 train.py \
        --model_id Qwen/Qwen3-4B --bd_size 32 --max_length 16384 --ar_loss_weight 0.2 ...

Wraps the model with FSDP2 (``fully_shard``, bf16 compute / fp32 reduce), uses the model's internal
gradient checkpointing, AdamW + cosine LR, manual gradient accumulation, and saves HF checkpoints via
``save_pretrained`` (full state dict gathered to rank 0).
"""

import json
import math
import os
import sys
import time

from train.cli import build_parser

# Answer --help here, BEFORE the heavy torch/FSDP imports below.
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    build_parser().print_help()
    sys.exit(0)


import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.utils.data import DataLoader

from train.data.text_sft_data import (
    JsonlAgenticDataset,
    MultiTurnTextSftCollator,
    PackedMultiTurnRowDataset,
    PackedRowDataset,
    PackedTextSftCollator,
    StreamPackedCollator,
    StreamPackedMultiTurnCollator,
    TextSftCollator,
    download_jsonl_paths,
    load_image_processor,
)
from train.hf_block_diffusion import HFBlockDiffusion
from train.jev import JevRowDataset, load_jev_records
from train.hf_block_diffusion_hybrid import HFBlockDiffusionHybrid, is_hybrid_model_id


# --------------------------------------------------------------------------- dist helpers
def rank():        return int(os.environ.get("RANK", 0))
def local_rank():  return int(os.environ.get("LOCAL_RANK", 0))
def world_size():  return int(os.environ.get("WORLD_SIZE", 1))
def is_master():   return rank() == 0
def log(msg):
    if is_master():
        print(msg, flush=True)


def get_lr(it, max_lr, warmup, total, min_ratio):
    if it < warmup:
        return max_lr * (it + 1) / max(1, warmup)
    if it >= total:
        return max_lr * min_ratio
    r = (it - warmup) / max(1, total - warmup)
    coeff = 0.5 * (1.0 + math.cos(math.pi * r))
    return max_lr * (min_ratio + (1.0 - min_ratio) * coeff)


def _identity(x):
    return x


def jev_micro_steps(model, jev_iter, args, dev, running):
    """The Jev micro-batches that close an optimizer step: lambda * mean Jev loss; the last one syncs gradients."""
    for j in range(args.jev_micro_per_step):
        jb = next(jev_iter)
        jb = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in jb.items()}
        model.set_requires_gradient_sync(j == args.jev_micro_per_step - 1)
        jloss, jlogs = model(None, None, None, jev=jb)
        (args.jev_loss_weight * jloss / args.jev_micro_per_step).backward()
        running["jev"] += jloss.item() / args.jev_micro_per_step
        running["jev_acc"] += jlogs["jev_acc"] / args.jev_micro_per_step
        running["jev_q"] += jlogs["jev_questions"]
        running["jev_rec"] += len({q["variant"] for q in jb["questions"]})
        running["jev_tok"] += jlogs["jev_tokens"]


def parse_int_list(s):
    return tuple(int(x) for x in s.split(",")) if s else None


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
        if getattr(model, "jev_head", None) is not None:   # the Jev pointer head (export: train/tools/export_kev_checkpoint.py)
            torch.save({"head": {k[len("jev_head."):]: v for k, v in sd.items() if k.startswith("jev_head.")},
                        "head_dim": model.jev_head.q.out_features}, os.path.join(out, "jev_head.pt"))
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


def main():
    p = build_parser()
    args = p.parse_args()

    # Bind the process group to this rank's GPU (set_device FIRST, then pass device_id). Without an
    # explicit device_id, NCCL collectives that run outside the main compute stream — notably DCP's
    # gather_object in save_train_state — pick the wrong CUDA context and die with "NCCL Error 1:
    # unhandled cuda error" (and emit the "barrier(): using the device under current context" warn).
    torch.cuda.set_device(local_rank())
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank()}"))
    torch.manual_seed(args.seed)
    dev = torch.cuda.current_device()

    # ---- model (always bf16 weights) ----
    # On resume, load weights from the checkpoint dir (a save_pretrained dir with the <|mask|> token
    # already added, so HFBlockDiffusion skips the resize); otherwise from the base model.
    model_source = args.resume if args.resume else args.model_id
    if args.resume:
        log(f"[resume] loading model weights from {args.resume}")
    # Hybrid linear-attention decoders (Qwen3.5) can't take an arbitrary attention mask on their
    # linear layers, so they use the linear-aware HFBlockDiffusionHybrid; auto-detect from
    # config.layer_types. Dense models (Qwen3-4B / Tri-7B) keep the original HFBlockDiffusion path.
    hybrid = is_hybrid_model_id(model_source, trust_remote_code=args.trust_remote_code)
    common = dict(
        model_id=model_source, bd_size=args.bd_size, max_length=args.max_length,
        max_response_length=args.max_response_length,
        response_buckets=parse_int_list(args.response_buckets),
        ar_loss_weight=args.ar_loss_weight, grad_checkpoint=True, gc_min_len=0,
        activation_offload=args.activation_offload, fused_ce=args.fused_ce,
        compile_layers=args.compile_layers, compile_mlp=args.compile_mlp, compile_glue=args.compile_glue,
        merged_views=args.merged_views,
        ar_only=args.ar_only, loss_weighting=args.loss_weighting,
        pos_decay_gamma=args.pos_decay_gamma, full_mask=args.full_mask,
        within_block_causal=args.within_block_causal,
        dtype=torch.bfloat16, trust_remote_code=args.trust_remote_code,
    )
    if hybrid:
        log(f"[model] hybrid linear-attention model detected -> HFBlockDiffusionHybrid "
            f"(linear_block_mode={args.linear_block_mode})")
        model = HFBlockDiffusionHybrid(**common, linear_block_mode=args.linear_block_mode, require_kernels=(not args.allow_slow_linear),
                                       jev_head_dim=args.jev_head_dim if args.jev_data else 0)
    else:
        model = HFBlockDiffusion(**common)
    head_init = os.path.join(model_source, "jev_head.pt")
    if getattr(model, "jev_head", None) is not None and os.path.exists(head_init):   # continue from a Jev checkpoint
        saved = torch.load(head_init, map_location="cpu")
        if saved["head_dim"] != args.jev_head_dim:
            raise ValueError(f"{head_init} has head_dim {saved['head_dim']}, --jev_head_dim is {args.jev_head_dim}")
        model.jev_head.load_state_dict(saved["head"])
        log(f"[jev] pointer head initialised from {head_init}")
    tokenizer = model.tokenizer
    image_processor = None
    if model.is_vl:
        image_processor = load_image_processor(model_source)
        model.image_processor = image_processor
        log(f"[model] vision-language checkpoint: frozen vision encoder, trainable projector | "
            f"images capped at {args.max_image_pixels} px")

    if args.fp32_master:   # fp32 masters (all params, so every FSDP unit is one dtype); FSDP gathers + computes in bf16
        for p in model.parameters():
            p.data = p.data.float()
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

    # ---- data (streaming JSONL; sharded per rank x worker, no Arrow) ----
    # Every rank resolves its OWN local paths (don't broadcast rank-0's paths — on a multi-node run
    # without a shared filesystem those wouldn't exist on other nodes). rank 0 downloads first so it
    # populates the (shared or per-node) HF cache; other ranks then hit the cache / download locally.
    if args.jev_only and not args.jev_data:
        raise ValueError("--jev_only needs --jev_data")
    if not args.jev_only and not args.dataset:
        raise ValueError("--dataset is required (unless --jev_only)")
    if args.jev_data and args.jev_micro_per_step < 1:
        raise ValueError("--jev_micro_per_step must be >= 1 with --jev_data (its last micro-batch syncs gradients)")
    if not args.jev_only:
        if is_master():
            paths = download_jsonl_paths(repo_id=args.dataset)
        dist.barrier()
        if not is_master():
            paths = download_jsonl_paths(repo_id=args.dataset)
        dataset = JsonlAgenticDataset(paths, rank=rank(), world=world_size(), seed=args.seed)
        if args.pack and args.multiturn:
            # Packed MULTI-TURN (FLARE two-stream): the dataset bin-packs whole conversations (every
            # assistant turn supervised), each block-aligned + doc-isolated via seg_id, and yields one
            # <=max_length row at a time via buffered first-fit — a conversation that doesn't fit waits
            # for the next row instead of being dropped. Each micro-batch is `micro_batch_size` rows.
            # Requires linear_block_mode=bidirectional.
            assert args.linear_block_mode == "bidirectional", \
                "--pack --multiturn is the FLARE two-stream path; needs --linear_block_mode bidirectional"
            dataset = PackedMultiTurnRowDataset(
                dataset, tokenizer, bd_size=args.bd_size, max_length=args.max_length,
                max_response_length=args.max_response_length,
                keep_all_reasoning=args.keep_all_reasoning, max_segments=args.max_segments,
                image_processor=image_processor, max_image_pixels=args.max_image_pixels)
            collate = StreamPackedMultiTurnCollator(
                tokenizer, bd_size=args.bd_size, max_length=args.max_length,
                length_buckets=parse_int_list(args.length_buckets))
            batch_size = args.micro_batch_size
        elif args.pack:
            # Streaming first-fit packer with carry-over: the dataset yields one <=max_length row at a
            # time (overflow sample starts the next row; nothing dropped), so each micro-batch is exactly
            # `micro_batch_size` packed rows.
            dataset = PackedRowDataset(
                dataset, tokenizer, max_length=args.max_length,
                max_response_length=args.max_response_length,
                keep_all_reasoning=args.keep_all_reasoning, max_segments=args.max_segments,
                # AR has no region-B response copy, so charging the block-diffusion 2*n_response cost
                # would half-fill every row.
                ar_only=args.ar_only)
            collate = StreamPackedCollator(
                tokenizer, max_length=args.max_length,
                length_buckets=parse_int_list(args.length_buckets))
            batch_size = args.micro_batch_size  # packed rows per micro-batch (1 for 32k)
        elif args.multiturn:
            collate = MultiTurnTextSftCollator(
                tokenizer, bd_size=args.bd_size, max_length=args.max_length,
                max_response_length=args.max_response_length,
                length_buckets=parse_int_list(args.length_buckets),
                keep_all_reasoning=args.keep_all_reasoning)
            batch_size = args.micro_batch_size
        else:
            collate = TextSftCollator(tokenizer, max_length=args.max_length,
                                      max_response_length=args.max_response_length,
                                      length_buckets=parse_int_list(args.length_buckets),
                                      keep_all_reasoning=args.keep_all_reasoning)
            batch_size = args.micro_batch_size
    jev_iter = None
    if args.jev_data:
        if not (hybrid and args.linear_block_mode == "bidirectional"):
            raise ValueError("--jev_data needs the hybrid FLARE two-stream path (--linear_block_mode bidirectional)")
        records = load_jev_records(args.jev_data.split(","))
        jev_ds = JevRowDataset(records, tokenizer, rank=rank(), world=world_size(), seed=args.seed,
                               max_tokens=args.jev_max_tokens, max_state=args.jev_max_state,
                               max_variant_tokens=args.jev_max_variant_tokens,
                               p_none_pair=args.jev_p_none_pair)
        jev_iter = iter(DataLoader(jev_ds, batch_size=None, num_workers=2, collate_fn=_identity))
        sched = "Jev only" if args.jev_only else f"after {args.grad_accum} SFT"
        log(f"[jev] {len(records)} records | weight {args.jev_loss_weight} | {args.jev_micro_per_step} Jev "
            f"micro-batch(es) per step, {sched} | {args.jev_max_tokens} tokens each")
    loader = None
    if not args.jev_only:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=args.num_workers, collate_fn=collate, drop_last=True)
        log(f"[data] packing={'on' if args.pack else 'off'} multiturn={'on' if args.multiturn else 'off'} "
            f"collator={type(collate).__name__} batch_size={batch_size}")

    # Discriminative LR: the vision projector trains at --projector_lr_scale x the language-model
    # LR (the vision encoder is frozen and not optimized). Text-only models get one group.
    groups = model.param_groups(projector_lr_scale=args.projector_lr_scale, jev_head_lr_scale=args.jev_head_lr_scale)
    optim = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, args.adam_beta2), weight_decay=args.weight_decay)
    if len(groups) > 1:
        log(f"[optim] language model lr {args.lr:g}"
            + "".join(f" | {name} lr {args.lr * scale:g} (x{scale:g})" for name, scale in
                      (("projector", args.projector_lr_scale), ("jev head", args.jev_head_lr_scale))
                      if (name != "projector" or model.is_vl) and (name != "jev head" or args.jev_data)))

    run = None
    if args.wandb and is_master():
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # ---- train loop ----
    # --resume: exact optimizer/step state load; else --start_step: warm-start LR/ckpt continuation
    global_step = load_train_state(model, optim, args.resume) if args.resume else args.start_step
    micro = 0
    running = {"loss": 0.0, "diff": 0.0, "ar": 0.0, "ntok": 0, "seqtok": 0, "tottok": 0,
               "jev": 0.0, "jev_acc": 0.0, "jev_q": 0, "jev_rec": 0, "jev_tok": 0}
    t0 = time.time()
    done = False
    while args.jev_only and not done:   # Jev-only: each optimizer step is just the Jev micro-batches
        jev_micro_steps(model, jev_iter, args, dev, running)
        lr = get_lr(global_step, args.lr, args.warmup, args.max_steps, args.lr_min_ratio)
        for g in optim.param_groups:
            g["lr"] = lr * g.get("lr_scale", 1.0)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optim.step()
        optim.zero_grad(set_to_none=True)
        global_step += 1
        if global_step % args.log_every == 0:
            dt = time.time() - t0
            counts = torch.tensor([running["jev_q"], running["jev_tok"], running["jev_rec"]], dtype=torch.float64,
                                  device=dev)
            dist.all_reduce(counts, op=dist.ReduceOp.SUM)
            log(f"step {global_step:6d} | jev {running['jev']/args.log_every:.4f} acc {running['jev_acc']/args.log_every:.3f} "
                f"| lr {lr:.2e} | gnorm {float(gnorm):.2f} | q/step {counts[0].item()/args.log_every:,.0f} "
                f"rec/step {counts[2].item()/args.log_every:,.0f} "
                f"| jev_tps {counts[1].item()/dt:,.0f} | mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
            if run is not None:
                run.log({"jev_loss": running["jev"]/args.log_every, "jev_acc": running["jev_acc"]/args.log_every,
                         "lr": lr, "grad_norm": float(gnorm)}, step=global_step)
            running = {k: 0 for k in running}
            t0 = time.time()
        if global_step % args.save_every == 0:
            save_hf(model, args, f"step_{global_step}")
            save_train_state(model, optim, global_step, args, f"step_{global_step}")
        done = global_step >= args.max_steps
    while not done:
        for batch in loader:
            if batch is None:  # collator dropped every sample in this micro-batch
                continue
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            is_boundary = (micro + 1) % args.grad_accum == 0
            model.set_requires_gradient_sync(is_boundary and jev_iter is None)   # with Jev, its last micro syncs
            images = {k: batch[k] for k in ("pixel_values", "image_grid_thw", "mrope_pos") if k in batch}
            loss, logs = model(batch["input_ids"], batch["labels"], batch["attention_mask"],
                               seg_id=batch.get("seg_id"), resp_pos=batch.get("resp_pos"),
                               resp_block=batch.get("resp_block"), turn_id=batch.get("turn_id"),
                               **images)
            (loss / args.grad_accum).backward()
            running["loss"] += loss.item()
            running["diff"] += logs["diff_loss"].item()
            running["ar"] += logs["ar_loss"].item()
            running["ntok"] += logs["ntok"]  # supervised (response) tokens this micro-batch
            running["seqtok"] += int(batch["attention_mask"].sum().item())  # real (non-pad) tokens
            running["tottok"] += int(batch["input_ids"].numel())  # all tokens incl. pad (B*L)
            micro += 1
            if not is_boundary:
                continue
            # Jev micro-batches close the step: lambda * mean Jev loss, SFT gradients keep their scale.
            if jev_iter is not None:
                jev_micro_steps(model, jev_iter, args, dev, running)

            lr = get_lr(global_step, args.lr, args.warmup, args.max_steps, args.lr_min_ratio)
            for g in optim.param_groups:
                g["lr"] = lr * g.get("lr_scale", 1.0)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

            if global_step % args.log_every == 0:
                n = args.grad_accum * args.log_every
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9
                # Sum token counts across ranks so the reported throughput is GLOBAL (all GPUs).
                counts = torch.tensor([running["ntok"], running["seqtok"], running["tottok"]],
                                      dtype=torch.float64, device=dev)
                dist.all_reduce(counts, op=dist.ReduceOp.SUM)
                g_ntok, g_seqtok, g_tottok = counts[0].item(), counts[1].item(), counts[2].item()
                tot_tps = g_tottok / dt   # all tokens incl. pad/s, global (GPU-work throughput)
                seq_tps = g_seqtok / dt   # real (non-pad) sequence tokens/s, global
                resp_tps = g_ntok / dt    # supervised (response) tokens/s, global
                mb = n * world_size()     # micro-batches this interval across all ranks
                log(f"step {global_step:6d} | loss {running['loss']/n:.4f} "
                    f"diff {running['diff']/n:.4f} ar {running['ar']/n:.4f} "
                    f"| lr {lr:.2e} | gnorm {float(gnorm):.2f} | {n/dt:.2f} mb/s "
                    f"| tot_tps {tot_tps:,.0f} seq_tps {seq_tps:,.0f} resp_tps {resp_tps:,.0f} "
                    f"| seqlen {g_seqtok/mb:,.0f} ntok {g_ntok/mb:,.0f} | mem {mem:.1f}GB"
                    + (f" | jev {running['jev']/args.log_every:.4f} acc {running['jev_acc']/args.log_every:.3f} "
                       f"q/step {running['jev_q']/args.log_every:.0f}" if jev_iter is not None else ""))
                if run is not None:
                    run.log({"loss": running["loss"]/n, "diff_loss": running["diff"]/n,
                             "ar_loss": running["ar"]/n, "lr": lr, "grad_norm": float(gnorm),
                             "peak_mem_gb": mem, "tot_tps": tot_tps, "seq_tps": seq_tps,
                             "resp_tps": resp_tps, "avg_seqlen": g_seqtok/mb,
                             "avg_ntok": g_ntok/mb,
                             **({"jev_loss": running["jev"]/args.log_every, "jev_acc": running["jev_acc"]/args.log_every}
                                if jev_iter is not None else {})}, step=global_step)
                running = {k: 0 for k in running}
                t0 = time.time()

            if global_step % args.save_every == 0:
                save_hf(model, args, f"step_{global_step}")
                save_train_state(model, optim, global_step, args, f"step_{global_step}")

            if global_step >= args.max_steps:
                done = True
                break

    save_hf(model, args, f"step_{global_step}")
    save_train_state(model, optim, global_step, args, f"step_{global_step}")
    if run is not None:
        run.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
