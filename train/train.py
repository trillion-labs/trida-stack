"""Native-torch FSDP2 training harness for HF block-diffusion (any HF causal LM).

Launch with torchrun:

    torchrun --nnodes=1 --nproc_per_node=8 train.py \
        --model_id Qwen/Qwen3-4B --bd_size 32 --max_length 16384 --ar_loss_weight 0.2 ...

Wraps the model with FSDP2 (``fully_shard``, bf16 compute / fp32 reduce), uses the model's internal
gradient checkpointing, AdamW + cosine LR, manual gradient accumulation, and saves HF checkpoints via
``save_pretrained`` (full state dict gathered to rank 0).
"""

import sys
from train.cli import build_parser

# Keep --help available before torch/FSDP is installed.
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    build_parser().print_help()
    sys.exit(0)

import time
import torch
import torch.distributed as dist
from train.distributed import setup_distributed, shard_model, is_master, log, world_size
from train.models.build import build_model
from train.data.loaders import build_data_loaders
from train.optimizer import build_optimizer, apply_optimizer_step
from train.checkpoint import load_train_state, save_checkpoint


def decision_micro_steps(model, decision_iter, args, dev, running):
    """The decision micro-batches that close an optimizer step: lambda * mean decision loss; the last one syncs gradients."""
    for j in range(args.decision_micro_per_step):
        jb = next(decision_iter)
        jb = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in jb.items()}
        model.set_requires_gradient_sync(j == args.decision_micro_per_step - 1)
        jloss, jlogs = model(None, None, None, decision=jb)
        (args.decision_loss_weight * jloss / args.decision_micro_per_step).backward()
        running["decision"] += jloss.item() / args.decision_micro_per_step
        running["decision_acc"] += jlogs["decision_acc"] / args.decision_micro_per_step
        running["decision_q"] += jlogs["decision_questions"]
        running["decision_rec"] += len({q["variant"] for q in jb["questions"]})
        running["decision_tok"] += jlogs["decision_tokens"]


def log_decision_progress(args, dev, running, global_step, lr, gnorm, t0, run):
    dt = time.time() - t0
    counts = torch.tensor([running["decision_q"], running["decision_tok"], running["decision_rec"]], dtype=torch.float64,
                          device=dev)
    dist.all_reduce(counts, op=dist.ReduceOp.SUM)
    log(f"step {global_step:6d} | decision {running['decision']/args.log_every:.4f} acc {running['decision_acc']/args.log_every:.3f} "
        f"| lr {lr:.2e} | gnorm {float(gnorm):.2f} | q/step {counts[0].item()/args.log_every:,.0f} "
        f"rec/step {counts[2].item()/args.log_every:,.0f} "
        f"| decision_tps {counts[1].item()/dt:,.0f} | mem {torch.cuda.max_memory_allocated() / 1e9:.1f}GB")
    if run is not None:
        run.log({"decision_loss": running["decision"]/args.log_every, "decision_acc": running["decision_acc"]/args.log_every,
                 "lr": lr, "grad_norm": float(gnorm)}, step=global_step)


def log_text_progress(args, dev, running, global_step, lr, gnorm, t0, run, decision_iter):
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
        + (f" | decision {running['decision']/args.log_every:.4f} acc {running['decision_acc']/args.log_every:.3f} "
           f"q/step {running['decision_q']/args.log_every:.0f}" if decision_iter is not None else ""))
    if run is not None:
        run.log({"loss": running["loss"]/n, "diff_loss": running["diff"]/n,
                 "ar_loss": running["ar"]/n, "lr": lr, "grad_norm": float(gnorm),
                 "peak_mem_gb": mem, "tot_tps": tot_tps, "seq_tps": seq_tps,
                 "resp_tps": resp_tps, "avg_seqlen": g_seqtok/mb,
                 "avg_ntok": g_ntok/mb,
                 **({"decision_loss": running["decision"]/args.log_every, "decision_acc": running["decision_acc"]/args.log_every}
                    if decision_iter is not None else {})}, step=global_step)


def run_training(args, dev, model, loaders, optim):
    """Resume if requested, train to max_steps, and save the final checkpoint."""
    loader, decision_iter = loaders.text, loaders.decision
    run = None
    if args.wandb and is_master():
        import wandb
        run = wandb.init(project=args.wandb_project, name=args.run_name, config=vars(args))

    # ---- train loop ----
    # --resume: exact optimizer/step state load; else --start_step: warm-start LR/ckpt continuation
    global_step = load_train_state(model, optim, args.resume) if args.resume else args.start_step
    micro = 0
    running = {"loss": 0.0, "diff": 0.0, "ar": 0.0, "ntok": 0, "seqtok": 0, "tottok": 0,
               "decision": 0.0, "decision_acc": 0.0, "decision_q": 0, "decision_rec": 0, "decision_tok": 0}
    t0 = time.time()
    done = False
    while args.decision_only and not done:   # Decision-only: each optimizer step is just the Decision micro-batches
        decision_micro_steps(model, decision_iter, args, dev, running)
        lr, gnorm = apply_optimizer_step(model, optim, global_step, args)
        global_step += 1
        if global_step % args.log_every == 0:
            log_decision_progress(args, dev, running, global_step, lr, gnorm, t0, run)
            running = {k: 0 for k in running}
            t0 = time.time()
        if global_step % args.save_every == 0:
            save_checkpoint(model, optim, global_step, args)
        done = global_step >= args.max_steps
    while not done:
        for batch in loader:
            if batch is None:  # collator dropped every sample in this micro-batch
                continue
            batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
            is_boundary = (micro + 1) % args.grad_accum == 0
            model.set_requires_gradient_sync(is_boundary and decision_iter is None)   # with Decision, its last micro syncs
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
            # Decision micro-batches close the step: lambda * mean decision loss, SFT gradients keep their scale.
            if decision_iter is not None:
                decision_micro_steps(model, decision_iter, args, dev, running)

            lr, gnorm = apply_optimizer_step(model, optim, global_step, args)
            global_step += 1

            if global_step % args.log_every == 0:
                log_text_progress(args, dev, running, global_step, lr, gnorm, t0, run, decision_iter)
                running = {k: 0 for k in running}
                t0 = time.time()

            if global_step % args.save_every == 0:
                save_checkpoint(model, optim, global_step, args)

            if global_step >= args.max_steps:
                done = True
                break

    save_checkpoint(model, optim, global_step, args)
    if run is not None:
        run.finish()


def main():
    args = build_parser().parse_args()
    device = setup_distributed(args)
    model = build_model(args)
    shard_model(model, args)
    loaders = build_data_loaders(args, model)
    optimizer = build_optimizer(args, model)

    run_training(args, device, model, loaders, optimizer)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
