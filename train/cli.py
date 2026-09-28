"""Torch-free CLI definition for the block-diffusion trainer.

Kept separate from train/train.py so `train.py --help` works before torch/FSDP are
installed: importing this module pulls in only argparse.
"""
import argparse


def build_parser():
    """Torch-free CLI definition, so `train.py --help` works before torch/FSDP are installed."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen3-4B")
    p.add_argument("--resume", default=None,
                   help="checkpoint dir to resume from (loads HF weights from it; also loads "
                        "optimizer state + global_step if train_state/ is present, else warm restart)")
    p.add_argument("--save_optim_state", action="store_true",
                   help="also DCP-save the optimizer state next to each HF checkpoint (enables exact "
                        "--resume). Needs GPU headroom for the gather. Without it, checkpoints are HF "
                        "weights only -> warm-restart via --model_id <ckpt> --start_step N.")
    p.add_argument("--trust_remote_code", action="store_true")
    p.add_argument("--dataset", required=True,
                   help="HF Hub dataset id or local dir of *.jsonl chat records "
                        "(each line: {\"messages\": [...], \"tools\": [...]})")
    p.add_argument("--bd_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=16384)
    p.add_argument("--max_response_length", type=int, default=4096)
    p.add_argument("--length_buckets", default="2048,4096,8192,16384")
    p.add_argument("--response_buckets", default="512,1024,2048,4096")
    p.add_argument("--ar_loss_weight", type=float, default=0.2)
    # pure autoregressive SFT: standard causal-LM next-token loss, NO block-diffusion masking. The
    # base model is a causal LM, so this just fine-tunes it token-causally on the same (multiturn)
    # data view. No <|mask|> token / embedding resize; the checkpoint is a plain HF causal LM served
    # by the AR server. Mutually exclusive with the diffusion objective (ar_loss_weight is ignored).
    p.add_argument("--ar_only", action="store_true",
                   help="pure AR SFT (causal-LM next-token loss); disables block-diffusion masking")
    # diffusion loss weighting over masked tokens. "weighted" = the MDLM/WeDLM 1/gamma ELBO weight
    # (gamma = per-block mask rate), normalized to sum 1 across the batch's masked tokens — the
    # principled absorbing-diffusion objective, default. "uniform" = plain mean CE (legacy).
    p.add_argument("--loss_weighting", choices=["weighted", "uniform"], default="weighted",
                   help="diffusion CE weighting over masked tokens: 1/gamma ELBO weight (default) or mean")
    # DFlash-style intra-block positional loss decay: token at 0-indexed position p in its bd_size
    # block weighted exp(-p/gamma). Up-weights early positions (early errors invalidate later
    # parallel-committed tokens). 0 disables. Multiplies onto --loss_weighting.
    p.add_argument("--pos_decay_gamma", type=float, default=0.0,
                   help="DFlash intra-block positional decay gamma (exp(-p/gamma)); 0 = off")
    # full_mask: mask every response token per block (gamma=1), single noisy view (no complement).
    p.add_argument("--full_mask", action="store_true",
                   help="mask 100%% of each block; single view (complementary masking disabled)")
    # hybrid linear-attention models (e.g. Qwen/Qwen3.5-9B) are auto-detected from config.layer_types
    # and routed to HFBlockDiffusionHybrid. This flag selects how the linear (gated-delta) layers
    # treat a diffusion block: "causal" (first cut — plain left-to-right scan; intra-block
    # bidirectionality comes only from the full-attention layers) or "bidirectional" (research-level
    # block-local-bidirectional gated-delta scan; needs GPU + FLA-kernel work, see docs). Ignored for
    # dense (all-full_attention) models like Qwen3-4B.
    p.add_argument("--linear_block_mode", choices=["causal", "bidirectional"], default="causal",
                   help="hybrid models only: gated-delta block handling (causal | bidirectional)")
    p.add_argument("--allow_slow_linear", action="store_true",
                   help="hybrid models: allow the torch fallback for linear layers when fla/causal_conv1d absent")
    p.add_argument("--micro_batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    # multi-turn: supervise EVERY assistant turn of a conversation in one sequence (causally ordered)
    p.add_argument("--multiturn", action="store_true",
                   help="supervise every assistant turn (vs only the last); combine with --pack for packed multi-turn")
    # keep non-last-turn <think> reasoning (needs the patched local Qwen3 template, e.g.
    # MODEL_ID=Qwen/Qwen3-4B); off = stock behavior (history reasoning stripped)
    p.add_argument("--keep_all_reasoning", action="store_true",
                   help="render/supervise reasoning on all assistant turns (interleaved), not just the last")
    # sequence packing (segment-isolated / document-masked; multiple samples per row)
    p.add_argument("--pack", action="store_true", help="pack multiple samples per row (doc-masked)")
    p.add_argument("--activation_offload", action="store_true",
                   help="offload checkpointed layer activations to CPU (fits long context, e.g. 32k)")
    p.add_argument("--fused_ce", action="store_true",
                   help="use Liger fused_linear_cross_entropy for the loss (no [N,vocab] logits)")
    p.add_argument("--fsdp_keep_params", action="store_true",
                   help="FSDP2 reshard_after_forward=False on the decoder layers: keep the all-gathered "
                        "bf16 params resident between forward and backward (and the grad-checkpoint "
                        "recompute) -> 1 all-gather per layer per micro-step instead of 3. Costs the "
                        "unsharded param size (~8GB for 4B) of GPU memory.")
    p.add_argument("--compile_mlp", action="store_true",
                   help="torch.compile each layer's MLP + RMSNorms only (shape-stable glue; no per-batch recompiles)")
    p.add_argument("--compile_glue", action="store_true",
                   help="--compile_mlp plus the GatedDeltaNet layers (gating/l2norm/norm-gate/projections) with the "
                        "recurrence+conv kernel calls dynamo-disabled; attention layers untouched")
    p.add_argument("--compile_layers", action="store_true",
                   help="torch.compile each decoder layer (per-layer, before FSDP shard)")
    p.add_argument("--merged_views", action="store_true",
                   help="store the shared prompt once (merged complementary views) instead of "
                        "duplicating it across a 2B batch; lossless, cuts forward compute + memory")
    p.add_argument("--within_block_causal", action="store_true",
                   help="make x_t within-block attention token-causal instead of bidirectional; "
                        "with x0-causal (ar_loss_weight>0) + linear_block_mode=causal the model is "
                        "globally causal (vLLM-servable). Default off = original BD behavior.")
    p.add_argument("--pack_examples", type=int, default=16,
                   help="raw samples pulled per packed step (DataLoader batch_size when --pack)")
    p.add_argument("--max_segments", type=int, default=0, help="max segments per packed row (0=unlimited)")
    p.add_argument("--max_packed_rows", type=int, default=0, help="OOM-safety cap on rows/step (0=off)")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--start_step", type=int, default=0,
                   help="initial global_step (warm-start continuation): continues the cosine LR "
                        "schedule and checkpoint numbering from this step. Weights only — optimizer "
                        "moments and data position still reset (checkpoints carry no optimizer state).")
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=50)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--save_dir", default="checkpoints/hf_block_diffusion")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", default=None,
                   help="W&B project; if unset, wandb falls back to $WANDB_PROJECT / its default")
    p.add_argument("--run_name", default=None)
    return p
