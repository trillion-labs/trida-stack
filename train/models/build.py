"""Construct the training model and restore pretrained components."""

import os
import torch
from train.cli import parse_int_list
from train.data.text_sft_data import load_image_processor
from train.distributed import log
from train.models.block_diffusion import HFBlockDiffusion
from train.models.hybrid import HFBlockDiffusionHybrid, is_hybrid_model_id


def build_model(args):
    """Load the dense/hybrid model, optional Jev head, and image processor."""
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
    image_processor = None
    if model.is_vl:
        image_processor = load_image_processor(model_source)
        model.image_processor = image_processor
        log(f"[model] vision-language checkpoint: frozen vision encoder, trainable projector | "
            f"images capped at {args.max_image_pixels} px")

    if args.fp32_master:   # fp32 masters (all params, so every FSDP unit is one dtype); FSDP gathers + computes in bf16
        for p in model.parameters():
            p.data = p.data.float()
    return model
