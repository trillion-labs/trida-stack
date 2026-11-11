"""Build text and Decision training streams for this rank."""

from dataclasses import dataclass
from collections.abc import Iterator
import torch.distributed as dist
from torch.utils.data import DataLoader
from train.cli import parse_int_list
from train.distributed import is_master, log, rank, world_size
from train.models.hybrid import HFBlockDiffusionHybrid
from train.data.decision import DecisionRowDataset, load_decision_records
from train.data.text_sft_data import (
    JsonlAgenticDataset, MultiTurnTextSftCollator, PackedMultiTurnRowDataset,
    PackedRowDataset, StreamPackedCollator, StreamPackedMultiTurnCollator,
    TextSftCollator, download_jsonl_paths,
)


@dataclass
class TrainingLoaders:
    text: DataLoader | None
    decision: Iterator | None  # Already started, preserving worker initialization before the optimizer.



def _identity(x):
    return x


def build_data_loaders(args, model):
    """Build this rank's text loader and optional running Decision iterator."""
    tokenizer = model.tokenizer
    image_processor = getattr(model, "image_processor", None)
    hybrid = isinstance(model, HFBlockDiffusionHybrid)
    # ---- data (streaming JSONL; sharded per rank x worker, no Arrow) ----
    # Every rank resolves its OWN local paths (don't broadcast rank-0's paths — on a multi-node run
    # without a shared filesystem those wouldn't exist on other nodes). rank 0 downloads first so it
    # populates the (shared or per-node) HF cache; other ranks then hit the cache / download locally.
    if args.decision_only and not args.decision_data:
        raise ValueError("--decision_only needs --decision_data")
    if not args.decision_only and not args.dataset:
        raise ValueError("--dataset is required (unless --decision_only)")
    if args.decision_data and args.decision_micro_per_step < 1:
        raise ValueError("--decision_micro_per_step must be >= 1 with --decision_data (its last micro-batch syncs gradients)")
    if not args.decision_only:
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
    decision_iter = None
    if args.decision_data:
        if not (hybrid and args.linear_block_mode == "bidirectional"):
            raise ValueError("--decision_data needs the hybrid FLARE two-stream path (--linear_block_mode bidirectional)")
        records = load_decision_records(args.decision_data.split(","))
        decision_ds = DecisionRowDataset(records, tokenizer, rank=rank(), world=world_size(), seed=args.seed,
                               max_tokens=args.decision_max_tokens, max_state=args.decision_max_state,
                               max_variant_tokens=args.decision_max_variant_tokens,
                               p_none_pair=args.decision_p_none_pair)
        decision_iter = iter(DataLoader(decision_ds, batch_size=None, num_workers=2, collate_fn=_identity))
        sched = "Decision only" if args.decision_only else f"after {args.grad_accum} SFT"
        log(f"[decision] {len(records)} records | weight {args.decision_loss_weight} | {args.decision_micro_per_step} Decision "
            f"micro-batch(es) per step, {sched} | {args.decision_max_tokens} tokens each")
    loader = None
    if not args.decision_only:
        loader = DataLoader(dataset, batch_size=batch_size,
                            num_workers=args.num_workers, collate_fn=collate, drop_last=True)
        log(f"[data] packing={'on' if args.pack else 'off'} multiturn={'on' if args.multiturn else 'off'} "
            f"collator={type(collate).__name__} batch_size={batch_size}")

    return TrainingLoaders(text=loader, decision=decision_iter)
