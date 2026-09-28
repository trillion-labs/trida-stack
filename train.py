#!/usr/bin/env python3
"""Block-diffusion training entry point.

Thin launcher for the HF block-diffusion trainer in `train/train.py` — a text-only port of
Trillion Labs' nanoVLM-DiffusionOCR (Fast-dLLM v2) that initializes from a pretrained HF model
(e.g. Qwen3-4B, Qwen3.5-9B, Tri-7B) via `from_pretrained` and trains on native-torch FSDP2.

Launch with torchrun (multi-GPU); NOT runnable on MPS:

  PYTHONPATH=. torchrun --nproc_per_node=8 train.py --model_id Qwen/Qwen3-4B
  # or: sbatch slurm/train_hf_block_diffusion.sbatch   (untracked/local; see .gitignore)

See `train/train.py --help` for flags; install deps from requirements.txt (torch 2.11 cu128,
transformers, datasets, accelerate).
"""
import sys

# Answer --help without importing the heavy trainer (torch/FSDP), so it works pre-install.
if any(a in ("-h", "--help") for a in sys.argv[1:]):
    from train.cli import build_parser

    build_parser().print_help()
    sys.exit(0)

from train.train import main

if __name__ == "__main__":
    main()
