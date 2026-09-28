#!/usr/bin/env python3
"""Shape a raw two-stream Qwen3.5 checkpoint into the form SGLang can serve.

The SGLang diffusion/self-spec decode paths load a checkpoint through the
``Qwen3_5DLLMForConditionalGeneration`` architecture (the text backbone nested
inside a vision-language wrapper — the vision tower is instantiated but never
used for text). A *raw* training checkpoint instead ships a flat
``Qwen3_5ForCausalLM`` / ``model_type: qwen3_5_text`` config, which SGLang has no
implementation for — serving it fails at load (``causal``) or at the first
decode block (``diffusion``: "missing the decode seed").

This tool produces a lightweight *shim* directory: a rewritten ``config.json``
plus symlinks (or copies) of the weights and tokenizer. The weights themselves
are unchanged — only the config that tells SGLang how to load them is swapped.
vLLM does not need this step (its plugin registers the raw arch directly).

Usage
-----
    python shape_for_sglang.py <raw_ckpt_dir> [out_dir] [--copy]

``out_dir`` defaults to ``<raw_ckpt_dir>_sglang``. Serve the result:

    python inference/serve.py <out_dir> --mode diffusion --port 30000
"""
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# Qwen3.5 vision-language wrapper stub. The tower is built from this but carries
# no weights and is never entered for text requests (deepstack indexes empty).
# Values match the reference Qwen3.5-VL config; they only need to be internally
# consistent for a text-only serve.
VISION_CONFIG = {
    "deepstack_visual_indexes": [],
    "depth": 24,
    "hidden_act": "gelu_pytorch_tanh",
    "hidden_size": 1024,
    "in_channels": 3,
    "initializer_range": 0.02,
    "intermediate_size": 4096,
    "model_type": "qwen3_5",
    "num_heads": 16,
    "num_position_embeddings": 2304,
    "out_hidden_size": 2560,
    "patch_size": 16,
    "spatial_merge_size": 2,
    "temporal_patch_size": 2,
}
# Multimodal placeholder token ids for the Qwen3.5 vocab. Unused by text
# requests; present so the wrapper config validates.
VISION_TOKEN_IDS = {
    "image_token_id": 248056,
    "video_token_id": 248057,
    "vision_start_token_id": 248053,
    "vision_end_token_id": 248054,
}

# Files carried over verbatim (symlinked/copied) if present in the raw checkpoint.
CARRY = [
    "model.safetensors",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "generation_config.json",
    "block_diffusion.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
]


def shape_config(raw: dict) -> dict:
    """Nest a raw Qwen3_5ForCausalLM text config into the DLLM VL wrapper."""
    text_config = dict(raw)
    text_config.pop("architectures", None)
    text_config.setdefault("model_type", "qwen3_5_text")
    return {
        "architectures": ["Qwen3_5DLLMForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": text_config,
        "vision_config": VISION_CONFIG,
        "tie_word_embeddings": raw.get("tie_word_embeddings", True),
        "transformers_version": raw.get("transformers_version", "4.57.0.dev0"),
        **VISION_TOKEN_IDS,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path, help="raw checkpoint directory")
    ap.add_argument("out", type=Path, nargs="?", default=None,
                    help="output dir (default: <raw>_sglang)")
    ap.add_argument("--copy", action="store_true",
                    help="copy weights/tokenizer instead of symlinking")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing output directory")
    args = ap.parse_args()

    raw_dir = args.raw.resolve()
    cfg_path = raw_dir / "config.json"
    if not cfg_path.is_file():
        sys.exit(f"no config.json in {raw_dir}")
    raw_cfg = json.loads(cfg_path.read_text())

    arch = raw_cfg.get("architectures", [])
    if "Qwen3_5DLLMForConditionalGeneration" in arch:
        sys.exit(f"{raw_dir} already looks shaped (arch={arch})")

    out_dir = (args.out or raw_dir.parent / f"{raw_dir.name}_sglang").resolve()
    if out_dir.exists():
        if not args.force:
            sys.exit(f"{out_dir} exists (use --force to overwrite)")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    (out_dir / "config.json").write_text(json.dumps(shape_config(raw_cfg), indent=2))

    carried = []
    for name in CARRY:
        src = raw_dir / name
        if not src.exists():
            continue
        dst = out_dir / name
        if args.copy:
            shutil.copy2(src, dst)
        else:
            os.symlink(src, dst)
        carried.append(name)

    if not any(n.endswith(".safetensors") or n.endswith(".index.json") for n in carried):
        sys.exit(f"no weights (*.safetensors) found in {raw_dir}")

    print(f"shaped: {out_dir}")
    print(f"  arch : Qwen3_5ForCausalLM -> Qwen3_5DLLMForConditionalGeneration")
    print(f"  files: config.json (rewritten) + {len(carried)} "
          f"{'copied' if args.copy else 'symlinked'} ({', '.join(carried)})")
    print(f"  serve: python inference/serve.py {out_dir} --mode diffusion --port 30000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
