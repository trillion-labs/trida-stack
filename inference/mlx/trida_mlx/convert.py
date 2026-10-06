"""Convert / quantize a Trida checkpoint to an MLX directory.

    python -m trida_mlx.convert --model trillionlabs/Trida2.0-4B --out ./Trida2.0-4B-mlx-q8 --bits 8
    python -m trida_mlx.convert --model trillionlabs/Trida2.0-4B --out ./Trida2.0-4B-mlx-bf16

The output keeps the tokenizer, chat template, generation config and block_diffusion.json,
and writes a config with ``model_type: qwen3_5`` + ``quantization``, so it also loads with
stock ``mlx_lm.load`` (useful as an AR baseline: ``mlx_lm.server``).

Quantization notes (from the ocr-monorepo MLX study): 8-bit and 6-bit kept greedy outputs
identical to bf16 there, 4-bit changed them. Self-spec stays lossless *relative to the
quantized AR model* at any bit width; check quality with ``bench.py --compare-ref``.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .model import _text_config, load_model

CARRY = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "generation_config.json",
         "block_diffusion.json", "special_tokens_map.json", "vocab.json", "merges.txt", "added_tokens.json",
         "preprocessor_config.json", "processor_config.json", "video_preprocessor_config.json"]
# multimodal config keys carried next to the (flattened) text config
VISION_KEYS = ["vision_config", "image_token_id", "video_token_id", "vision_start_token_id",
               "vision_end_token_id"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trillionlabs/Trida2.0-4B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bits", type=int, default=None, choices=[3, 4, 5, 6, 8], help="omit for bf16")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--keep-embed", action="store_true",
                    help="keep embed_tokens / (tied) LM head in bf16 (bigger, safer quality)")
    ap.add_argument("--shard-gb", type=float, default=4.0)
    a = ap.parse_args(argv)

    model, vision, cfg, src = load_model(a.model)
    text = _text_config(cfg)
    out_cfg = dict(text)
    out_cfg["model_type"] = "qwen3_5"
    out_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    # transformers >= 5 treats a local tokenizer whose config.json lacks transformers_version as a
    # possible old Mistral one and prints a (false) "incorrect regex pattern" warning
    out_cfg.setdefault("transformers_version", cfg.get("transformers_version", text.get("transformers_version", "5.12.1")))
    if vision is not None:  # the vision tower stays bf16 (small, and the most quant-sensitive part)
        for k in VISION_KEYS:
            if k in cfg:
                out_cfg[k] = cfg[k]
    if a.bits:
        def pred(path, m):
            if not hasattr(m, "to_quantized"):
                return False
            if a.keep_embed and ("embed_tokens" in path or "lm_head" in path):
                return False
            w = getattr(m, "weight", None)
            return w is not None and w.shape[-1] % a.group_size == 0
        nn.quantize(model, group_size=a.group_size, bits=a.bits, class_predicate=pred)
        out_cfg["quantization"] = {"group_size": a.group_size, "bits": a.bits}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    weights = dict(tree_flatten(model.parameters()))
    if vision is not None:
        weights.update({f"vision_tower.{k}": v for k, v in tree_flatten(vision.parameters())})
    shards, cur, cur_bytes = [], {}, 0
    limit = int(a.shard_gb * (1 << 30))
    for k, v in weights.items():
        if cur and cur_bytes + v.nbytes > limit:
            shards.append(cur); cur, cur_bytes = {}, 0
        cur[k] = v; cur_bytes += v.nbytes
    shards.append(cur)
    index = {"metadata": {"total_size": sum(v.nbytes for v in weights.values())}, "weight_map": {}}
    for i, sh in enumerate(shards):
        name = "model.safetensors" if len(shards) == 1 else f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        mx.save_safetensors(str(out / name), sh, metadata={"format": "mlx"})
        for k in sh:
            index["weight_map"][k] = name
    if len(shards) > 1:
        (out / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
    (out / "config.json").write_text(json.dumps(out_cfg, indent=2))
    for name in CARRY:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    gb = index["metadata"]["total_size"] / 1e9
    print(f"wrote {out}  ({gb:.2f} GB, {len(shards)} shard(s), "
          f"{'bf16' if not a.bits else f'q{a.bits} g{a.group_size}'}"
          f"{', + bf16 vision tower' if vision is not None else ''})")


if __name__ == "__main__":
    main()
