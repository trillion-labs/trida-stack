#!/usr/bin/env python3
"""Attach the Qwen3.5 vision tower to a text-only block-diffusion checkpoint.

Checkpoints trained from a text-only backbone are flat ``Qwen3_5ForCausalLM`` (``model.*`` +
``lm_head``), without the vision encoder and projector. This writes a
``Qwen3_5ForConditionalGeneration`` checkpoint the trainer loads for image+text training:

    model.*            (text ckpt)  -> model.language_model.*
    lm_head.weight     (text ckpt)  -> kept only if it is not tied to the embedding
    model.visual.*     (base VL)    -> copied as-is: encoder + projector (merger)
    mtp.*              (base VL)    -> dropped (the trainer has no MTP head)

Config = base VL config with ``text_config`` taken from the text checkpoint (so the grown vocab,
248078 with ``<|mask|>``, carries over). Tokenizer, the patched ``chat_template.jinja``,
``block_diffusion.json`` and the image processor configs are copied alongside.

    python train/tools/compose_vl_checkpoint.py \
        --text <text_ckpt> --vision /path/to/Qwen3.5-4B --out <text_ckpt>-vl
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TEXT_ONLY_KEYS = ("architectures", "transformers_version", "bos_token_id", "pad_token_id")
COPY_FROM_TEXT = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "block_diffusion.json",
                  "generation_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
COPY_FROM_VISION = ("preprocessor_config.json", "video_preprocessor_config.json")
SHARD_BYTES = 5 * 1024**3


def read_tensors(ckpt_dir, keep=lambda k: True):
    out = {}
    for f in sorted(glob.glob(os.path.join(ckpt_dir, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if keep(k):
                    out[k] = s.get_tensor(k)
    return out


def save_sharded(tensors, out_dir):
    shards, cur, size = [], {}, 0
    for k in sorted(tensors):
        n = tensors[k].numel() * tensors[k].element_size()
        if cur and size + n > SHARD_BYTES:
            shards.append(cur)
            cur, size = {}, 0
        cur[k] = tensors[k].contiguous()
        size += n
    shards.append(cur)
    weight_map = {}
    for i, shard in enumerate(shards, 1):
        name = f"model-{i:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, os.path.join(out_dir, name), metadata={"format": "pt"})
        weight_map.update({k: name for k in shard})
    total = sum(t.numel() * t.element_size() for t in tensors.values())
    json.dump({"metadata": {"total_size": total}, "weight_map": weight_map},
              open(os.path.join(out_dir, "model.safetensors.index.json"), "w"), indent=2)


def compose(text_dir, vision_dir, out_dir):
    text_cfg = json.load(open(os.path.join(text_dir, "config.json")))
    if "vision_config" in text_cfg:
        raise SystemExit(f"{text_dir} already has a vision_config")
    vl_cfg = json.load(open(os.path.join(vision_dir, "config.json")))
    if "vision_config" not in vl_cfg:
        raise SystemExit(f"{vision_dir} has no vision_config")

    text = read_tensors(text_dir)
    lm_head = text.pop("lm_head.weight", None)
    tensors = {}
    for k, v in text.items():
        if not k.startswith("model."):
            raise SystemExit(f"unexpected text key {k}")
        tensors["model.language_model." + k[len("model."):]] = v
    embed = tensors["model.language_model.embed_tokens.weight"]
    tied = lm_head is None or torch.equal(lm_head, embed)
    if not tied:
        tensors["lm_head.weight"] = lm_head
    # The whole tower: encoder (patch_embed, pos_embed, blocks) AND the projector (merger) that maps
    # vision features into the text hidden size. Both are required.
    visual = read_tensors(vision_dir, keep=lambda k: k.startswith("model.visual."))
    for part in ("patch_embed", "blocks", "merger"):
        if not any(k.startswith(f"model.visual.{part}.") for k in visual):
            raise SystemExit(f"no model.visual.{part}.* tensors in {vision_dir}")
    tensors.update(visual)

    tc = {k: v for k, v in text_cfg.items() if k not in TEXT_ONLY_KEYS}
    tc["tie_word_embeddings"] = tied
    vl_cfg["text_config"] = tc
    vl_cfg["tie_word_embeddings"] = tied
    vl_cfg["architectures"] = ["Qwen3_5ForConditionalGeneration"]
    vl_cfg["vision_config"]["out_hidden_size"] = tc["hidden_size"]
    for k in ("bos_token_id", "pad_token_id"):
        if k in text_cfg:
            vl_cfg[k] = text_cfg[k]

    os.makedirs(out_dir, exist_ok=True)
    save_sharded(tensors, out_dir)
    json.dump(vl_cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    for name in COPY_FROM_TEXT:
        if os.path.exists(os.path.join(text_dir, name)):
            shutil.copy2(os.path.join(text_dir, name), out_dir)
    for name in COPY_FROM_VISION:
        if os.path.exists(os.path.join(vision_dir, name)):
            shutil.copy2(os.path.join(vision_dir, name), out_dir)
    print(f"wrote {out_dir}: {len(text)} text + {len(visual)} vision tensors, "
          f"lm_head {'tied' if tied else 'untied'}, vocab {tc['vocab_size']}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--text", required=True, help="flat Qwen3_5ForCausalLM checkpoint dir")
    ap.add_argument("--vision", required=True, help="base Qwen3.5 VL checkpoint dir (source of model.visual.*)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    compose(a.text, a.vision, a.out)
