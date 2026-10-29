#!/usr/bin/env python3
"""Export a Jev-trained checkpoint (train.py --jev_data) to kev's full-weight checkpoint layout, so kev's own tools
score and serve it: `kev.benchmark --run <out>`, `kev.serve --run <out>`.

    python train/tools/export_kev_checkpoint.py --ckpt <save_dir>/step_N --out <dir> [--temperature T]

Writes the text backbone (a flat Qwen3_5ForCausalLM config + `model.*` shards; a VL checkpoint's vision tower is
dropped, kev reads text only), the tokenizer, and `head.pt` in kev's Meta format (kev/checkpoint.py): weights="full",
the pointer head, head_dim, weights_dtype, temperature (1.0 = raw; fit one with kev.metrics.fit_temperature).
"""
import argparse
import glob
import json
import os
import shutil

import torch
from safetensors import safe_open
from safetensors.torch import save_file

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "special_tokens_map.json",
                   "vocab.json", "merges.txt")
DTYPES = {torch.bfloat16: ("bf16", "bfloat16"), torch.float32: ("fp32", "float32")}


def export(ckpt, out, temperature=1.0):
    cfg = json.load(open(os.path.join(ckpt, "config.json")))
    head = torch.load(os.path.join(ckpt, "jev_head.pt"), map_location="cpu")
    vl = "vision_config" in cfg
    prefix = "model.language_model." if vl else "model."
    tensors = {}
    for f in sorted(glob.glob(os.path.join(ckpt, "*.safetensors"))):
        with safe_open(f, "pt") as s:
            for k in s.keys():
                if k.startswith(prefix):
                    tensors["model." + k[len(prefix):]] = s.get_tensor(k)
    if not tensors:
        raise SystemExit(f"no text weights under {prefix!r} in {ckpt}")
    dtype = next(iter(tensors.values())).dtype
    weights_dtype, dtype_name = DTYPES[dtype]
    text_cfg = dict(cfg["text_config"]) if vl else dict(cfg)
    text_cfg.update(architectures=["Qwen3_5ForCausalLM"], model_type="qwen3_5_text", dtype=dtype_name)
    text_cfg.pop("torch_dtype", None)
    os.makedirs(out, exist_ok=True)
    save_file(tensors, os.path.join(out, "model.safetensors"), metadata={"format": "pt"})
    json.dump(text_cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
    for name in TOKENIZER_FILES:
        if os.path.exists(os.path.join(ckpt, name)):
            shutil.copy2(os.path.join(ckpt, name), out)
    meta = {"base": os.path.abspath(out), "head": {k: v.float() for k, v in head["head"].items()},
            "head_dim": head["head_dim"], "weights": "full", "weights_dtype": weights_dtype,
            "temperature": float(temperature), "lora": 0, "option_isolation": False, "special_embeddings": False,
            "holdout": [], "exported_from": os.path.abspath(ckpt)}
    torch.save(meta, os.path.join(out, "head.pt"))
    print(f"exported {ckpt} -> {out}: {len(tensors)} text tensors ({weights_dtype}), head_dim {head['head_dim']}, "
          f"T {temperature}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    a = ap.parse_args()
    export(a.ckpt, a.out, a.temperature)
