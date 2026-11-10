"""Tiny Trida-like *multimodal* raw-HF checkpoint (random weights) for offline tests."""
import json, os, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mlx.core as mx
from mlx.utils import tree_flatten
from test_tiny import TINY, tiny_runtime
from trida_mlx.vision import VisionTower, VisionConfig

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>",
            "</tool_call>", "<|mask|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>", "<|video_pad|>"]
VCFG = dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, patch_size=4, temporal_patch_size=2,
            spatial_merge_size=2, in_channels=3, out_hidden_size=TINY["hidden_size"], num_position_embeddings=16,
            deepstack_visual_indexes=[], model_type="qwen3_5")

def build(out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast
    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE()); tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["hello world the weather in seoul is sunny describe this image", "what is in the picture"] * 20
    tok.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=TINY["vocab_size"] - len(SPECIALS), special_tokens=[],
                            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    tok.add_special_tokens(SPECIALS)
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = (Path(__file__).parent / "fixtures" / "trida_chat_template.jinja").read_text()
    fast.save_pretrained(str(out))
    ids = {t: fast.convert_tokens_to_ids(t) for t in SPECIALS}
    rt = tiny_runtime(seed=0)
    hf = {}
    norm_sfx = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight", ".q_norm.weight", ".k_norm.weight")
    for k, v in tree_flatten(rt.model.parameters()):
        k2 = ("model.language_model." + k[len("language_model.model."):]) if k.startswith("language_model.model.") else k[len("language_model."):]
        if "conv1d.weight" in k2: v = v.moveaxis(1, 2)
        if any(k2.endswith(s) for s in norm_sfx): v = v - 1.0
        hf[k2] = v if k2.endswith("A_log") else v.astype(mx.bfloat16)
    mx.random.seed(7)
    vt = VisionTower(VisionConfig.from_dict(VCFG))
    for k, v in tree_flatten(vt.parameters()):
        if k == "patch_embed.proj.weight": v = v.transpose(0, 4, 1, 2, 3)  # MLX NDHWC -> PyTorch NCDHW
        hf["model.visual." + k] = v.astype(mx.bfloat16)
    mx.save_safetensors(str(out / "model.safetensors"), hf)
    text = dict(TINY); text["model_type"] = "qwen3_5_text"
    cfg = {"architectures": ["Qwen3_5ForConditionalGeneration"], "model_type": "qwen3_5", "text_config": text,
           "vision_config": VCFG, "tie_word_embeddings": True, "image_token_id": ids["<|image_pad|>"],
           "video_token_id": ids["<|video_pad|>"], "vision_start_token_id": ids["<|vision_start|>"],
           "vision_end_token_id": ids["<|vision_end|>"]}
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "preprocessor_config.json").write_text(json.dumps({"patch_size": 4, "temporal_patch_size": 2, "merge_size": 2,
        "size": {"shortest_edge": 16 * 16, "longest_edge": 48 * 48}, "image_mean": [0.5] * 3, "image_std": [0.5] * 3}))
    (out / "block_diffusion.json").write_text(json.dumps({"bd_size": 4, "mask_id": ids["<|mask|>"]}))
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": [fast.eos_token_id]}))
    return out, ids

def png(seed, w=40, h=28):
    from PIL import Image
    import io
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    b = io.BytesIO(); img.save(b, "PNG"); return b.getvalue()
