"""Build a tiny *raw-HF-format* Trida-like checkpoint (random weights, byte-level BPE
tokenizer, Qwen-style chat template with tools + thinking) for offline end-to-end tests."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mlx.core as mx
from mlx.utils import tree_flatten

from test_tiny import TINY, tiny_runtime  # noqa: E402

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>",
            "</tool_call>", "<|mask|>"]

CHAT_TEMPLATE = r"""{%- if tools %}<|im_start|>system
{%- if messages[0].role == 'system' %}
{{ messages[0].content }}
{%- endif %}
# Tools
<tools>
{%- for t in tools %}
{{ t | tojson }}
{%- endfor %}
</tools>
Call a tool with <tool_call>{"name": ..., "arguments": {...}}</tool_call><|im_end|>
{% elif messages[0].role == 'system' %}<|im_start|>system
{{ messages[0].content }}<|im_end|>
{% endif %}
{%- for m in messages %}
{%- if m.role == 'user' %}<|im_start|>user
{{ m.content }}<|im_end|>
{% elif m.role == 'assistant' %}<|im_start|>assistant
{{ m.content }}
{%- if m.tool_calls %}{% for tc in m.tool_calls %}<tool_call>{"name": "{{ tc.function.name }}", "arguments": {{ tc.function.arguments | tojson }}}</tool_call>{% endfor %}{% endif %}<|im_end|>
{% elif m.role == 'tool' %}<|im_start|>user
<tool_response>{{ m.content }}</tool_response><|im_end|>
{% endif %}
{%- endfor %}
{%- if add_generation_prompt %}<|im_start|>assistant
{%- if enable_thinking is defined and enable_thinking is false %}
<think>

</think>

{% else %}
<think>
{% endif %}
{%- endif %}"""


def build(out: Path):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast

    out.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    corpus = ["hello world the weather in seoul is sunny", "def fib(n): return n", "tool call json name",
              "안녕하세요 온디바이스"] * 20
    trainer = trainers.BpeTrainer(vocab_size=TINY["vocab_size"] - len(SPECIALS), special_tokens=[],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(corpus, trainer)
    tok.add_special_tokens(SPECIALS)
    assert tok.get_vocab_size() <= TINY["vocab_size"], tok.get_vocab_size()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    fast.chat_template = CHAT_TEMPLATE
    fast.save_pretrained(str(out))
    mask_id = fast.convert_tokens_to_ids("<|mask|>")

    rt = tiny_runtime(seed=0)
    params = dict(tree_flatten(rt.model.parameters()))
    hf = {}
    norm_sfx = (".input_layernorm.weight", ".post_attention_layernorm.weight", "model.norm.weight",
                ".q_norm.weight", ".k_norm.weight")
    for k, v in params.items():
        k2 = k.replace("language_model.model.", "model.").replace("language_model.", "")
        if "conv1d.weight" in k2:
            v = v.moveaxis(1, 2)  # MLX (C, K, 1) -> HF (C, 1, K)
        if any(k2.endswith(s) for s in norm_sfx):
            v = v - 1.0  # HF Qwen3.5 stores zero-centred RMSNorm weights
        hf[k2] = v.astype(mx.bfloat16) if not k2.endswith("A_log") else v
    hf["lm_head.weight"] = hf["model.embed_tokens.weight"]  # real ckpt ships the tied head too
    mx.save_safetensors(str(out / "model.safetensors"), hf)
    cfg = dict(TINY, architectures=["Qwen3_5ForCausalLM"])
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    (out / "block_diffusion.json").write_text(json.dumps({"bd_size": 4, "mask_id": mask_id}))
    (out / "generation_config.json").write_text(json.dumps({"eos_token_id": [fast.eos_token_id]}))
    return out, mask_id


if __name__ == "__main__":
    print(build(Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiny_trida")))
