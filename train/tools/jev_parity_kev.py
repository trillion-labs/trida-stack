"""kev venv: (1) encode N real training records (plain + augmented with a fixed seed) with kev's own code;
(2) build a tiny random Qwen3.5 model, run kev's DecisionModel on a few records, dump probabilities.

    KEV_DIR=/path/to/kev python train/tools/jev_parity_kev.py <out_dir> <tokenizer_dir>
"""
import json, os, random, sys, torch
import transformers.utils.import_utils as iu   # CPU parity: force the torch reference GDN / conv kernels
iu.is_causal_conv1d_available = lambda: False; iu.is_flash_linear_attention_available = lambda: False
KEV_DIR = os.environ["KEV_DIR"]   # a kev checkout (https://github.com/jaredpalmer/kev)
sys.path.insert(0, KEV_DIR)
from transformers import AutoTokenizer
from kev.data import augment, materialize, none_pair
from kev.model import encode, rows_of, DecisionModel
S = sys.argv[1]; TOK = sys.argv[2]
tok = AutoTokenizer.from_pretrained(TOK)
recs = []
for f in ["evals/v7/decision-v7/train.jsonl", "evals/night2/dates_unknowable.jsonl", "evals/round15/joint/train.jsonl"]:
    lines = open(os.path.join(KEV_DIR, f)).readlines()
    recs += [json.loads(l) for l in random.Random(0).sample(lines, 40)]
out = []
for i, r in enumerate(recs):
    rng = random.Random(1000 + i)
    for variant in [r, augment(r, rng)] + none_pair(r, rng):
        enc = encode(tok, materialize(variant), max_state=7552, max_branch=1024 + 7552 - 384)
        s_ids, _, rows = rows_of(enc)
        out.append([{"ids": s_ids + row["ids"], "decide": len(s_ids) + row["decide"], "opts": [len(s_ids) + o for o in row["opts"]]} for row in rows])
json.dump({"records": recs, "rows": out}, open(f"{S}/kev_encode.json", "w"))
# tiny model
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
torch.manual_seed(0)
cfg = Qwen3_5TextConfig(hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
                        head_dim=16, vocab_size=len(tok), layer_types=["linear_attention"] * 3 + ["full_attention"],
                        linear_num_key_heads=2, linear_num_value_heads=4, linear_key_head_dim=16, linear_value_head_dim=16)
m = Qwen3_5ForCausalLM(cfg).to(torch.float32)
m.save_pretrained(f"{S}/tiny"); tok.save_pretrained(f"{S}/tiny")
dm = DecisionModel(f"{S}/tiny", tok, "cpu", lora=None, dtype=torch.float32).eval()
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq
import torch.nn.functional as F
def _ref_conv(x, weight, bias=None, activation=None, **_):   # CPU reference causal depthwise conv (single row, no packing)
    o = F.conv1d(x, weight.unsqueeze(1), bias, padding=weight.shape[-1] - 1, groups=x.shape[1])[..., : x.shape[-1]]
    return F.silu(o) if activation in ("silu", "swish") else o
mq.causal_conv1d_fn = _ref_conv
mq.torch_chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule.__wrapped__   # undecorated torch reference (no fla dispatch)
for layer in dm.lm.layers:   # CPU: the torch reference kernels
    if getattr(layer, "layer_type", "") == "linear_attention":
        layer.linear_attn.causal_conv1d_fn = None
        layer.linear_attn.chunk_gated_delta_rule = mq.torch_chunk_gated_delta_rule
torch.save(dm.head.state_dict(), f"{S}/tiny_head.pt")
probs = []
with torch.no_grad():
    for r in recs[:6]:
        enc = encode(tok, materialize(r), max_state=7552, max_branch=1024 + 7552 - 384)
        probs.append([p.tolist() for p in dm.probs(enc)])
json.dump({"records": recs[:6], "probs": probs}, open(f"{S}/kev_probs.json", "w"))
print("kev side done:", len(out), "encoded variants;", len(probs), "records scored")
