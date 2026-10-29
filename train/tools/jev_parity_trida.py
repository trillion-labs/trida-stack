"""trida venv: compare our vendored encoder and forward_jev with kev's dumps."""
import json, os, random, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # repo root
from transformers.models.qwen3_5 import modeling_qwen3_5 as mq     # CPU: torch reference GDN / conv kernels
mq.chunk_gated_delta_rule = None; mq.causal_conv1d_fn = None; mq.causal_conv1d_update = None; mq.FusedRMSNormGated = None
from transformers import AutoTokenizer
from train.jev import augment, materialize, none_pair, encode_rows, training_context, pack_jev, jev_loss
S = sys.argv[1]
K = json.load(open(f"{S}/kev_encode.json"))
tok = AutoTokenizer.from_pretrained(f"{S}/tiny")
ours = []
for i, r in enumerate(K["records"]):
    rng = random.Random(1000 + i)
    for v in [r, augment(r, rng)] + none_pair(r, rng):
        ours.append(encode_rows(tok, materialize(v), **training_context(7552)))
same = sum(a == [{k: row[k] for k in ("ids", "decide", "opts")} for row in b] for a, b in zip(K["rows"], ours))
print(f"[1] encoding: {same}/{len(K['rows'])} variants identical to kev (counts {len(K['rows'])} vs {len(ours)})")

from train.hf_block_diffusion_hybrid import HFBlockDiffusionHybrid
torch.manual_seed(0)
def build(gc, attn_impl="flex_attention"):
    m = HFBlockDiffusionHybrid(model_id=f"{S}/tiny", bd_size=8, dtype=torch.float32, ar_loss_weight=0.1, full_mask=True,
                               grad_checkpoint=gc, linear_block_mode="bidirectional", require_kernels=False, jev_head_dim=256,
                               attn_impl=attn_impl)
    m.jev_head.load_state_dict(torch.load(f"{S}/tiny_head.pt"))
    return m
m = build(False).eval()
P = json.load(open(f"{S}/kev_probs.json"))
rows = [encode_rows(tok, materialize(r), **training_context(7552)) for r in P["records"]]
def probs(batch):
    with torch.no_grad():
        h = None
        loss, logs = m.forward_jev(batch)
    return logs
worst = 0.0
def z_of(batch):   # rerun the head to get logits
    from train.hf_block_diffusion_hybrid import _cu_seqlens_from_seg
    return None
for single in (True, False):
    batches = [pack_jev([r], tok) for r in rows] if single else [pack_jev(rows, tok)]
    got = []
    for b in batches:
        with torch.no_grad():
            m._jev_capture = []
            orig = m.jev_head.forward
            m.jev_head.forward = lambda d, o, w, f=orig: (m._jev_capture.append(f(d, o, w)), m._jev_capture[-1])[1]
            m.forward_jev(b)
            m.jev_head.forward = orig
            z = m._jev_capture[0]
        got += [torch.softmax(z[q["start"]:q["start"] + q["k"]], -1).tolist() for q in b["questions"]]
    want = [p for rec in P["probs"] for p in rec]
    err = max(abs(x - y) for g, w in zip(got, want) for x, y in zip(g, w))
    print(f"[2] forward parity vs kev ({'one record per batch' if single else 'all 6 records packed in one batch'}): "
          f"{len(got)} questions, max |p - p_kev| = {err:.2e}")

# [3] gradient checkpointing: recompute must use the plain kernels too (grads equal with gc on / off)
b = pack_jev(rows, tok)
grads = []
for gc in (False, True):
    torch.manual_seed(0); mm = build(gc, attn_impl="sdpa").train()   # dense mask: FlexAttention has no CPU backward
    mm.gc_min_len = 0
    loss, _ = mm.forward_jev(b); loss.backward()
    grads.append({n: p.grad.detach().clone() for n, p in mm.named_parameters() if p.grad is not None})
diff = max((grads[0][n] - grads[1][n]).abs().max().item() for n in grads[0])
print(f"[3] grad checkpointing: {len(grads[0])} params with grads, max |g_gc - g_nogc| = {diff:.2e}")
