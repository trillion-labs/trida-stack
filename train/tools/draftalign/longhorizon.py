"""Long-horizon determinacy probe (first-principles test, 2026-09-12).

QUESTION: is a meaningful amount of FAR-future text already determined by the context?
Speculative decoding inherits from AR the rule that drafts must be the next few CONTIGUOUS tokens.
A diffusion model can predict ANY masked position. If far positions exist that the model is both
confident about AND right about, they could be committed out of order -- tokens for free, because at
batch 1 a forward costs one weight load regardless of how many positions it resolves.

METHOD: teacher-force the model's own greedy text; mask a long block (bd, default 32) after a clean
seed; read the shifted noisy-stream logits; record per masked offset k the top-1 probability and
whether top-1 == the true token. Report accuracy by offset, and -- the decisive number -- among FAR
offsets, what fraction clear a confidence bar and how accurate those are.

usage: python -m train.tools.draftalign.longhorizon --ckpt DIR --details gsm8k_details.json [--bd 32]
"""
import argparse, json, collections
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--details", required=True)
ap.add_argument("--bd", type=int, default=32, help="block size = horizon; slot 0 is the clean seed")
ap.add_argument("--limit", type=int, default=30)
ap.add_argument("--near", type=int, default=3, help="offsets <= near are the 'near' bucket (what we draft today)")
ap.add_argument("--out", default="")
ap.add_argument("--tag", default="")
a = ap.parse_args()

from train.hf_block_diffusion_hybrid import HFBlockDiffusionHybrid

dev = "cuda"
rows = json.load(open(a.details))[: a.limit]
bd = a.bd

model = HFBlockDiffusionHybrid(
    model_id=a.ckpt, bd_size=bd, ar_loss_weight=0.0, mask_pattern="canvas",
    linear_block_mode="bidirectional", within_block_causal=False, max_length=32768,
    max_response_length=32768, response_buckets=(32768,), dtype=torch.bfloat16,
    trust_remote_code=True, require_kernels=True).to(dev).eval()
tok = model.tokenizer


def fixed_views(input_ids, labels, answer_pos, resp_block=None, _bd=bd):
    """Cold canvas at every block: slot 0 clean, slots 1..bd-1 MASK."""
    B, L = input_ids.shape
    off = torch.arange(L, device=input_ids.device)[None, :].expand(B, -1) % _bd
    mask = (off >= 1) & answer_pos
    noisy = torch.where(mask, model.mask_id, input_ids)
    vlab = labels.clone(); vlab[~mask] = -100
    return noisy, vlab, torch.ones(B, L, device=input_ids.device)


model._make_canvas_views = fixed_views
acc = collections.defaultdict(lambda: [0, 0])            # offset -> [correct, total]
conf_hist = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))  # offset-bucket -> conf-bucket -> [correct,total]
CONF_EDGES = [0.0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01]


def cbucket(p):
    for i in range(len(CONF_EDGES) - 1):
        if CONF_EDGES[i] <= p < CONF_EDGES[i + 1]:
            return f"{CONF_EDGES[i]:.2f}-{CONF_EDGES[i+1]:.2f}"
    return "1.00"


for r in rows:
    pt = tok.apply_chat_template([{"role": "user", "content": r["question"]}], add_generation_prompt=True, tokenize=False)
    p_ids = tok(pt, add_special_tokens=False)["input_ids"]
    g_ids = tok(r["generation"], add_special_tokens=False)["input_ids"]
    ids = torch.tensor(list(p_ids) + list(g_ids)); lab = torch.tensor([-100] * len(p_ids) + list(g_ids))
    plen = len(p_ids)
    pad = (-plen) % bd
    if pad:
        ids = torch.cat([torch.full((pad,), tok.pad_token_id or 0), ids]); lab = torch.cat([torch.full((pad,), -100), lab]); plen += pad
    rp = (-len(ids)) % bd
    if rp:
        ids = torch.cat([ids, torch.full((rp,), tok.pad_token_id or 0)]); lab = torch.cat([lab, torch.full((rp,), -100)])
    am = torch.ones_like(ids); am[len(ids) - rp:] = 0; am[:pad] = 0
    ids, lab, am = ids[None].to(dev), lab[None].to(dev), am[None].to(dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits, _, _ = model.forward_flare(ids, lab, am, return_logits=True)
    L = ids.shape[1]
    lg = logits[0, :-1].float()                                   # position t predicts token t+1
    probs = torch.softmax(lg, dim=-1)
    top_p, top_i = probs.max(dim=-1)
    truth = ids[0, 1:]
    off = (torch.arange(L, device=dev) % bd)[1:]                  # offset of the PREDICTED token
    sel = (lab[0, 1:] != -100) & (off >= 1)
    for k, c, pr in zip(off[sel].tolist(), (top_i[sel] == truth[sel]).tolist(), top_p[sel].tolist()):
        acc[k][0] += int(c); acc[k][1] += 1
        ob = "near" if k <= a.near else ("mid" if k <= 8 else "far")
        d = conf_hist[ob][cbucket(pr)]
        d[0] += int(c); d[1] += 1

print(f"[{a.tag}] accuracy by offset (offset = distance from the clean seed):")
for k in sorted(acc):
    c, n = acc[k]
    print(f"  k={k:2d}  acc {c/n:.3f}  n={n}")
print(f"[{a.tag}] confidence vs correctness:")
for ob in ("near", "mid", "far"):
    tot = sum(v[1] for v in conf_hist[ob].values()) or 1
    print(f"  --- {ob} ---")
    for cb in sorted(conf_hist[ob]):
        c, n = conf_hist[ob][cb]
        print(f"    conf {cb}: {100*n/tot:5.1f}% of positions, acc {c/n:.3f} (n={n})")
if a.out:
    json.dump({"ckpt": a.ckpt, "bd": bd,
               "acc_by_offset": {str(k): acc[k] for k in acc},
               "conf": {ob: {cb: conf_hist[ob][cb] for cb in conf_hist[ob]} for ob in conf_hist}},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)
