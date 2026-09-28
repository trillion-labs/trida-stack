"""Offline per-slot draft diagnostic (decides "inference fidelity" vs "capacity" after the flat pilot).

Runs a checkpoint through the trainer's own two-stream forward on teacher-forced text with the EXACT self-spec
canvas shapes and measures, slot by slot, how often the noisy stream's argmax agrees with the clean stream's
argmax (= the AR-Trust acceptance criterion). Cold canvas for spec N: block 2N-1 = [1 clean, 2N-2 MASK];
warm canvas: [N clean, N-1 MASK]. Reports agreement per masked slot offset, for bidirectional (training) and
token-causal (vLLM-like) within-block attention.

usage: python -m train.tools.draftalign.slot_diag --ckpt DIR --details gsm8k_details.json [--N 4 8] [--out out.json]
"""
import argparse, json, sys, time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--details", required=True, help="gsm8k_details_*.json (question + generation) = teacher-forced text")
ap.add_argument("--N", type=int, nargs="+", default=[4, 8])
ap.add_argument("--limit", type=int, default=30)
ap.add_argument("--out", default="")
ap.add_argument("--tag", default="")
a = ap.parse_args()

from train.hf_block_diffusion_hybrid import HFBlockDiffusionHybrid

dev = "cuda"
rows = json.load(open(a.details))[: a.limit]
out = {"ckpt": a.ckpt, "tag": a.tag, "details": a.details, "n_items": len(rows), "results": {}}


def build(model, tok, q, gen):
    msgs = [{"role": "user", "content": q}]
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    gen_ids = tok(gen, add_special_tokens=False)["input_ids"]
    ids = list(prompt_ids) + list(gen_ids)
    labels = [-100] * len(prompt_ids) + list(gen_ids)
    return torch.tensor(ids), torch.tensor(labels), len(prompt_ids)


for N in a.N:
    blk = 2 * N - 1
    for attn_mode in ("bidirectional", "causal"):
        t0 = time.time()
        model = HFBlockDiffusionHybrid(
            model_id=a.ckpt, bd_size=blk, ar_loss_weight=0.0, mask_pattern="canvas",
            linear_block_mode="bidirectional", within_block_causal=(attn_mode == "causal"), max_length=32768,
            max_response_length=32768, response_buckets=(32768,), dtype=torch.bfloat16, trust_remote_code=True,
            require_kernels=True).to(dev).eval()
        tok = model.tokenizer
        for shape in ("cold", "warm"):
            m = (2 * N - 2) if shape == "cold" else (N - 1)
            # fixed-m canvas views: every response block gets exactly m suffix masks
            def fixed_views(input_ids, labels, answer_pos, resp_block=None, _m=m, _bd=blk):
                B, L = input_ids.shape
                idx = torch.arange(L, device=input_ids.device)[None, :].expand(B, -1)
                offset = idx % _bd
                mask = (offset >= (_bd - _m)) & answer_pos
                noisy = torch.where(mask, model.mask_id, input_ids)
                vlab = labels.clone(); vlab[~mask] = -100
                return noisy, vlab, torch.ones(B, L, device=input_ids.device)
            model._make_canvas_views = fixed_views
            agree = torch.zeros(m, dtype=torch.long); total = torch.zeros(m, dtype=torch.long)
            agree_true = torch.zeros(m, dtype=torch.long)
            for r in rows:
                ids, labels, plen = build(model, tok, r["question"], r["generation"])
                # block-align the RESPONSE start: pad the prompt on the left so plen % blk == 0
                pad = (-plen) % blk
                if pad:
                    ids = torch.cat([torch.full((pad,), tok.pad_token_id or 0), ids]); labels = torch.cat([torch.full((pad,), -100), labels]); plen += pad
                # right-pad to a block multiple
                rp = (-len(ids)) % blk
                if rp:
                    ids = torch.cat([ids, torch.full((rp,), tok.pad_token_id or 0)]); labels = torch.cat([labels, torch.full((rp,), -100)])
                am = torch.ones_like(ids); am[len(ids) - rp:] = 0; am[:pad] = 0
                ids, labels, am = ids[None].to(dev), labels[None].to(dev), am[None].to(dev)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    # noisy head + clean (x0) head from the SAME two-stream forward; the clean head is the
                    # AR-Trust verify reference (forward_ar is not usable with the two-stream wrappers installed)
                    logits, clean_logits, _, _ = model.forward_flare(ids, labels, am, return_logits=True, return_clean_logits=True)
                L = ids.shape[1]
                noisy_pred = logits[0, :-1].argmax(-1)          # predicts token at t+1 from position t
                clean_pred = clean_logits[0, :-1].argmax(-1)
                truth = ids[0, 1:]
                offset = (torch.arange(L, device=dev) % blk)[1:]  # offset of the PREDICTED position
                is_resp = (labels[0, 1:] != -100)
                masked = (offset >= (blk - m)) & is_resp
                slot = offset - (blk - m)                       # 0..m-1 within the masked suffix
                for j in range(m):
                    sel = masked & (slot == j)
                    total[j] += int(sel.sum()); agree[j] += int((noisy_pred[sel] == clean_pred[sel]).sum())
                    agree_true[j] += int((noisy_pred[sel] == truth[sel]).sum())
            key = f"N{N}/{attn_mode}/{shape}"
            res = {"block": blk, "m": m, "positions": total.tolist(),
                   "agree_with_clean_argmax": [round(x, 4) for x in (agree / total.clamp(min=1)).tolist()],
                   "agree_with_text": [round(x, 4) for x in (agree_true / total.clamp(min=1)).tolist()]}
            out["results"][key] = res
            print(f"[{a.tag}] {key}: slot-agree(clean) {res['agree_with_clean_argmax']}  n={total.sum().item()}", flush=True)
        del model; torch.cuda.empty_cache()
        print(f"[{a.tag}] N={N} {attn_mode} took {time.time()-t0:.0f}s", flush=True)
if a.out:
    json.dump(out, open(a.out, "w"), indent=1); print("wrote", a.out)
