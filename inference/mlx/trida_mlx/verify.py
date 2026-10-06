"""First-contact check for a real checkpoint on the Mac (run this before benchmarking).

    python -m trida_mlx.verify --model ./Trida2.0-4B-mlx-q8

1. prints the resolved config (layers, GDN/attention split, mask id, eos ids);
2. canvas vs AR logits: rows 0..N-1 of a verify canvas must equal step-by-step AR logits
   (max |diff| should be at bf16 noise level, argmax identical);
3. greedy self-spec vs greedy AR for --max-tokens tokens. In bf16 the canvas (7-row matmuls)
   and the AR step (1-row matmuls) use different kernels, so an exact tie-break can flip at a
   near-tie; if outputs diverge, the AR top-1/top-2 logit margin at that position is printed
   (a tiny margin = numerics, a large margin = a real bug).
"""
from __future__ import annotations

import argparse
import time

import mlx.core as mx

from .decode import DecodeStats, SamplingParams, ar_generate, selfspec_generate
from .engine import Engine

PROMPT = [{"role": "user", "content": "Explain in a short paragraph why the sky is blue, then give a one-line summary."}]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="trillionlabs/Trida2.0-4B")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gen-block", type=int, default=None, help="N (canvas 2N-1); default 4")
    ap.add_argument("--image", default=None, help="also run the checks on an image prompt (path or URL)")
    a = ap.parse_args(argv)
    eng = Engine(a.model, gen_block=a.gen_block, prompt_cache=False)
    rt, n = eng.rt, eng.n
    args = rt.args
    n_lin = sum(l.is_linear for l in rt.backbone.layers)
    print(f"model {a.model}: {args.num_hidden_layers} layers ({n_lin} gated-delta, {args.num_hidden_layers - n_lin} "
          f"full-attn), hidden {args.hidden_size}, vocab {args.vocab_size}, tied={args.tie_word_embeddings}, "
          f"quant={eng.cfg.get('quantization')}")
    print(f"mask_id={eng.mask_id} ({eng.tokenizer.convert_ids_to_tokens(eng.mask_id)!r})  eos={sorted(eng.eos_ids)}  "
          f"load {eng.load_s:.1f}s  device={mx.default_device()}  N={n} (canvas {2 * n - 1})  "
          f"vision={'yes' if eng.supports_vision else 'no'}")

    if a.image:
        if not eng.supports_vision:
            raise SystemExit("--image given but this checkpoint has no vision encoder")
        t = time.perf_counter()
        key, n_img = eng.add_image(a.image)
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": "Describe this image in two sentences."}]}]
        prompt = eng.encode(eng.render(msgs, enable_thinking=False), [(key, n_img)])
        print(f"[image] {a.image}: grid {eng.rt.images[key]['grid']} -> {n_img} tokens "
              f"(preprocess {time.perf_counter() - t:.2f}s)")
    else:
        prompt = eng.encode(eng.render(PROMPT, enable_thinking=False))
    sp = SamplingParams(temperature=0.0)

    # 1) AR reference with margins
    c = rt.make_cache()
    logits = rt.prefill(c, prompt)
    ar, margins, hit_eos = [], [], False
    t = time.perf_counter()
    for _ in range(a.max_tokens):
        top2 = mx.topk(logits, 2)
        tok = int(mx.argmax(logits).item())
        margins.append(float((mx.max(top2) - mx.min(top2)).item()))
        if tok in eng.eos_ids:
            hit_eos = True
            break
        ar.append(tok)
        logits = rt.ar_step(c, tok)
    ar_s = time.perf_counter() - t

    # 2) canvas rows vs AR logits on the first N generated tokens
    c1 = rt.make_cache(); rt.prefill(c1, prompt)
    L = rt.canvas(c1, ar[:n] + [eng.mask_id] * (n - 1), n)
    c2 = rt.make_cache(); rt.prefill(c2, prompt)
    worst, same = 0.0, True
    for i in range(n):
        ref = rt.ar_step(c2, ar[i])
        worst = max(worst, float(mx.max(mx.abs(L[i] - ref)).item()))
        same &= int(mx.argmax(L[i]).item()) == int(mx.argmax(ref).item())
    print(f"[canvas==AR] rows 0..{n - 1}: max|dlogit|={worst:.4f}  argmax identical={same}")

    # 3) greedy self-spec vs AR
    st = DecodeStats()
    t = time.perf_counter()
    ss = [x for ch in selfspec_generate(rt, rt.make_cache(), prompt, max_new_tokens=len(ar) + (1 if hit_eos else 0),
                                        eos_ids=eng.eos_ids, sp=sp, mask_id=eng.mask_id, n=n, stats=st) for x in ch]
    ss_s = time.perf_counter() - t
    k = next((i for i in range(min(len(ar), len(ss))) if ar[i] != ss[i]), None)
    if k is None and len(ar) == len(ss):
        print(f"[lossless] YES: {len(ar)} tokens identical")
    else:
        k = k if k is not None else min(len(ar), len(ss))
        print(f"[lossless] diverged at token {k}/{len(ar)}; AR top1-top2 margin there = "
              f"{margins[k] if k < len(margins) else float('nan'):.4f} (median margin {sorted(margins)[len(margins) // 2]:.3f})")
    print(f"[speed] AR {len(ar) / ar_s:.1f} tok/s | self-spec {len(ss) / ss_s:.1f} tok/s "
          f"({st.tokens_per_forward:.2f} tok/fwd, accept hist {st.accept_hist}, cold {st.cold_forwards})  "
          f"peak mem {mx.get_peak_memory() / 1e9:.2f} GB")
    print("---\n" + eng.tokenizer.decode(ss)[:600])


if __name__ == "__main__":
    main()
