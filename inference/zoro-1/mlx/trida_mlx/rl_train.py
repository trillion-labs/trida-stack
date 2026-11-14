"""Online-RL phase 2 PoC: train a LoRA adapter on Zoro-1 from ``trida-mlx-rewards`` turns, on Apple Silicon.

    trida-mlx-rl-train --model ~/models/Zoro-1-mlx-q8 --probe            # can we backprop at all? time + memory
    trida-mlx-rl-train --model ~/models/Zoro-1-mlx-q8 \\
        --rewards ~/zoro-rl/rewards/turn_rewards.jsonl --out ~/zoro-rl/adapters/0001

What it trains: LoRA (rank 16) on the q8 weights' attention, gated-delta and MLP projections; the base weights
stay frozen and quantized. The data are turns the base model produced (self-spec is lossless, so they are AR
samples), so the base is pi_old and the loss is PPO's clipped surrogate on the AR head:

    ratio = exp(logp - logp_base)          per reply token
    pg    = -mean(min(ratio * A, clip(ratio, 1 - 0.2, 1 + 0.28) * A)),   A = reward - mean reward
    kl    = mean(exp(logp_base - logp) - (logp_base - logp) - 1)         (k3, to the base: LoRA off)
    sft   = -mean(logp of the user's edited reply)                        (turns the user edited)

The gated-delta layers train through mlx-lm's plain-op path (``model.train()`` turns off the Metal kernel, which
has no gradient), with gradient checkpointing per decoder layer so only one layer's recurrent states are held.

Caveats of this PoC: prompts are rebuilt from the saved chats (the app's system prompt, tool schemas and the
reply's reasoning are not in them), thinking is rendered off, there is no diffusion loss yet (so check self-spec
acceptance before serving an adapter), and the evaluation is a sanity check, not the phase 2 safety gate.

Writes ``adapters.safetensors`` + ``adapter_config.json`` (``load_adapter`` puts it on a loaded model) and
``report.json``: losses, how the adapter moved log-probs on rewarded / penalised / held-out turns, and
base-vs-adapter replies on a few held-out prompts.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

from .model import load_model

LORA_KEYS = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
             "linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj",
             "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
CLIP_LO, CLIP_HI = 0.2, 0.28


# ---- LoRA -------------------------------------------------------------------------------------------------

def _lora_cls():
    from mlx_lm.tuner.lora import LoRALinear
    return LoRALinear


def add_lora(model, rank: int = 16, scale: float = 2.0, keys=LORA_KEYS) -> int:
    """Wrap the listed projections of every decoder layer in LoRA (base frozen). Returns how many."""
    LoRALinear = _lora_cls()
    model.freeze()
    n = 0
    for layer in model.language_model.model.layers:
        for key in keys:
            *path, name = key.split(".")
            parent = layer
            for p in path:
                parent = getattr(parent, p, None)
                if parent is None:
                    break
            if parent is None or not hasattr(parent, name):
                continue
            setattr(parent, name, LoRALinear.from_base(getattr(parent, name), r=rank, scale=scale))
            n += 1
    return n


def checkpoint_layers(model) -> None:
    """Gradient checkpointing per decoder layer while training (only one layer's activations, including the
    gated-delta recurrent states, are held for the backward pass); plain calls in eval mode, so cached
    generation still works."""
    cls = type(model.language_model.model.layers[0])
    if getattr(cls, "_trida_checkpointed", False):
        return
    fn = cls.__call__

    def call(self, *args, **kwargs):
        if not self.training:
            return fn(self, *args, **kwargs)

        def inner(params, *args, **kwargs):
            self.update(params)
            return fn(self, *args, **kwargs)
        return mx.checkpoint(inner)(self.trainable_parameters(), *args, **kwargs)
    cls.__call__ = call
    cls._trida_checkpointed = True


def lora_modules(model):
    LoRALinear = _lora_cls()
    return [m for _, m in model.named_modules() if isinstance(m, LoRALinear)]


def set_lora_scale(model, scale: float) -> None:
    for m in lora_modules(model):
        m.scale = scale


def save_adapter(model, out: Path, config: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(out / "adapters.safetensors"), dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps(config, indent=1))


def load_adapter(model, path) -> dict:
    """Put a saved adapter on a loaded model (the server and the evaluation use this)."""
    path = Path(path).expanduser()
    cfg = json.loads((path / "adapter_config.json").read_text())
    add_lora(model, cfg["rank"], cfg["scale"], tuple(cfg["keys"]))
    model.load_weights(list(mx.load(str(path / "adapters.safetensors")).items()), strict=False)
    mx.eval(model.parameters())
    return cfg


# ---- samples ----------------------------------------------------------------------------------------------

def _messages(prompt: list) -> list:
    """trida-mlx-rewards' prompt (datamine's OpenAI view) -> messages the chat template takes."""
    out = []
    for m in prompt:
        e = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            calls = []
            for c in m["tool_calls"]:
                try:
                    args = json.loads(c.get("arguments") or "{}")
                except (TypeError, ValueError):
                    args = {"raw": c.get("arguments")}
                calls.append({"type": "function", "function": {"name": c["name"], "arguments": args}})
            e["tool_calls"] = calls
        if m["role"] == "tool" and m.get("name"):
            e["name"] = m["name"]
        out.append(e)
    return out


def encode(tok, prompt: list, reply: str, max_len: int, max_reply: int):
    """(token ids, index of the first reply token): the prompt is cut from the left to fit."""
    text = tok.apply_chat_template(_messages(prompt), add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    p = tok.encode(text, add_special_tokens=False)
    r = tok.encode(reply, add_special_tokens=False)[:max_reply] + tok.encode("<|im_end|>", add_special_tokens=False)
    p = p[-max(16, max_len - len(r)):]
    return p + r, len(p)


def build_samples(rows: list, tok, max_len: int, max_reply: int) -> list:
    """One policy sample per rewarded turn, plus one SFT sample per edited reply."""
    rewards = [r["reward"] for r in rows if r.get("reward")]
    baseline = sum(rewards) / len(rewards) if rewards else 0.0
    out = []
    for r in rows:
        if not r.get("reply"):
            continue
        adv = r.get("reward", 0.0) - baseline
        if abs(adv) > 1e-3:
            ids, start = encode(tok, r["prompt"], r["reply"], max_len, max_reply)
            out.append({"kind": "pg", "ids": ids, "start": start, "adv": adv, "reward": r.get("reward", 0.0),
                        "turn": f'{r["chat"]}:{r["user_id"]}:{r.get("kind") or "turn"}'})
        if r.get("sft_target"):
            ids, start = encode(tok, r["prompt"], r["sft_target"], max_len, max_reply)
            out.append({"kind": "sft", "ids": ids, "start": start, "adv": 0.0, "reward": None,
                        "turn": f'{r["chat"]}:{r["user_id"]}:edit'})
    return out


# ---- the loss -----------------------------------------------------------------------------------------------

def reply_logprobs(model, ids: list, start: int) -> mx.array:
    """log p(token) of every reply token, from the AR head; only the reply positions go through the LM head."""
    lm = model.language_model
    x = mx.array(ids)[None]
    h = lm.model(x[:, :-1])[:, start - 1:]
    logits = lm.model.embed_tokens.as_linear(h) if lm.args.tie_word_embeddings else lm.lm_head(h)
    logits = logits.astype(mx.float32)
    tgt = x[:, start:]
    lp = mx.take_along_axis(logits, tgt[..., None], axis=-1)[..., 0] - mx.logsumexp(logits, axis=-1)
    return lp[0]


def sample_loss(model, s: dict, base_lp, beta: float, sft_weight: float, neg_weight: float = 1.0):
    """(loss, kl to the base per token). Negative advantages are scaled by ``neg_weight``: pushing a whole reply's
    probability down moves its mass to junk tokens, the usual way these small RL runs collapse."""
    lp = reply_logprobs(model, s["ids"], s["start"])
    if s["kind"] == "sft":
        return -sft_weight * lp.mean(), mx.array(0.0)
    ratio = mx.exp(lp - base_lp)
    a = s["adv"] * (neg_weight if s["adv"] < 0 else 1.0)
    pg = -mx.minimum(ratio * a, mx.clip(ratio, 1 - CLIP_LO, 1 + CLIP_HI) * a).mean()
    d = base_lp - lp
    kl = (mx.exp(d) - d - 1).mean()
    return pg + beta * kl, kl


# ---- probe ----------------------------------------------------------------------------------------------------

def probe(model, lengths=(256, 512, 1024)) -> dict:
    """One forward + backward per length on a random sequence: does the gradient reach a gated-delta layer's LoRA,
    how long does a step take, how much memory does it need?"""
    add_lora(model)
    checkpoint_layers(model)
    model.train()
    vocab = model.language_model.args.vocab_size
    out = {"trainable_params": sum(v.size for _, v in tree_flatten(model.trainable_parameters()))}
    gdn = next(i for i, layer in enumerate(model.language_model.model.layers) if layer.is_linear)
    for n in lengths:
        ids = [random.randrange(1000, min(vocab, 100000)) for _ in range(n)]
        s = {"kind": "sft", "ids": ids, "start": n // 2}

        def loss_fn(m, s=s):
            return sample_loss(m, s, None, 0.0, 1.0)[0]
        mx.reset_peak_memory()
        t0 = time.perf_counter()
        loss, grads = nn.value_and_grad(model, loss_fn)(model)
        mx.eval(loss, grads)
        dt = time.perf_counter() - t0
        g = dict(tree_flatten(grads))
        key = f"language_model.model.layers.{gdn}.linear_attn.in_proj_qkv.lora_b"
        gnorm = float(mx.linalg.norm(g[key]).item()) if key in g else None
        out[str(n)] = {"seconds": round(dt, 2), "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
                       "loss": round(float(loss.item()), 4), "gdn_layer": gdn, "gdn_lora_b_grad_norm": gnorm,
                       "gradient_reaches_gdn": bool(gnorm and math.isfinite(gnorm) and gnorm > 0)}
        print(json.dumps({n: out[str(n)]}), flush=True)
    return out


# ---- train + evaluate ------------------------------------------------------------------------------------------

def degenerate(text: str) -> bool:
    """A reply stuck in a loop: some 3-word sequence repeats 5+ times, or one word 6+ times in a row."""
    w = text.split()
    if len(w) < 12:
        return False
    grams: dict = {}
    for g in zip(w, w[1:], w[2:]):
        grams[g] = grams.get(g, 0) + 1
    run = best = 1
    for x, y in zip(w, w[1:]):
        run = run + 1 if x == y else 1
        best = max(best, run)
    return max(grams.values()) >= 5 or best >= 6


def logprob_means(model, samples: list) -> list:
    return [float(reply_logprobs(model, s["ids"], s["start"]).mean().item()) for s in samples]


def greedy(model, tok, prompt: list, max_tokens: int) -> str:
    from mlx_lm.models.cache import make_prompt_cache
    text = tok.apply_chat_template(_messages(prompt), add_generation_prompt=True, tokenize=False,
                                   enable_thinking=False)
    ids = tok.encode(text, add_special_tokens=False)
    cache = make_prompt_cache(model)
    end = tok.convert_tokens_to_ids("<|im_end|>")
    logits = model(mx.array(ids)[None], cache=cache)
    out = []
    for _ in range(max_tokens):
        t = int(mx.argmax(logits[0, -1]).item())
        if t == end:
            break
        out.append(t)
        logits = model(mx.array([[t]]), cache=cache)
    return tok.decode(out)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--probe", action="store_true", help="only check backprop: time and memory per length")
    ap.add_argument("--rewards", help="trida-mlx-rewards' turn_rewards.jsonl")
    ap.add_argument("--out", help="adapter folder to write")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--scale", type=float, default=2.0)
    # conservative defaults: adapter 0001 (lr 5e-5, 2 epochs, beta 0.05, full-weight negatives) collapsed into
    # repetition on 43 judge-only samples
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.2, help="KL weight to the base model")
    ap.add_argument("--neg-weight", type=float, default=0.25, help="scale of negative advantages")
    ap.add_argument("--max-kl", type=float, default=0.05,
                    help="stop early when the mean KL to the base over the last 8 steps passes this (nats/token)")
    ap.add_argument("--sft-weight", type=float, default=0.5)
    # 512 tokens: about 22 s and 12 GB per step on an 18 GB M3 Pro (1024 needs 18 GB and swaps)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--max-reply", type=int, default=256)
    ap.add_argument("--holdout", type=float, default=0.15, help="share of turns kept out of training")
    ap.add_argument("--samples", type=int, default=3, help="held-out prompts to show base vs adapter replies for")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    random.seed(a.seed)
    mx.random.seed(a.seed)
    model, _, _cfg, path = load_model(a.model, load_vision=False)
    if a.probe:
        print(json.dumps(probe(model), indent=1))
        return 0
    if not (a.rewards and a.out):
        ap.error("--rewards and --out are needed to train (or use --probe)")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(path))
    rows = [json.loads(line) for line in Path(a.rewards).expanduser().read_text().splitlines() if line.strip()]
    chats = sorted({r["chat"] for r in rows})
    random.shuffle(chats)
    held_chats = set(chats[: max(1, round(len(chats) * a.holdout))]) if len(chats) > 1 else set()
    train_rows = [r for r in rows if r["chat"] not in held_chats]
    held_rows = [r for r in rows if r["chat"] in held_chats]
    train = build_samples(train_rows, tok, a.max_len, a.max_reply)
    held = build_samples(held_rows, tok, a.max_len, a.max_reply)
    if not held and len(train) > 4:   # the held-out chats had nothing to score: hold out turns instead
        random.shuffle(train)
        k = max(1, round(len(train) * a.holdout))
        held, train = train[:k], train[k:]
        held_ids = {s["turn"].rsplit(":", 1)[0] for s in held}
        held_rows = [r for r in rows if f'{r["chat"]}:{r["user_id"]}' in held_ids]
    print(f"{len(rows)} turns -> {len(train)} training samples, {len(held)} held out", flush=True)
    if not train:
        print("nothing to train on: no rewarded or edited turns", file=sys.stderr)
        return 1

    # pi_old = the base model (LoRA's B starts at zero, so the wrapped model IS the base before any update)
    n_lora = add_lora(model, a.rank, a.scale)
    checkpoint_layers(model)
    model.train()   # the same plain-op gated-delta path the updates use, so the first ratios are exactly 1
    t0 = time.perf_counter()
    base = []
    for i, s in enumerate(train):   # one forward each, before any update: the reference the ratios and KL use
        base.append(reply_logprobs(model, s["ids"], s["start"]))
        mx.eval(base[-1])
        print(json.dumps({"base": i + 1, "of": len(train)}), flush=True)
    print(f"base log-probs: {time.perf_counter() - t0:.0f}s", flush=True)
    model.eval()
    base_held = logprob_means(model, held)

    print(json.dumps({"plan": {"samples": len(train), "epochs": a.epochs, "steps": len(train) * a.epochs}}),
          flush=True)
    model.train()
    opt = optim.Adam(learning_rate=a.lr)
    history = []
    stopped = None
    for epoch in range(a.epochs):
        order = list(range(len(train)))
        random.shuffle(order)
        for step, i in enumerate(order):
            s = train[i]

            def loss_fn(m, s=s, b=base[i]):
                return sample_loss(m, s, b, a.beta, a.sft_weight, a.neg_weight)
            t = time.perf_counter()
            (loss, kl), grads = nn.value_and_grad(model, loss_fn)(model)
            opt.update(model, grads)
            mx.eval(loss, kl, model.trainable_parameters(), opt.state)
            history.append({"epoch": epoch, "step": step, "kind": s["kind"], "adv": round(s["adv"], 3),
                            "loss": round(float(loss.item()), 5), "kl": round(float(kl.item()), 5),
                            "tokens": len(s["ids"]), "seconds": round(time.perf_counter() - t, 2)})
            print(json.dumps(history[-1]), flush=True)
            recent = [h["kl"] for h in history[-8:] if h["kind"] == "pg"]
            if len(recent) >= 4 and sum(recent) / len(recent) > a.max_kl:
                stopped = f"mean KL {sum(recent) / len(recent):.3f} > {a.max_kl} after {len(history)} steps"
                print(json.dumps({"early_stop": stopped}), flush=True)
                break
        if stopped:
            break
    model.eval()
    out = Path(a.out).expanduser()
    save_adapter(model, out, {"rank": a.rank, "scale": a.scale, "keys": list(LORA_KEYS), "base_model": str(path),
                              "rewards": str(a.rewards), "train_samples": len(train), "lr": a.lr,
                              "epochs": a.epochs, "beta": a.beta, "neg_weight": a.neg_weight,
                              "lora_modules": n_lora})

    # sanity checks: did log-probs move the way the rewards say? (not the phase 2 safety gate)
    after = logprob_means(model, train)
    moved = [{"turn": s["turn"], "kind": s["kind"], "adv": round(s["adv"], 3),
              "delta_mean_logp": round(x - float(b.mean().item()), 4)} for s, b, x in zip(train, base, after)]
    pos = [m["delta_mean_logp"] for m in moved if m["kind"] == "pg" and m["adv"] > 0]
    neg = [m["delta_mean_logp"] for m in moved if m["kind"] == "pg" and m["adv"] < 0]
    sft = [m["delta_mean_logp"] for m in moved if m["kind"] == "sft"]
    held_after = logprob_means(model, held)

    def avg(xs):
        return round(sum(xs) / len(xs), 4) if xs else None
    examples = []
    for r in (held_rows or train_rows)[: a.samples]:
        set_lora_scale(model, 0.0)
        b = greedy(model, tok, r["prompt"], 200)
        set_lora_scale(model, a.scale)
        ad = greedy(model, tok, r["prompt"], 200)
        examples.append({"user": r["user"][:300], "base": b, "adapter": ad, "adapter_degenerate": degenerate(ad),
                         "base_degenerate": degenerate(b), "logged_reply": r["reply"][:300],
                         "logged_reward": r.get("reward")})
    broken = sum(e["adapter_degenerate"] and not e["base_degenerate"] for e in examples)
    report = {
        "train_samples": len(train), "held_out_samples": len(held),
        "mean_delta_logp": {"rewarded": avg(pos), "penalised": avg(neg), "edited_targets": avg(sft),
                            "held_out": avg([x - y for x, y in zip(held_after, base_held)])},
        "expect": "rewarded > 0 > penalised, edited targets > 0, held-out close to 0",
        "seconds_per_step": avg([h["seconds"] for h in history]),
        "mean_kl": avg([h["kl"] for h in history if h["kind"] == "pg"]),
        "early_stop": stopped,
        "sanity": "failed: the adapter loops where the base does not" if broken else "passed",
        "peak_gb": round(mx.get_peak_memory() / 2**30, 2),
        "turns": moved, "examples": examples, "history": history,
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1))
    print(json.dumps({k: report[k] for k in ("train_samples", "held_out_samples", "mean_delta_logp", "mean_kl",
                                             "early_stop", "sanity", "seconds_per_step", "peak_gb")},
                     ensure_ascii=False), flush=True)
    print(f"adapter: {out}  (report.json has the per-turn moves and base-vs-adapter replies)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
