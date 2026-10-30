# trida-stack

**English** | [한국어](README.ko.md)

**One repo for turning an autoregressive LLM into a block-diffusion LLM, and for serving it.**

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97%20Zoro--1-diffusion%20%2B%20decisions-yellow)](https://huggingface.co/trillionlabs/Zoro-1)
[![Trida2.0-4B](https://img.shields.io/badge/%F0%9F%A4%97%20Trida2.0--4B-block%20diffusion-yellow)](https://huggingface.co/trillionlabs/Trida2.0-4B)
[![Trida-2.0-4B-1006](https://img.shields.io/badge/%F0%9F%A4%97%20Trida--2.0--4B--1006-vision--language-yellow)](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)

Two independent stacks:

| | stack | what it does |
|---|---|---|
| 🏋️ | **`train/`** | **AR → diffusion conversion.** Block-diffusion SFT starting from a *pretrained AR checkpoint* (no diffusion pretraining), with an AR auxiliary loss so the same weights keep a working causal head. Optional **typed-decision joint training** (a pointer head for yes/no, choice and score questions) and **image inputs** for vision-language checkpoints. |
| ⚡ | **`inference/<model>/`** | **nano-inference: one small, nanoGPT-style serving stack per model.** [`inference/zoro-1/`](inference/zoro-1/) serves `trillionlabs/Zoro-1`: self-speculative generation and typed decisions from one checkpoint, OpenAI-compatible, on **SGLang**, **vLLM** or **MLX**, with a [speedrun leaderboard](inference/zoro-1/README.md#speedrun). |

`train/` and `inference/` do not import each other. `benchmark/` holds the heavier agentic eval
harnesses; `inference/` ships its own quick quality/throughput benchmarks (gsm8k / mmlu_pro / ifeval).

---

## Models

Checkpoints trained with this repo, on Hugging Face:

| model | what it is | serve it with |
|---|---|---|
| [![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FZoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1) | 4B two-stream block-diffusion model (Qwen3.5-4B), jointly trained as an agent model **and** a decision model (`jev_head.pt` pointer head). Gated: request access, then `hf auth login`. | [`inference/zoro-1/`](inference/zoro-1/): `--mode self-spec` (default) / `causal` / `decide` |
| [![Trida2.0-4B](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FTrida2.0--4B-yellow)](https://huggingface.co/trillionlabs/Trida2.0-4B) | 4B block-diffusion model (Qwen3.5-4B, hybrid attention) with a causal head for lossless self-speculative decoding. Apache-2.0. | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) (Apple Silicon), [`inference/zoro-1/vllm/`](inference/zoro-1/vllm/) |
| [![Trida-2.0-4B-1006](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FTrida--2.0--4B--1006-yellow)](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006) | vision-language variant (Qwen3.5 vision tower + projector). | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) (image input) |

The agentic benchmark harnesses in [`benchmark/`](benchmark/) also run
[`trillionlabs/Trida-7B-Preview`](https://huggingface.co/trillionlabs/Trida-7B-Preview).

---

## Supported model families

The trainer is architecture-agnostic in principle (any HF causal LM whose decoder is reachable via
`model.get_decoder()`), but the block-diffusion masking and the cached decoders are validated on:

| family | attention | notes |
|---|---|---|
| **Qwen3** | **full attention** (dense) | the baseline path: dense block masks, standard KV cache |
| **Qwen3.5** | **hybrid** — gated-delta *linear* attention layers + periodic *full* attention layers | needs hybrid-aware masking and cache handling: linear layers have no length dimension to crop, so speculative rollback uses conv/recurrent **state snapshot + restore** rather than a KV crop |

The hybrid (Qwen3.5) path is the **two-stream** model behind Zoro-1, served in `inference/zoro-1/`
(`--mode causal / self-spec / decide`).

---

## How the conversion works

Block-diffusion SFT (Fast-dLLM v2 style). Each row places a noised and a clean copy of the response
after the prompt:

```
[ S (prompt) | x_t (response, noised) | x_0 (response, clean) ]
```

A custom attention mask ties them together:

- **S (prompt)** — causal within itself, visible to everything after it (conditioning).
- **x_t (noised block k)** — bidirectional *within* its own block, and attends to the prompt and the
  *previous* response blocks (their clean x_0 copies). So generation is autoregressive **across**
  blocks but diffusion **within** a block.
- **x_0 (clean)** — block-causal over itself (token-causal when the AR aux loss is on).

Loss = masked cross-entropy at the noised x_t positions (the diffusion objective) + an optional
next-token CE on x_0 (the AR auxiliary), combined as `(diff + w·ar) / (1 + w)`. Fast-dLLM v2's
**complementary masking** supervises every token across two paired views.

Two knobs matter most:

- **`--bd_size`** — block length (e.g. 32). Decoding fills one block at a time.
- **`--ar_loss_weight`** (w) — weight of the AR auxiliary loss. `> 0` keeps a usable **causal head**,
  which is what makes speculative decoding (and plain AR serving) possible from the same checkpoint.

`--within_block_causal` makes x_t token-causal within the block too; combined with a causal x_0 this
makes the model *fully causal* — loadable by a standard autoregressive serving engine — at the cost
of some draft quality.

### Typed decisions (joint training)

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-decision%20model%3A%20Zoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1)

The same weights can also be trained as a **decision model**: given a *state* (any JSON, plus an
optional image) and typed *questions*, it returns calibrated probabilities instead of free text.

| type | question | answer |
|---|---|---|
| `noul` | yes/no | P(yes) |
| `choice` | `{key: description}` | a distribution over the keys |
| `score` | a list of levels, lowest first | the expected level |

`--jev_data <kev-format JSONL>` adds this objective in [kev](https://github.com/jaredpalmer/kev)'s
scheme (`train/jev.py`): a small **pointer head** on the clean causal stream scores each option's
`</opt>` state against the question's `<decide>` state. Each optimizer step runs the usual SFT
micro-batches, then `--jev_micro_per_step` decision micro-batches whose loss is added with weight
`--jev_loss_weight`. Every checkpoint gets a `jev_head.pt`; `train/tools/export_kev_checkpoint.py`
exports kev's layout. Details: [`train/README.md`](train/README.md#typed-decision-training-jevpy).

### Images (vision-language checkpoints)

A Qwen3.5 checkpoint with a `vision_config` trains on conversations with image parts
(`{"type": "image", "image": "imgs/0001.png"}`) alongside text-only data. The vision encoder is
frozen; the projector trains at `--projector_lr_scale` × `--lr`; images above `--max_image_pixels`
are downscaled. `train/tools/compose_vl_checkpoint.py` attaches a base vision tower to a text-only
checkpoint. Details: [`train/README.md`](train/README.md#images-vision-language-checkpoints).

---

## Quickstart

### Tutorials

| notebook | what it walks through | model |
|---|---|---|
| [`train/getting_started.ipynb`](train/getting_started.ipynb) | the data format and a minimal block-diffusion SFT run (1 GPU, 2 steps) | your own AR base (e.g. Qwen3.5-4B) |
| [`inference/zoro-1/getting_started.ipynb`](inference/zoro-1/getting_started.ipynb) | serve, call, benchmark, and read the numbers | [![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-Zoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1) |

### Train (AR → diffusion)
> Qwen3.5 two-stream (hybrid) runs need the one-time kernel fetch first —
> `bash train/block_gated_delta_rule/fetch_kernels.sh` (see [Install](#install)).

```bash
PYTHONPATH=. torchrun --nproc_per_node=8 train.py \
  --model_id Qwen/Qwen3-4B \
  --bd_size 32 --ar_loss_weight 0.2 \
  --max_length 32768 --pack \
  --save_dir checkpoints/my-bd-run
```
Native-torch **FSDP2**, no DeepSpeed. `python train.py --help` lists all flags (packing,
length/response bucketing, multi-turn supervision, activation offload, fused CE, …).

Add a typed-decision objective to the same run with `--jev_data <kev-format JSONL files/dirs>`
(see [Typed decisions](#typed-decisions-joint-training)); `--jev_only` trains the decision objective alone (no SFT data, no diffusion loss).

The saved checkpoint is a **stock HF causal LM** plus an added `<|mask|>` token and a
`block_diffusion.json` sidecar (`bd_size`, `mask_id`, `ar_loss_weight`) — so it loads with plain
`AutoModelForCausalLM.from_pretrained`.

### Inference (serving — `nano-inference`, one stack per model)

[![Zoro-1](https://img.shields.io/badge/%F0%9F%A4%97-trillionlabs%2FZoro--1-yellow)](https://huggingface.co/trillionlabs/Zoro-1)

Each model gets its own small, readable serving stack under `inference/<model>/`, in the spirit of nanoGPT.
[`inference/zoro-1/`](inference/zoro-1/) serves **[`trillionlabs/Zoro-1`](https://huggingface.co/trillionlabs/Zoro-1)**
(gated: request access, then `hf auth login`):
```bash
python inference/zoro-1/serve.py                  --port 30000   # default: self-spec (diffusion draft -> AR verify, b15_g8)
python inference/zoro-1/serve.py --mode causal    --port 30000   # native AR
python inference/zoro-1/serve.py --mode decide    --port 30001   # typed decisions + chat (stock SGLang)
```
Self-speculative decoding is **lossless** at matched sampling settings: every committed token is
either an accepted draft that equals the AR token, or the AR correction. Its benefit is
**tokens per forward**, not a different answer. See [`inference/zoro-1/README.md`](inference/zoro-1/README.md)
for chat, decisions, benchmarks, the model contract and the **speedrun leaderboard** (H100, L40S, MLX).

### Decisions (`--mode decide`)
`--mode decide` runs `serve_decide.py` on **stock SGLang ≥ 0.5.22** (no fork or patch): typed
decisions at `POST /v1/decide`, reason-then-decide (`"reasoning": true`), and OpenAI chat with tool
calls, from one engine and one copy of the weights.
```bash
curl -s localhost:30001/v1/decide -H 'content-type: application/json' -d '{
  "request": {
    "state": {"ticket": "Order #W1 arrived broken. I want my money back."},
    "questions": {
      "intent": {"type": "choice", "instructions": "What does the customer want?",
                 "criteria": {"refund": "money back", "exchange": "a replacement", "info": "just information"}},
      "urgent": {"type": "noul", "instructions": "Should this jump the queue?"}}},
  "reasoning": false}'
# -> {"answers": {"intent": {"type": "choice", "choice": "refund", "probabilities": {...}},
#                 "urgent": {"type": "noul", "noul": 0.93}}, ...}
```
See [`inference/zoro-1/README.md#decisions`](inference/zoro-1/README.md#decisions).

### Backends
| backend | where | what |
|---|---|---|
| **SGLang** (patched) | [`inference/zoro-1/sglang/`](inference/zoro-1/sglang/) | true block-diffusion (bidirectional-within-block attention, variable-length commits) and the self-speculative path |
| **SGLang** (stock) | `inference/zoro-1/serve_decide.py` | `--mode decide`: typed decisions + chat |
| **vLLM** | [`inference/zoro-1/vllm/`](inference/zoro-1/vllm/) | out-of-tree plugin, self-spec |
| **MLX** | [`inference/zoro-1/mlx/`](inference/zoro-1/mlx/) | on-device (Apple Silicon): lossless self-spec, OpenAI-compatible server, image input, a small agent; q8 fits 16–18 GB |

### Speedrun
A [NanoGPT-speedrun](https://github.com/KellerJordan/modded-nanogpt)-style leaderboard: answer a fixed
GSM8K subsample with Zoro-1, one request at a time, as fast as possible, with accuracy ≥ 87.0%.
Tracks: 1× H100, 1× L40S, Apple Silicon (MLX). Rules and records:
[`inference/zoro-1/README.md#speedrun`](inference/zoro-1/README.md#speedrun).

---

## Layout

```
train/                   AR -> diffusion SFT trainer    (entry: train.py -> train.train:main)
  README.md                  launching, resume, model-family notes
  hf_block_diffusion.py      block masks, loss, HF model wrapper (dense / Qwen3)
  hf_block_diffusion_hybrid.py   hybrid path (Qwen3.5: gated-delta linear + full attention)
  jev.py                     typed-decision objective (pointer head, kev's scheme)
  tools/                     export_kev_checkpoint.py, compose_vl_checkpoint.py, parity/FSDP checks
  getting_started.ipynb      training tutorial
  block_gated_delta_rule/    fetch recipe for the block-causal gated-delta Triton kernels
                             (PolyForm-NC upstream, NOT vendored -- run fetch_kernels.sh first)
  data/text_sft_data.py      packing / bucketing / multi-turn
inference/               nano-inference: one small serving stack per model
  zoro-1/                    trillionlabs/Zoro-1
    README.md                  serving, decisions, benchmarks, speedrun leaderboard
    getting_started.ipynb      serving tutorial
    MODEL_CONTRACT.md          what the serving code expects from the checkpoint
    serve.py                   --mode causal | self-spec | decide; launches the server
    serve_decide.py, decide_*.py, jev*.py   typed-decision server (/v1/decide) on stock SGLang
    chat.py                    OpenAI-compatible client
    eval.py / eval_ifeval.py   gsm8k / mmlu_pro / ifeval quality + throughput
    speedrun/                  leaderboard rules, records per hardware track, charts
    ifeval_lib/                vendored IFEval scorers
    configs/                   self-spec decode config (b15_g8)
    sglang/                    recipe for the patched SGLang backend
    vllm/                      out-of-tree vLLM plugin (self-spec)
    mlx/                       on-device backend (Apple Silicon): MLX self-spec decoder, OpenAI server, agent
benchmark/               heavier agentic eval harnesses (bfcl_v4, tau2, functionchat, ko_agentbench,
                         swe_bench, terminal_bench) + serving/
data/                    dataset download scripts + catalog
```

## Install
```bash
pip install -r requirements.txt              # training stack
pip install -r inference/zoro-1/requirements.txt    # serving/eval client
```
Serving needs the diffusion-serving **SGLang** backend — install separately (not vendored here);
see [`inference/zoro-1/README.md`](inference/zoro-1/README.md). The **vLLM** backend is a plugin:
`pip install -e inference/zoro-1/vllm` and `VLLM_PLUGINS=trida_diffusion` — see [`inference/zoro-1/vllm/README.md`](inference/zoro-1/vllm/README.md).

### Third-party kernels — one-time fetch (required)

Three pieces of this stack are **PolyForm Noncommercial 1.0.0** upstream, so they are *not*
vendored in this Apache-2.0 repo. Each ships a pinned fetch-and-patch recipe instead. Run the one
for the path you use, **before** training or serving — without it the import fails.

| you are… | run | what it assembles |
|---|---|---|
| training the Qwen3.5 two-stream (hybrid) path | `bash train/block_gated_delta_rule/fetch_kernels.sh` | two-stream Gated-DeltaNet + ShortConv Triton kernels — [recipe](train/block_gated_delta_rule/README.md) |
| serving via **SGLang** | follow [`inference/zoro-1/sglang/README.md`](inference/zoro-1/sglang/README.md) | the patched HybridDiffusion SGLang backend (clone → patch → install → shape) |
| serving via **vLLM** | follow [`inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md`](inference/zoro-1/vllm/vllm_native_diffusion/KERNELS.md) | `block_causal_readout.py`, the block-end readout kernel |

Each recipe pins an upstream commit (`yuchen-zhu-zyc/HybridDiffusion@6ca547a`) and applies our
patch. **The fetched code is noncommercial-licensed and is not covered by this repo's Apache-2.0
license** — see `NOTICE` and `COMPLIANCE.md`.

## Getting help, contributing, reporting

| | |
|---|---|
| Questions about install or usage | [Discussions · Q&A](https://github.com/trillion-labs/trida-stack/discussions/categories/q-a) |
| A reproducible bug | [Issues](https://github.com/trillion-labs/trida-stack/issues/new/choose) |
| A feature idea | [Discussions · Ideas](https://github.com/trillion-labs/trida-stack/discussions/categories/ideas) |
| A problem with a model revision | [Zoro-1 Community](https://huggingface.co/trillionlabs/Zoro-1/discussions) |
| Contributing | [CONTRIBUTING.md](CONTRIBUTING.md) |
| A security vulnerability | the private channel in [SECURITY.md](SECURITY.md), **not a public issue** |

What each channel is for and what response to expect: [COMMUNITY.md](COMMUNITY.md). Code of conduct:
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Who decides what, and where decisions are recorded:
[GOVERNANCE.md](GOVERNANCE.md).

## License
See `LICENSE`, plus `NOTICE` and `COMPLIANCE.md` for third-party attribution.

## Star History

<!-- Renders once the repo is public. SLUG = trillion-labs/trida-stack (current remote);
     if the repo is renamed for the open-source cut, update BOTH urls below. -->
<a href="https://star-history.com/#trillion-labs/trida-stack&Date">
  <img src="https://api.star-history.com/svg?repos=trillion-labs/trida-stack&type=Date" alt="Star History Chart" width="600">
</a>
