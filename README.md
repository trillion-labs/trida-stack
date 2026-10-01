# trida-stack

**One repo for turning an autoregressive LLM into a block-diffusion LLM, and for serving it.**

Two independent stacks:

| | stack | what it does |
|---|---|---|
| 🏋️ | **`train/`** | **AR → diffusion conversion.** Block-diffusion SFT starting from a *pretrained AR checkpoint* (no diffusion pretraining), with an AR auxiliary loss so the same weights keep a working causal head. |
| ⚡ | **`inference/`** | **Diffusion decoding + serving — the `nano-inference` stack.** A thin, nanoGPT-style wrapper (`serve.py` / `chat.py` / `eval.py`) that serves one checkpoint in several decode modes (causal / block-diffusion / self-speculative) behind **SGLang** — or a self-contained **vLLM** backend (`inference/vllm/`) — OpenAI-compatible. |

`train/` and `inference/` do not import each other. `benchmark/` holds the heavier agentic eval
harnesses; `inference/` ships its own quick quality/throughput benchmarks (gsm8k / mmlu_pro / ifeval).

---

## Supported model families

The trainer is architecture-agnostic in principle (any HF causal LM whose decoder is reachable via
`model.get_decoder()`), but the block-diffusion masking and the cached decoders are validated on:

| family | attention | notes |
|---|---|---|
| **Qwen3** | **full attention** (dense) | the baseline path: dense block masks, standard KV cache |
| **Qwen3.5** | **hybrid** — gated-delta *linear* attention layers + periodic *full* attention layers | needs hybrid-aware masking and cache handling: linear layers have no length dimension to crop, so speculative rollback uses conv/recurrent **state snapshot + restore** rather than a KV crop |

The hybrid (Qwen3.5) path is the **two-stream** reference model served in `inference/`
(`--mode causal / diffusion / self-spec`).

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

---

## Quickstart

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

The saved checkpoint is a **stock HF causal LM** plus an added `<|mask|>` token and a
`block_diffusion.json` sidecar (`bd_size`, `mask_id`, `ar_loss_weight`) — so it loads with plain
`AutoModelForCausalLM.from_pretrained`.

### Inference (serving — the `nano-inference` stack)
`inference/` is a thin, nanoGPT-style serving wrapper: it maps a `--mode` to a backend decode
algorithm and launches an OpenAI-compatible server; the heavy decode kernels live in the backend.
One checkpoint, several decode modes:
```bash
# trillionlabs/Trida2.0-4B is public; no login needed. Note its licence differs from this repo's -- see COMPLIANCE.md.
python inference/serve.py                    --port 30000   # default: self-spec (diffusion draft -> AR verify)
python inference/serve.py --mode causal     --port 30000   # native AR
```
The model defaults to **[`trillionlabs/Trida2.0-4B`](https://huggingface.co/trillionlabs/Trida2.0-4B)**
(override with a positional `<hf-id-or-local-path>` or `TRIDA_MODEL=…`).
Beyond this stack's two-stream model, `serve.py` also fronts other block-diffusion families the
backend supports: `sdar`, `llada2-0`, `llada2-1-speed`, `llada2-1-quality`.
Self-speculative decoding is **lossless** at matched sampling settings: every committed token is
either an accepted draft that equals the AR token, or the AR correction. Its benefit is
**tokens per forward**, not a different answer. See [`inference/README.md`](inference/README.md)
for chat, benchmarks, and the model contract.

### Serving
Served via **SGLang**: true block-diffusion (bidirectional-within-block attention,
variable-length commits) and the self-speculative path.

---

## Layout

```
train/                   AR -> diffusion SFT trainer    (entry: train.py -> train.train:main)
  README.md                  launching, resume, model-family notes
  hf_block_diffusion.py      block masks, loss, HF model wrapper (dense / Qwen3)
  hf_block_diffusion_hybrid.py   hybrid path (Qwen3.5: gated-delta linear + full attention)
  block_gated_delta_rule/    fetch recipe for the block-causal gated-delta Triton kernels
                             (PolyForm-NC upstream, NOT vendored -- run fetch_kernels.sh first)
  data/text_sft_data.py      packing / bucketing / multi-turn
inference/               nano-inference serving stack (thin, nanoGPT-style)
  README.md                  serving, chat, benchmarks
  MODEL_CONTRACT.md          what a checkpoint must provide to be servable
  serve.py                   --mode -> backend decode algorithm; launches the server
  chat.py                    OpenAI-compatible client
  eval.py / eval_ifeval.py   gsm8k / mmlu_pro / ifeval quality + throughput
  ifeval_lib/                vendored IFEval scorers
  configs/                   per-family decode configs
  test_smoke.py              offline smoke test
  vllm/                      second backend: out-of-tree vLLM plugin (block-diffusion + self-spec/AR-Trust)
    vllm_native_diffusion/     the plugin package (VLLM_PLUGINS=trida_diffusion)
    tools/, docs/              Slurm eval jobs, worklog + design notes
benchmark/               heavier agentic eval harnesses (bfcl_v4, tau2, functionchat, ko_agentbench,
                         swe_bench, terminal_bench) + serving/
data/                    dataset download scripts + catalog
```

## Install
```bash
pip install -r requirements.txt              # training stack
pip install -r inference/requirements.txt    # serving/eval client
```
Serving needs the diffusion-serving **SGLang** backend — install separately (not vendored here);
see [`inference/README.md`](inference/README.md). The **vLLM** backend is a plugin:
`pip install -e inference/vllm` and `VLLM_PLUGINS=trida_diffusion` — see [`inference/vllm/README.md`](inference/vllm/README.md).

### Third-party kernels — one-time fetch (required)

Three pieces of this stack are **PolyForm Noncommercial 1.0.0** upstream, so they are *not*
vendored in this Apache-2.0 repo. Each ships a pinned fetch-and-patch recipe instead. Run the one
for the path you use, **before** training or serving — without it the import fails.

| you are… | run | what it assembles |
|---|---|---|
| training the Qwen3.5 two-stream (hybrid) path | `bash train/block_gated_delta_rule/fetch_kernels.sh` | two-stream Gated-DeltaNet + ShortConv Triton kernels — [recipe](train/block_gated_delta_rule/README.md) |
| serving via **SGLang** | follow [`inference/sglang/README.md`](inference/sglang/README.md) | the patched HybridDiffusion SGLang backend (clone → patch → install → shape) |
| serving via **vLLM** | follow [`inference/vllm/vllm_native_diffusion/KERNELS.md`](inference/vllm/vllm_native_diffusion/KERNELS.md) | `block_causal_readout.py`, the block-end readout kernel |

Each recipe pins an upstream commit (`yuchen-zhu-zyc/HybridDiffusion@6ca547a`) and applies our
patch. **The fetched code is noncommercial-licensed and is not covered by this repo's Apache-2.0
license** — see `NOTICE` and `COMPLIANCE.md`.

## License
See `LICENSE`, plus `NOTICE` and `COMPLIANCE.md` for third-party attribution.

## Star History

<!-- Renders once the repo is public. SLUG = trillion-labs/trida-stack (current remote);
     if the repo is renamed for the open-source cut, update BOTH urls below. -->
<a href="https://star-history.com/#trillion-labs/trida-stack&Date">
  <img src="https://api.star-history.com/svg?repos=trillion-labs/trida-stack&type=Date" alt="Star History Chart" width="600">
</a>
