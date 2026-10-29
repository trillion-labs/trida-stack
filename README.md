# trida-stack

**One repo for turning an autoregressive LLM into a block-diffusion LLM, and for serving it.**

Two independent stacks:

| | stack | what it does |
|---|---|---|
| 🏋️ | **`train/`** | **AR → diffusion conversion.** Block-diffusion SFT starting from a *pretrained AR checkpoint* (no diffusion pretraining), with an AR auxiliary loss so the same weights keep a working causal head. |
| ⚡ | **`inference/<model>/`** | **nano-inference: one small, nanoGPT-style serving stack per model.** [`inference/zoro-1/`](inference/zoro-1/) serves `trillionlabs/Zoro-1`: self-speculative generation and typed decisions from one checkpoint, OpenAI-compatible, on **SGLang**, **vLLM** or **MLX**, with a [speedrun leaderboard](inference/zoro-1/README.md#speedrun). |

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

### Inference (serving — `nano-inference`, one stack per model)
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
inference/               nano-inference: one small serving stack per model
  zoro-1/                    trillionlabs/Zoro-1
    README.md                  serving, decisions, benchmarks, speedrun leaderboard
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
| 설치·사용 질문 | [Discussions · Q&A](https://github.com/trillion-labs/trida-stack/discussions/categories/q-a) |
| 재현 가능한 결함 | [Issues](https://github.com/trillion-labs/trida-stack/issues/new/choose) |
| 기능 아이디어 | [Discussions · Ideas](https://github.com/trillion-labs/trida-stack/discussions/categories/ideas) |
| 모델 revision 문제 | [Zoro-1 Community](https://huggingface.co/trillionlabs/Zoro-1/discussions) |
| 기여하려면 | [CONTRIBUTING.md](CONTRIBUTING.md) |
| 보안 취약점 | **공개 Issue가 아니라** [SECURITY.md](SECURITY.md)의 비공개 경로 |

채널별 기준과 응답 기대치는 [COMMUNITY.md](COMMUNITY.md), 행동 규범은
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md), 누가 무엇을 결정하고 그 결정이 어디에
기록되는지는 [GOVERNANCE.md](GOVERNANCE.md)에 있습니다.

## License
See `LICENSE`, plus `NOTICE` and `COMPLIANCE.md` for third-party attribution.

## Star History

<!-- Renders once the repo is public. SLUG = trillion-labs/trida-stack (current remote);
     if the repo is renamed for the open-source cut, update BOTH urls below. -->
<a href="https://star-history.com/#trillion-labs/trida-stack&Date">
  <img src="https://api.star-history.com/svg?repos=trillion-labs/trida-stack&type=Date" alt="Star History Chart" width="600">
</a>
