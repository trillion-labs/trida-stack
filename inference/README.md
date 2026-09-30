<div align="center">

# nano-inference

**A minimal, hackable inference stack for diffusion LLMs.**
Serve one checkpoint three ways — autoregressive, block-diffusion, or self-speculative — behind one OpenAI-compatible API, in a handful of readable files.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Backends](https://img.shields.io/badge/backends-vLLM%20%C2%B7%20SGLang-8A2BE2.svg)](#backends)
[![API](https://img.shields.io/badge/API-OpenAI--compatible-green.svg)](#quickstart)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](#roadmap--contributing)

[Quickstart](#quickstart) · [Results](#results) · [How it works](#how-it-works) · [Decode modes](#three-decode-modes) · [Models](#supported-models) · [Backends](#backends) · [Roadmap](#roadmap--contributing)

</div>

---

**Diffusion LLMs** decode a *block* of tokens at once and refine it over a few denoising passes, instead of one token per forward pass. nano-inference gives you the two things you actually need to use one — **serving** and **reproducible benchmarks** — without a framework to learn. Point it at any compatible checkpoint (a Hugging Face id or a local path) and you're running; the same weights serve autoregressively, as block-diffusion, or as lossless self-speculative decoding by flipping `--mode`.

It ships with **[`trillionlabs/Trida2.0-4B`](https://huggingface.co/trillionlabs/Trida2.0-4B)** — a Qwen3.5-based two-stream model — as the worked reference, so you can watch the whole loop — serve → chat → benchmark — end to end. **New here? Open [`getting_started.ipynb`](getting_started.ipynb)** for a runnable walkthrough.

## Quickstart

```bash
# 1. install client deps + point at your serving backend (see Backends below)
source env.example.sh              # sets SGLANG_PYTHON + cache dirs
pip install -r requirements.txt

# 2. serve the reference model (trillionlabs/Trida2.0-4B; private during preview — see Model access)
huggingface-cli login
python serve.py --port 30000                 # default mode: self-spec
#    …or any compatible checkpoint: python serve.py <hf-id-or-local-path> --port 30000

# 3. call it (OpenAI-compatible)
python chat.py "What is 17 times 24? Answer with just the number."
```

```text
17 × 24 = 408
408
```

That's it — an OpenAI-compatible endpoint at `http://localhost:30000/v1/chat/completions`. The default mode is **`self-spec`** (the recommended decoder for Trida2.0-4B); `--mode causal` serves the *same weights* autoregressively; add `--dry-run` to print the launch command without starting a server.

### Model access

The model is resolved as **positional arg → `$TRIDA_MODEL` → `trillionlabs/Trida2.0-4B`**, and the same rule applies to the vLLM script (`CKPT` → `$TRIDA_MODEL` → default) and the notebook. `trillionlabs/Trida2.0-4B` is **private on Hugging Face during the preview period**: your account needs access to the repo (ask the Trillion Labs team), then authenticate once before the first download —

```bash
huggingface-cli login            # or: export HF_TOKEN=hf_...
```

A `401` / `Repository not found` from the backend at startup means the token is missing or the account has no access yet. A local checkpoint directory needs no token at all.

## Why nano-inference

- **One checkpoint, three decoders.** `causal`, `diffusion`, and `self-spec` share one set of weights — compare quality vs. speed by changing a flag, not a model.
- **Lossless speedup that's real.** `self-spec` drafts with diffusion and verifies against exact AR logits, so it matches autoregressive output while running **~1.5× faster** (see [Results](#results)).
- **Two backends, one API.** A thin **SGLang** wrapper *and* a native **vLLM 0.27.x** plugin — pick your engine, keep your client code.
- **Actually readable.** `serve.py`, `chat.py`, `eval.py` + decode configs. No plugin registry, no config sprawl. Read it in an afternoon; fork it in one.
- **Bring your own model.** Several block-diffusion families work out of the box; adding one is a decode config plus a `MODES` entry. See [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md).

## Results

Measured on **`trillionlabs/Trida2.0-4B`** — GSM8K (200 problems), thinking on, temperature 1.0 / top_k 50 / top_p 0.95, 8192-token budget, **vLLM backend, one H100 per replica**, client concurrency C. Your numbers depend on your model; these show the *shape* of the trade-off.

| C | `causal` tok/s (agg / per request) | `self-spec` tok/s (agg / per request) | vs. AR (agg) | p50 latency |
|---:|---:|---:|---:|---:|
| 1 | 213 / 210 | **311 / 327** | **1.46×** | 1.29 s |
| 4 | 745 / 188 | **966 / 271** | **1.30×** | 1.47 s |
| 8 | 1251 / 168 | **1495 / 222** | **1.20×** | 1.85 s |
| 16 | 1690 / 141 | **1854 / 161** | **1.10×** | 2.52 s |

Self-speculative decoding keeps the causal model's output distribution (speculative rejection sampling — every emitted token is an accepted draft or the recovery sample from the AR distribution) while decoding faster: accuracy 85.5–89.5 % vs AR 89.0 % across these runs (n = 200, ± 3 pt), runaway requests ≤ 0.5 %. The per-stream gain narrows with concurrency because a self-spec step costs more than an AR step and the batch amortises that less at high C. Full run-by-run detail: [`vllm/README.md`](vllm/README.md).

## How it works

```
  your prompt                                         decode mode
      │                                              (causal / diffusion / self-spec)
      ▼                                                     │
  chat.py / eval.py ──HTTP──▶ serve.py ──▶  backend  ──────┘──▶ tokens ──▶ response
  (OpenAI-compatible)         (thin wrapper:   (SGLang or native vLLM:
                               mode→algo→config) KV cache, batching, decode kernels)
```

`serve.py` is a thin wrapper: it maps `--mode` to the backend's decode algorithm + config and launches the server. Everything heavy — batching, KV cache, the decode kernels — lives in the backend. Clients just speak the OpenAI API.

## Three decode modes

All three run over the **same weights** — they differ only in how tokens are decoded:

| `--mode` | what it is | decode block |
|---|---|---|
| `causal` | native autoregressive | — |
| `diffusion` | pure iterative block-diffusion (confidence-based denoising). **Not supported for Trida2.0-4B** — its full-mask training recipe leaves partial-mask states untrained; use `self-spec`. Kept for the other block-diffusion families below and for future checkpoints | runtime `block_size: 3` = block **B=4** |
| **`self-spec`** (default) | self-speculative: diffusion draft + autoregressive verify — lossless vs `causal`, faster per stream | block 7 / gen 4 |

Why `block_size: 3` means block **B=4**: the shift variant carries a seed token across blocks (`[seed, MASK, MASK]`), and the final shifted logit produces the next block's seed — so a logical block spans 4 positions using 3 query slots. `threshold` is the per-step commit gate (higher = stricter, more denoising passes).

## Supported models

nano-inference is as broad as the serving backend's zoo — several **block-diffusion LLM families**, not just one. Point `--model` at the checkpoint and pick the matching `--mode`:

| family | `--mode` | example public checkpoints |
|---|---|---|
| two-stream (reference) | `causal`, `diffusion`, `self-spec` | your checkpoint |
| SDAR (Qwen-based diffusion) | `sdar` | `JetLM/SDAR-1.7B` / `-4B` / `-8B` / `-30B-A3B-Chat` |
| LLaDA 2.0 | `llada2-0` | `inclusionAI/LLaDA2.0-mini`, `…-flash` |
| LLaDA 2.1 | `llada2-1-speed`, `llada2-1-quality` | `inclusionAI/LLaDA2.1-mini`, `…-flash` |

```bash
python serve.py JetLM/SDAR-4B-Chat        --mode sdar     --port 30000
python serve.py inclusionAI/LLaDA2.0-mini --mode llada2-0 --port 30000
```

Adding a new family = one backend decode algorithm + one `MODES` entry. The two-stream modes are validated end-to-end; the SDAR / LLaDA2 modes are wired to the backend's shipped configs but **not yet verified** — validate against a checkpoint before relying on them. Models outside the backend's zoo (Fast-dLLM, Dream, OpenDLLM) need backend support added first.

## Usage

**Serve** — three modes over one checkpoint:

```bash
python serve.py --port 30000                       # trillionlabs/Trida2.0-4B (or $TRIDA_MODEL), self-spec (default)
python serve.py --mode causal    --port 30000
python serve.py <hf-id-or-local-path> --port 30000   # any other checkpoint
```

Optional `--reasoning-parser qwen3` (or `deepseek-r1`) makes the *server* split the `<think>…</think>` trace, returning `message.reasoning_content` and a clean `message.content`.

**Chat:**

```bash
python chat.py "What is 17 * 23?"     # one-shot
python chat.py                         # interactive REPL
```

**Evaluate** — one run, all the numbers. Serve with `--log` so the server's own counters are captured, then point `eval.py` at both the endpoint and that log:

```bash
python serve.py --port 30000 --log server.log &          # any mode; self-spec also logs tok/fwd + acceptance
python eval.py gsm8k --server-log server.log              # full test set
python eval.py mmlu_pro --num-problems 1000               # random subsample (seed 42)
python eval_ifeval.py --port 30000                        # IFEval strict/loose
```

```text
gsm8k: 36/40 = 90.00%   errors=0   wall=110s
  throughput : 170 tok/s aggregate (C=8), 45 tok/s per request
  latency    : p50 4.81s  p90 7.92s  mean 5.03s   (mean output 228 tokens, 0.0% hit max_tokens=8192)
  tok/forward: 1.87   [vLLM DiffusionDecoding counters]
  -> results/gsm8k_20260918-160102/summary.json, .../requests.jsonl
```

From the *same* requests you get **accuracy**, **tok/s** (aggregate at the client concurrency `--max-workers`, and per request), **latency** p50/p90/mean, mean output length, and — with `--server-log` — **tokens per forward** (vLLM: from its `DiffusionDecoding` counters; SGLang self-spec: from the algorithm's stats line; SGLang causal/diffusion log no counter → `n/a`; causal is 1 by definition). Every request is written to `requests.jsonl` (prediction, gold, tokens, latency) so you can re-slice without re-running. For a per-stream latency profile use `--max-workers 1`; for aggregate throughput raise it. Keep `--max-tokens` a *budget*, not a knob: if the truncation percentage is not ~0, the number measures the budget, not the model. On the vLLM script use `LOG=server.log bash vllm/serve_diffusion.sh`.

Defaults: temp 1.0 / top_p 0.95 / top_k 50, no presence penalty, thinking on, `--max-tokens 8192` — the canonical self-spec setting, identical to the SGLang self-spec config, so a number from `eval.py` means the same thing on either backend. Both backends honor these fields in self-spec (vLLM via its stock rejection sampler, `TRIDA_SS_SAMPLE=0` for exact greedy). `gsm8k`/`mmlu_pro` are self-contained (extraction-based); **ifeval** uses the Google Research IFEval scorer vendored in `ifeval_lib/` (Apache-2.0). `chat.py`/`eval.py` pick the served model id up from `/v1/models`, so they work unchanged against either backend.

## Backends

Two ways to run the decode kernels behind `serve.py`'s API — pick one.

### SGLang (wrapped by `serve.py`)

The default path. `serve.py` launches an SGLang server built from its public upstream. One-time setup: **clone → patch → install → shape** (see [`sglang/README.md`](sglang/README.md)). A raw two-stream checkpoint needs the one-time shaping step:

```bash
python sglang/tools/shape_for_sglang.py <raw_ckpt>   # -> <raw_ckpt>_sglang
export SGLANG_PYTHON=/path/to/backend-venv/bin/python
python serve.py <raw_ckpt>_sglang --mode diffusion --port 30000
```

### vLLM (native plugin, no fork)

A **self-contained** serve inside a stock **vLLM 0.27.x** install — no fork, no SGLang, no shaping:

```bash
pip install vllm==0.27.*
pip install -e vllm                       # registers the `trida_diffusion` plugin
# one-time: fetch the block-end readout kernel (PolyForm-NC upstream, not vendored)
#   -> vllm/vllm_native_diffusion/KERNELS.md
bash vllm/serve_diffusion.sh                              # trillionlabs/Trida2.0-4B (or $TRIDA_MODEL), self-spec
MAX_NUM_SEQS=8 bash vllm/serve_diffusion.sh               # batched self-spec (PIECEWISE cuda graphs)
MODE=causal bash vllm/serve_diffusion.sh                  # AR reference; CKPT=<hf-id-or-local-path> for another checkpoint
```

Runs self-spec (and pure diffusion) as an out-of-tree plugin (arch `Qwen3_5ForBlockDiffusion`): FULL cuda graphs for a single stream, PIECEWISE for batched serving, lossless in both. See [`vllm/README.md`](vllm/README.md) for env knobs and status.

## Install & environment

`serve.py` stays thin and does not set backend plumbing (cache dirs, flashinfer workspace, CUDA stubs). On a fresh machine, source the example first:

```bash
source env.example.sh          # TRIDA_MODEL + SGLANG_PYTHON + cache dirs
pip install -r requirements.txt
```

The **SGLang** backend is not vendored here — build it from its public upstream with the recipe in [`sglang/`](sglang/), then point `SGLANG_PYTHON` at its interpreter. The `shape` step is SGLang-only; vLLM loads the raw architecture directly.

The **vLLM** backend likewise needs one file fetched: `block_causal_readout.py`, the block-end readout kernel, which is PolyForm-Noncommercial upstream. Follow [`vllm/vllm_native_diffusion/KERNELS.md`](vllm/vllm_native_diffusion/KERNELS.md) once; until it is present the plugin will not import.

## Tests

```bash
python test_smoke.py           # no GPU: checks command construction + configs
```

## Roadmap / contributing

A **foundation to build on** — bring your model, play, contribute. Open directions:

- packaging: pin the SGLang backend dependency (currently the `SGLANG_PYTHON` seam)
- prefer server-parsed `content`/`reasoning_content` in `chat.py`/`eval.py` when `--reasoning-parser` is set
- load-balanced multi-worker / multi-node dispatch for large eval runs
- new decode modes; more model families; throughput & memory tuning
- slim / refactor (e.g. compact the vendored IFEval scorer)
- (stretch) a from-scratch minimal engine

PRs and issues welcome. See [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md) before wiring a new checkpoint.

## License

Apache-2.0 — see [`LICENSE`](../LICENSE). Bundled third-party components retain their own licenses; see [`NOTICE`](NOTICE).
