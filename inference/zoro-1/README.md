<div align="center">

# Zoro-1 nano-inference

**A minimal, hackable inference stack for one model: [`trillionlabs/Zoro-1`](https://huggingface.co/trillionlabs/Zoro-1).**
Self-speculative generation and typed decisions from one set of weights, behind an OpenAI-compatible API, in a handful of readable files.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Backends](https://img.shields.io/badge/backends-SGLang%20%C2%B7%20vLLM%20%C2%B7%20MLX-8A2BE2.svg)](#backends)
[![Speedrun](https://img.shields.io/badge/speedrun-H100%20%C2%B7%20L40S%20%C2%B7%20MLX-orange.svg)](#speedrun)

[Speedrun](#speedrun) · [Quickstart](#quickstart) · [Test it](#test-it) · [Usage](#usage) · [Decisions](#decisions) · [Backends](#backends) · [Contributing](#contributing)

</div>

---

In the spirit of nanoGPT, this directory serves **one model** and nothing else, so every line can be read with that model in mind. Zoro-1 is a 4B two-stream block-diffusion model (Qwen3.5-4B, trained with this repo's [`train/`](../../train/)) that is both an agent model and a decision model:

| `--mode` | what it does |
|---|---|
| `causal` | native autoregressive decoding — the reference |
| **`self-spec`** (default) | the diffusion stream drafts a block, the model's own AR stream verifies it in the same forward pass: generation block 8 (`b15_g8`), several tokens per forward, the AR model's output distribution |
| `decide` | typed decisions (`/v1/decide`: yes/no, choice, score with probabilities), reason-then-decide, and OpenAI chat from one engine ([Decisions](#decisions)) |

Zoro-1 does not support pure iterative diffusion decoding, so there is no `diffusion` mode. Other models get their own nano-inference next to this one (`inference/<model>/`).

## Speedrun

Like the [NanoGPT speedrun](https://github.com/KellerJordan/modded-nanogpt): one fixed task, a leaderboard per hardware track, and anyone can take a record.

**The task:** answer a fixed GSM8K subsample with Zoro-1, one request at a time, as fast as possible.

- **Record = wall-clock** of `eval.py` (`wall_s` in its `summary.json`). Lower is better.
- **Accuracy floor: 87.0%.** Any change counts — kernels, configs, decode tricks, lossy ones too — as long as accuracy stays at or above the floor.
- **Fixed:** model `trillionlabs/Zoro-1` (weights unchanged), client concurrency 1, temperature 1.0 / top-p 0.95 / top-k 20, thinking on, 8192-token budget, no failed requests.
- **Problems:** 200 on the GPU tracks, 50 on MLX (`eval.py --num-problems N` takes the same seed-42 subsample every time).

### 1× NVIDIA H100

![H100 speedrun](speedrun/h100.svg)

<!-- speedrun:h100 -->
| # | Wall-clock | GSM8K | tok/s per request | Description | Date | Log | Contributors |
|---|---|---|---|---|---|---|---|
| – | no record yet: be the first | | | | | | |
<!-- /speedrun:h100 -->

### 1× NVIDIA L40S

![L40S speedrun](speedrun/l40s.svg)

<!-- speedrun:l40s -->
| # | Wall-clock | GSM8K | tok/s per request | Description | Date | Log | Contributors |
|---|---|---|---|---|---|---|---|
| – | no record yet: be the first | | | | | | |
<!-- /speedrun:l40s -->

### Apple Silicon (MLX)

Say the chip and memory in the description (e.g. "M3 Pro 18 GB, q8").

![MLX speedrun](speedrun/mlx.svg)

<!-- speedrun:mlx -->
| # | Wall-clock | GSM8K | tok/s per request | Description | Date | Log | Contributors |
|---|---|---|---|---|---|---|---|
| – | no record yet: be the first | | | | | | |
<!-- /speedrun:mlx -->

### Submit a record

```bash
python serve.py --port 30000 --log server.log &                    # or vllm/serve_diffusion.sh, or the MLX server
python eval.py gsm8k --num-problems 200 --max-workers 1 --server-log server.log   # 50 on MLX
python speedrun/leaderboard.py add results/gsm8k_<stamp>/summary.json --track h100 \
    --description "what you changed, in one line" --log <link to summary.json + requests.jsonl> --contributors @you
```

`add` checks the run against the rules above and that it beats the track's record, appends it to `speedrun/records/<track>.csv`, and redraws the table and chart. Open a PR with your change, the new row, and the run's `summary.json` + `requests.jsonl` (attach them or link them). A maintainer re-runs it on the same hardware class before merging. CI fails if the README tables or charts drift from the record files (`python speedrun/leaderboard.py check`).

## Quickstart

```bash
# 1. install client deps + point at your serving backend (see Backends below)
source env.example.sh              # sets SGLANG_PYTHON + cache dirs
pip install -r requirements.txt

# 2. serve Zoro-1 (gated on Hugging Face: request access, then `hf auth login`)
python serve.py --port 30000                 # default mode: self-spec

# 3. call it (OpenAI-compatible)
python chat.py "What is 17 times 24? Answer with just the number."
```

That's it — an OpenAI-compatible endpoint at `http://localhost:30000/v1/chat/completions`. `--mode causal` serves the *same weights* autoregressively; `--mode decide` serves decisions; `--dry-run` prints the launch command without starting a server.

### Model access

The model is resolved as **positional arg → `$TRIDA_MODEL` → `trillionlabs/Zoro-1`** (for `decide`, `$TRIDA_DECIDE_MODEL` first), and the same rule applies to the vLLM script (`CKPT` → `$TRIDA_MODEL` → default) and the notebook. Zoro-1 is **gated** on Hugging Face: request access on its model page, then authenticate —

```bash
hf auth login            # or: export HF_TOKEN=hf_...
```

A `401` / `Repository not found` from the backend at startup means the token is missing or the account has no access yet. A local checkpoint directory needs no token.

## Test it

Three levels, from no hardware to the full stack:

1. **No model, any machine** — wiring, request encoding, decision head, chat parsing, the speedrun checker:
   ```bash
   pytest -q            # from the repo root; CI runs the same
   ```
2. **Apple Silicon, real generation** — the MLX backend runs Zoro-1 on a Mac (q8 fits 16–18 GB). `verify` checks that self-spec reproduces greedy AR exactly:
   ```bash
   cd mlx && uv sync
   uv run trida-mlx-convert --model trillionlabs/Zoro-1 --out ~/models/Zoro-1-mlx-q8 --bits 8
   uv run trida-mlx-verify  --model ~/models/Zoro-1-mlx-q8
   uv run trida-mlx-server  --model ~/models/Zoro-1-mlx-q8 --gen-block 8 --port 8080
   ```
   Decisions on MLX are not supported yet.
3. **GPU, everything** — serve each mode and compare with the model card (GSM8K 89.9, MMLU-Pro 62.7 self-spec):
   ```bash
   python serve.py --mode causal    --port 30000 &   python eval.py gsm8k --num-problems 200
   python serve.py --mode self-spec --port 30000 &   python eval.py gsm8k --num-problems 200
   python serve.py --mode decide    --port 30001      # then the /v1/decide example under Decisions
   ```

## Usage

**Serve** — two decode modes over one checkpoint, plus `decide`:

```bash
python serve.py --port 30000                       # trillionlabs/Zoro-1 (or $TRIDA_MODEL), self-spec (default)
python serve.py --mode causal    --port 30000
python serve.py /path/to/local/zoro-1 --port 30000   # a local copy
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

From the *same* requests you get **accuracy**, **tok/s** (aggregate at the client concurrency `--max-workers`, and per request), **latency** p50/p90/mean, mean output length, and — with `--server-log` — **tokens per forward** (vLLM: from its `DiffusionDecoding` counters; SGLang self-spec: from the algorithm's stats line; SGLang causal logs no counter → `n/a`; causal is 1 by definition). Every request is written to `requests.jsonl` (prediction, gold, tokens, latency) so you can re-slice without re-running. For a per-stream latency profile use `--max-workers 1`; for aggregate throughput raise it. Keep `--max-tokens` a *budget*, not a knob: if the truncation percentage is not ~0, the number measures the budget, not the model. On the vLLM script use `LOG=server.log bash vllm/serve_diffusion.sh`.

Defaults: temp 1.0 / top_p 0.95 / top_k 20, no presence penalty, thinking on, `--max-tokens 8192` — Zoro-1's self-spec setting (the model card's), identical to the SGLang self-spec config, so a number from `eval.py` means the same thing on either backend. Both backends honor these fields in self-spec (vLLM via its stock rejection sampler, `TRIDA_SS_SAMPLE=0` for exact greedy). `gsm8k`/`mmlu_pro` are self-contained (extraction-based); **ifeval** uses the Google Research IFEval scorer vendored in `ifeval_lib/` (Apache-2.0). `chat.py`/`eval.py` pick the served model id up from `/v1/models`, so they work unchanged against either backend.

## Decisions

`serve_decide.py` serves a **decision checkpoint**: one that carries a pointer head (`jev_head.pt`, see [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md#decision-checkpoints-serve_decidepy)). It answers typed questions about a state with calibrated probabilities. It can also reason first and then decide, and it chats with tool calls. All three run from one in-process engine and one copy of the weights, on **stock SGLang ≥ 0.5.22** with causal decoding. No fork or patch is needed.

```bash
pip install "sglang>=0.5.22" fastapi uvicorn             # a stock SGLang env, separate from the diffusion backend's
python serve.py --mode decide --port 30001
#   = python serve_decide.py /path/to/decision-checkpoint --port 30001
#   serve.py runs it under $DECIDE_PYTHON (default: the current python); $TRIDA_DECIDE_MODEL can name the checkpoint
```

```bash
curl -s localhost:30001/v1/decide -H 'content-type: application/json' -d '{
  "request": {
    "state": {"ticket": "Order #W1 arrived broken. I want my money back.", "order": {"status": "delivered", "total": 42.0}},
    "questions": {
      "intent":  {"type": "choice", "instructions": "What does the customer want?",
                  "criteria": {"refund": "money back", "exchange": "a replacement", "info": "just information"}},
      "urgent":  {"type": "noul", "instructions": "Should this jump the queue?"},
      "anger":   {"type": "score", "instructions": "How upset is the customer?", "criteria": ["calm", "annoyed", "angry"]}}},
  "reasoning": false}'
# -> {"answers": {"intent": {"type": "choice", "choice": "refund", "confidence": 0.9794,
#                             "probabilities": {"refund": 0.9863, "exchange": 0.011, "info": 0.0027}},
#                 "urgent": {"type": "noul", "noul": 0.9256},
#                 "anger":  {"type": "score", "score": 1.6872, "legend": ["calm", "annoyed", "angry"], ...}}, ...}
```

- **Question types:**
  - `noul`: yes/no; the answer is P(yes).
  - `choice`: `{key: description}`.
  - `score`: a list of levels, lowest first; the answer is the expected level.
- **`"reasoning": true`:** the model first writes its reasoning (returned as `reasoning`), then decides with that reasoning added to the state.
- **Images:** add an `image` (path, base64 or data URL) to the request.
- **`POST /v1/chat/completions`:** OpenAI-compatible, streaming (`"stream": true`) or not.
  - It takes `tools`; tool calls come back as `tool_calls`.
  - The thinking comes back as `reasoning_content` (streamed as `reasoning_content` deltas before the content).
  - It honours `max_tokens`, `temperature`, `top_p`, `top_k`, `presence_penalty` and `chat_template_kwargs.enable_thinking`.
  - Responses carry `usage`, and `finish_reason` is `"length"` when `max_tokens` cut the turn.
- **`GET /v1/models`:** the served model id, so `chat.py` and `eval.py` work against this server unchanged.
- **Concurrency:** decisions and chat share the engine and batch together. A decision sent while a long chat is generating returns in tens of milliseconds.
- **From Python:** `DecideEngine(ckpt).decide(...)` / `.chat(...)` / `.chat_stream(...)`, or the `async` versions (`adecide`, `achat`, ...) on `eng.loop` ([`decide_engine.py`](decide_engine.py)).

How it fits a stock engine: a request row is causal, so an option's hidden state is the last hidden state of the row cut off at that option. Each question with K options therefore becomes K+1 prefix prompts in one batched call with `return_hidden_states="last"`, and SGLang's prefix cache shares the work. The head then runs on those vectors. Checked against a transformers implementation of the same head on 200 held-out decision records, the outputs matched to within 0.1% mean probability difference.

## Backends

Two ways to run the decode kernels behind `serve.py`'s API — pick one.

### SGLang (wrapped by `serve.py`)

The default path. `serve.py` launches an SGLang server built from its public upstream. One-time setup: **clone → patch → install → shape** (see [`sglang/README.md`](sglang/README.md)). A raw two-stream checkpoint needs the one-time shaping step:

```bash
python sglang/tools/shape_for_sglang.py <raw_ckpt>   # -> <raw_ckpt>_sglang
export SGLANG_PYTHON=/path/to/backend-venv/bin/python
python serve.py <raw_ckpt>_sglang --port 30000
```

### vLLM (native plugin, no fork)

A **self-contained** serve inside a stock **vLLM 0.27.x** install — no fork, no SGLang, no shaping:

```bash
pip install vllm==0.27.*
pip install -e vllm                       # registers the `trida_diffusion` plugin
# one-time: fetch the block-end readout kernel (PolyForm-NC upstream, not vendored)
#   -> vllm/vllm_native_diffusion/KERNELS.md
bash vllm/serve_diffusion.sh                              # trillionlabs/Zoro-1 (or $TRIDA_MODEL), self-spec
MAX_NUM_SEQS=8 bash vllm/serve_diffusion.sh               # batched self-spec (PIECEWISE cuda graphs)
MODE=causal bash vllm/serve_diffusion.sh                  # AR reference; CKPT=<hf-id-or-local-path> for another checkpoint
```

Runs self-spec as an out-of-tree plugin (arch `Qwen3_5ForBlockDiffusion`): FULL cuda graphs for a single stream, PIECEWISE for batched serving, lossless in both. See [`vllm/README.md`](vllm/README.md) for env knobs and status.

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
python test_smoke.py              # no GPU: command construction, modes, the b15_g8 config
pytest -q test_decide_cpu.py      # no GPU: decision requests, prefix prompts, head, chat parsing, HTTP
pytest -q speedrun/               # the speedrun rules and the leaderboard rendering
```

## Contributing

The best way in is the [speedrun](#speedrun): make Zoro-1 faster on your hardware and take a record. Beyond that:

- prefer server-parsed `content`/`reasoning_content` in `chat.py`/`eval.py` when `--reasoning-parser` is set
- decisions on vLLM and MLX ([#60](https://github.com/trillion-labs/trida-stack/issues/60), [#61](https://github.com/trillion-labs/trida-stack/issues/61))
- packaging: pin the SGLang backend dependency (currently the `SGLANG_PYTHON` seam)
- slim / refactor (e.g. compact the vendored IFEval scorer)

Read [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md) for what the serving code expects from the checkpoint, and the repo's [`CONTRIBUTING.md`](../../CONTRIBUTING.md) for the PR process.

## License

Apache-2.0 — see [`LICENSE`](../../LICENSE). Bundled third-party components retain their own licenses; see [`NOTICE`](NOTICE).
