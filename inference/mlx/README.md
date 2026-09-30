# inference/mlx — on-device self-speculative decoding (Apple Silicon)

Runs **[`trillionlabs/Trida2.0-4B`](https://huggingface.co/trillionlabs/Trida2.0-4B)** on a Mac with
**MLX**, in the same three pieces as the GPU stacks next door — but in-process, one stream, no CUDA:

| | file | what |
|---|---|---|
| decoder | `trida_mlx/model.py`, `trida_mlx/decode.py` | `causal` (AR baseline) and **`self-spec`** (diffusion draft + exact AR verify, lossless) |
| server | `trida_mlx/server.py` | OpenAI-compatible `/v1/chat/completions` (+ `/v1/completions`), SSE streaming, **tool calls**, reasoning split, prompt cache |
| agent | `trida_mlx/agent.py` | a tiny tool-using agent loop against the server (files, search, calculator; opt-in shell/write) |
| tools | `convert.py`, `verify.py`, `bench.py` | quantize to MLX, first-contact correctness check, causal-vs-self-spec benchmark |

Everything is built on `mlx_lm.models.qwen3_5` (weights, quantized layers, RoPE and the Metal
gated-delta kernel come from mlx-lm unchanged); this package only adds the self-spec forward,
the cache commit logic and the serving layer. It follows the approach of the `ocr-monorepo`
MLX self-spec OCR decoder, extended to the **hybrid** Qwen3.5 backbone.

## Quickstart

Run these one by one (zsh does not treat `#` as a comment in interactive shells, so don't paste
trailing comments). If another venv is active, `deactivate` first.

```bash
cd inference/mlx
uv sync
uv run hf auth login
uv run trida-mlx-convert --model trillionlabs/Trida2.0-4B --out ~/models/Trida2.0-4B-mlx-q8 --bits 8
uv run trida-mlx-verify --model ~/models/Trida2.0-4B-mlx-q8
uv run trida-mlx-bench --model ~/models/Trida2.0-4B-mlx-q8 --max-tokens 512 --out bench_q8.json
uv run trida-mlx-server --model ~/models/Trida2.0-4B-mlx-q8 --port 8080
uv run trida-mlx-agent --workdir ~/some/project "summarize what this repo does"
```

The console scripts are aliases for `python -m trida_mlx.<server|agent|convert|verify|bench>`.
Without uv: `pip install -e .` (or `pip install -r requirements.txt` and run from this directory).

`uv sync` makes a Python 3.12 `.venv` from `uv.lock`. `convert` resolves the private checkpoint
(needs `hf auth login`), `verify` checks canvas rows == AR logits and greedy self-spec == greedy AR,
`bench` compares causal vs self-spec (add `--temperature 1` for sampled), then serve and point the
agent (second terminal) at it. q8 is ≈4.5 GB, bf16 ≈8 GB; on 16–18 GB Macs use q8 or q6.

Any OpenAI client works against the server:

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="local")
r = c.chat.completions.create(model="trida", messages=[{"role": "user", "content": "hi"}],
                              extra_body={"chat_template_kwargs": {"enable_thinking": False}})
print(r.choices[0].message.content, r.usage)   # usage.trida has tok/s, tok/forward, accept histogram
```

Server flags: `--mode self-spec|causal`, `--gen-block N` (canvas `2N-1`, default 4 → the shipped
`b7/g4` config), `--temperature/--top-k/--top-p` defaults (1.0 / 50 / 0.95, the canonical Trida
self-spec setting), `--no-think`, `--max-tokens`, `--no-prompt-cache`. Per request: `trida_mode`,
`chat_template_kwargs`, `tools`, `stop`, `seed`, `stream`.

## How self-spec maps onto MLX

Reference semantics are the SGLang `HybridDiffusionSelfSpec` (`bd_bidir_shift`, block 7, gen 4,
`causal_prefill`, `draft_mode: strict_truncated`) — the same algorithm the vLLM plugin ports.

```
cold start : [t0,  M,  M,  M,  M,  M,  M ]   clean = row 0, specs = rows 1..3
verify     : [p,  s0, s1, s2,  M,  M,  M ]   s_i checked against row i (exact AR logits)
                                            all accepted -> 4 tokens: s0 s1 s2 + clean(row 3), new specs = rows 4..6
                                            reject at i  -> s_<i + correction; next step is a cold start
```

| piece | reference (SGLang / vLLM) | here |
|---|---|---|
| full attention (8 layers) | custom block mask: rows `0..N-1` causal, MASK rows see the whole canvas | explicit boolean mask to `mx.fast.scaled_dot_product_attention`; KV appended then trimmed to the accepted length |
| gated-delta (24 layers) | `causal_mode=2, num_clean=N`: clean rows read `h[t]`, MASK rows read `h[block_end]` | two calls of mlx-lm's Metal GDN kernel (clean rows from the committed state, then the MASK rows) + a block-end readout `S_end·q` |
| state commit | intermediate-state ring, scatter the state after the accepted prefix | keep the canvas' clean-row q/k/v/g/β; commit = state after `N` rows (free) or one tiny kernel over the accepted `adv` rows; conv window sliced from the saved pre-conv rows |
| verify / sampling | greedy argmax, or `min(1,p/q)` accept + `norm(max(0,p−q))` correction on top-k/top-p truncated p, q | same, on-GPU, one host sync per forward |

Nothing a forward computes is persisted until `commit(adv)`, so the verify rows are *exactly* the
causal computation — greedy output equals `causal` output (see the bf16 note below).

## Using it from Hermes Agent

Hermes talks to any OpenAI-compatible endpoint; the server advertises `context_length` on
`/v1/models` (Hermes requires ≥ 64K; default `--context-length 65536`) and returns a
`context_length_exceeded` 400 that Hermes turns into compaction.

```bash
# terminal 1
uv run trida-mlx-server --model ~/models/Trida2.0-4B-mlx-q8 --port 8080
# terminal 2 (one-time config)
hermes config set model.provider custom
hermes config set model.base_url http://127.0.0.1:8080/v1
hermes config set model.default trida2.0-4b
hermes config set model.api_key local
hermes -t terminal,file          # a small toolset: ~4.6K prompt tokens instead of ~11K
```

What the server does for Hermes specifically:
- renders Hermes histories into the Trida template (`developer` → system, leading system messages
  merged, later system nudges → a user `[System note]`, JSON-string tool arguments → dicts,
  `reasoning` replayed as `reasoning_content`);
- `reasoning_effort: none|minimal` (or `enable_thinking: false`) turns thinking off;
- tool calls come back as OpenAI `tool_calls`, with parameters typed from each tool's schema;
- SSE `: prefill n/m` keep-alive comments during long prefills;
- **3 cache slots**, so Hermes' side requests (titles, summaries) don't evict the main loop, and
  a snapshot after the system block, so even a rewritten user turn reuses the ~5–11K-token
  system + tools prefix;
- all MLX work runs on one worker thread (MLX streams are per-thread); requests are queued.
- `--log-requests DIR` dumps every request body for debugging; each request logs one line
  (prompt, cached, prefill s, tok/s, tok/fwd).

### Prompt cache (what makes the agent loop fast)

Each agent turn re-sends the whole conversation. Gated-delta layers can't rewind, so each cache
slot keeps resumable points: the live cache after the last generation, a snapshot right after the
system block, and one just before the last `<|im_start|>` of the previous prompt (where the next
turn's re-rendered history still agrees with it). A follow-up turn then only prefills the new messages; `usage.prompt_tokens_details.cached_tokens`
shows the reuse.

## Tests

```bash
uv run pytest                 # all offline tests (~30 s)
uv run python tests/test_tiny.py   # decoder invariants on a tiny random Qwen3.5-hybrid (CPU or GPU)
uv run python tests/test_e2e.py    # tiny raw-HF checkpoint: loader, convert q8, prompt cache, server, parsing
```

Pinned: chunked prefill = one-shot; canvas clean rows = AR logits; MASK rows read the block-end GDN
state (independent per-token re-derivation); `commit(adv)` = AR state for every `adv`; greedy
self-spec = greedy AR (also with oracle drafts that exercise the accept path, ~3 tok/forward);
top-k/top-p matches the reference sampler; sampled self-spec reproduces the AR sampling
distribution; snapshot/restore = fresh prefill; server non-stream / SSE / completions round trip.
On a Mac the same tests run on the GPU and therefore also cover the Metal kernel path.

## Notes and limits

- **bf16/quantized numerics.** The canvas runs 7-row matmuls and the AR step 1-row matmuls, which
  are different kernels; at a near-tie the greedy argmax can flip. `verify.py` prints the AR
  top-1/top-2 margin at the first divergence so you can tell numerics (tiny margin) from a bug.
- **Quantization.** Self-spec is lossless relative to *the model it runs*; quantization changes that
  model. In the OCR study q8/q6 kept outputs identical to bf16 and q4 did not; draft acceptance can
  also drop at low bit widths. `convert.py --keep-embed` keeps the tied LM head in bf16.
- **Single stream.** Requests are serialized (one device); no batching.
- **Step cost.** `trida-mlx-profile` splits the AR step and the canvas step into LM head / MLPs /
  attention / gated-delta and prints the break-even tokens-per-forward. Done: a fused canvas GDN
  Metal kernel (`trida_mlx/kernels.py`, one launch per layer; `--no-fused-gdn` to A/B): verify step
  49.8 → 44.5 ms on M3 Pro q8, i.e. 1.13× an AR step (39.3 ms). LM-head row skipping on cold starts
  is available (`bench --skip-rows`) but measured slower, so it is off.
- The GGUF quants (`mradermacher/Trida2.0-4B-*GGUF`) run in llama.cpp as plain AR only; self-spec
  there would need the canvas mask and the block-end GDN readout in llama.cpp's qwen35 graph.
