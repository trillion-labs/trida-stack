# inference/mlx — on-device self-speculative decoding (Apple Silicon)

Runs **[`trillionlabs/Trida2.0-4B`](https://huggingface.co/trillionlabs/Trida2.0-4B)** and the
vision-language **[`trillionlabs/Trida-2.0-4B-1006`](https://huggingface.co/trillionlabs/Trida-2.0-4B-1006)**
on a Mac with **MLX**, in the same three pieces as the GPU stacks next door — but in-process, one
stream, no CUDA:

| | file | what |
|---|---|---|
| decoder | `trida_mlx/model.py`, `trida_mlx/decode.py` | `causal` (AR baseline) and **`self-spec`** (diffusion draft + exact AR verify, lossless) |
| server | `trida_mlx/server.py` | OpenAI-compatible `/v1/chat/completions` (+ `/v1/completions`), SSE streaming, **tool calls**, reasoning split, prompt cache |
| agent | `trida_mlx/agent.py` | a tiny tool-using agent loop against the server (files, search, calculator; opt-in shell/write) |
| decisions | `trida_mlx/decision.py`, `trida_mlx/jev.py` | `/v1/decide`: typed decisions (choice / yes-no / score) through Zoro-1's pointer head, escalating to self-spec reasoning when unsure |
| vision | `trida_mlx/vision.py` | Qwen3.5 vision encoder + projector, image preprocessing, multimodal RoPE (checkpoints with a `vision_config`) |
| data | `trida_mlx/datamine.py` | `trida-mlx-datamine`: the app's chats + request logs → candidate training data (preference pairs, feedback, tool use, decisions, failures, trajectories), scrubbed |
| tools | `convert.py`, `verify.py`, `bench.py`, `profile.py` | quantize to MLX, first-contact correctness check, causal-vs-self-spec benchmark, step profile |

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

Server flags: `--mode self-spec|causal`, `--gen-block N` (canvas `2N-1`, default 4 → `b7_g4`; see
"Block size" below), `--temperature/--top-k/--top-p` defaults
(1.0 / 50 / 0.95, the canonical Trida self-spec setting), `--no-think`, `--max-tokens`,
`--no-prompt-cache`, `--max-image-pixels`. Per request: `trida_mode`, `chat_template_kwargs`,
`tools`, `stop`, `seed`, `stream`.

## Block size

Both checkpoints were trained with `bd_size` 8, and the 1006 model card's GPU self-spec config is
`b15_g8`. On a Mac the canvas is not free: M3 Pro 18 GB, q8 Trida-2.0-4B-1006, greedy, 512 tokens:

| N (canvas) | causal tok/s | self-spec tok/s | tok/forward | step vs AR step | speedup |
|---|---:|---:|---:|---:|---:|
| 4 (7) | 26.2 | **52.4** | 2.40 | 1.18× | **2.00×** |
| 8 (15) | 25.9 | 43.0 | 2.72 | 1.64× | 1.66× |

N = 8 accepts more per step but each step costs more than it gains, so the default is N = 4.

## Images (Trida-2.0-4B-1006)

Checkpoints in the vision-language layout (`Qwen3_5ForConditionalGeneration`, with `vision_config`)
load their Qwen3.5 vision encoder + projector automatically; `convert.py` quantizes the language
model and keeps the vision tower in bf16. Images go through the server as OpenAI content parts
(`{"type": "image_url", "image_url": {"url": "data:image/png;base64,..." | "https://..." | "/path"}}`):

- preprocessing = Qwen2-VL smart resize to multiples of 32 px, normalize, 16×16×2 patches
  (bit-identical to the HF `Qwen2VLImageProcessor` PIL path); `--max-image-pixels` (default
  1024×1024 → ≤ 1024 image tokens) caps the token cost on device;
- the template's `<|image_pad|>` is expanded to one token per merged 2×2 patch, image rows get the
  vision features, and the prompt uses interleaved multimodal RoPE (t/h/w positions); text after an
  image continues from `max(t, h, w)`, so decoding and the self-spec canvas stay plain RoPE;
- the prompt cache keys image tokens by a hash of the image, so a different image never reuses a
  cache, and a follow-up turn about the same image does not re-encode it;
- checked offline against mlx-vlm's Qwen3.5 implementation: same positions, last-token logits
  within 4e-6 in float32 on a random tiny model.

`verify.py --image photo.jpg` runs the lossless/speed checks on an image prompt. In Hermes, set
`hermes config set model.supports_vision true` so images are sent natively.

## Decisions (Zoro-1 pointer head)

[`trillionlabs/Zoro-1`](https://huggingface.co/trillionlabs/Zoro-1) ships `jev_head.pt`, a small pointer
head that answers typed questions about a state without generating text. The server loads it when the
file sits next to the weights (`convert.py` carries it; for a folder converted earlier,
`hf download trillionlabs/Zoro-1 jev_head.pt --local-dir ~/models/Zoro-1-mlx-q8`) and then serves
generation and decisions from the same model:

```bash
curl -s localhost:8000/v1/decide -d '{
  "state": {"ticket": "the printer on floor 3 is jammed again"},
  "questions": {"team": {"type": "choice", "instructions": "Which team handles it?",
                         "criteria": {"it": "IT support", "facilities": "Facilities"}}}}'
```

- Request and prompt layout are `inference/jev.py`'s (vendored unchanged as `trida_mlx/jev.py`; a test
  pins it): one causal row per question, `state` first, the pointer head scores each option's `</opt>`
  hidden state against the `<decide>` one. The state is prefilled once and every question branches
  from a snapshot of it, so N questions cost one state prefill plus N short branches.
- `image` (data URL, URL or path) puts the picture in the state; `max_image_pixels` defaults to the
  head's training size (448×448).
- Temperatures: text 1.52 (the Zoro-1 card), image 1.0 (not calibrated yet); override with
  `--decision-text-temperature` / `--decision-image-temperature` or a per-request `temperature`.
- **Escalation** (`"reasoning": "auto"`, the default): if any question's top probability is below
  `threshold` (0.7), the model reasons once with thinking on, in the request's `trida_mode`
  (self-spec by default), up to `max_reasoning_tokens` (8192), stopping at `</think>`; the reasoning is
  added to the state and it decides again. Reasoning cut off by the budget is never used: the
  direct answer stands and `cut` says why. `"always"` / `"never"` force either path.
- Response: `answers` (the ones to use), `direct`, `escalated`, `reason`, and when it reasoned
  `reasoning` plus `reasoned`. `"stream": true` sends SSE events `direct`, `reasoning` (text pieces)
  and `result`.
- The model card (`/v1/models`) reports `decision: true|false`; without a head, `/v1/decide` returns 400.

## How self-spec maps onto MLX

Reference semantics are the SGLang `HybridDiffusionSelfSpec` (`bd_bidir_shift`, block `2N-1`,
gen `N`, `causal_prefill`, `draft_mode: strict_truncated`) — the same algorithm the vLLM plugin
ports. Shown for N = 4:

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
`/v1/models` (Hermes requires ≥ 64K; default `--context-length 131072`) and returns a
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
uv run pytest tests/test_vision.py  # tiny multimodal checkpoint: image input end to end
uv run pytest tests/test_decision.py  # tiny multimodal checkpoint + random pointer head: /v1/decide
```

Pinned: chunked prefill = one-shot; canvas clean rows = AR logits; MASK rows read the block-end GDN
state (independent per-token re-derivation); `commit(adv)` = AR state for every `adv`; greedy
self-spec = greedy AR (also with oracle drafts that exercise the accept path, ~3 tok/forward);
top-k/top-p matches the reference sampler; sampled self-spec reproduces the AR sampling
distribution; snapshot/restore = fresh prefill; server non-stream / SSE / completions round trip;
with an image: greedy self-spec = greedy AR, chunked prefill = one-shot, MRoPE positions, no prompt-cache
reuse across different images, convert keeps the vision tower bit-identical, server `image_url` round trip.
Decisions: the vendored jev.py equals inference/jev.py; branching from the shared state snapshot gives the
same option logits as each question's full row (text and image states); escalation only when unsure,
never on cut-off reasoning; server `/v1/decide` plain and streamed.
On a Mac the same tests run on the GPU and therefore also cover the Metal kernel path.

## Notes and limits

- **Loop guard (server, `--loop-guard`, off by default; Zoro-1 Chat turns it on).** A reply that ends in one block of up to 256 tokens repeated 6 times in a row
  (at least 96 tokens), such as the same tool call emitted until `max_tokens`, is stopped there
  (`usage.trida.finish_reason: "repetition"`, OpenAI `finish_reason: "stop"`). It also stops looser loops: when at least half of the last 1,536 tokens end a 16-token phrase already
  written in that window (re-deriving the same steps of a hard problem over and over). If that happens while
  the model is still thinking, the server closes the thinking itself ("I keep going in circles…" +
  `</think>`) and lets the model answer from what it has (`usage.trida.loop_stopped_thinking: true`); a
  decision's reasoning that loops counts as cut off (the direct answer is used). Off by default so evals and
  speedrun runs decode unguarded and stay comparable; `/v1/models` reports `loop_guard` and `tool_call_gate`
  so a run can record what it used. `verify.py`, `bench.py` and the tests decode without it.
- **Tool-call gate** (`--tool-call-gate P`, off by default; Zoro-1 Chat uses 0.7; per request
  `tool_call_min_prob`). With tools,
  `<tool_call>` may only be generated where its raw probability (temperature 1, before top-k/top-p) is at
  least 0.7. Temperature 0.6 and top-k sharpen a single-token option against text answers spread over many
  first tokens: in a logged translation request a raw 0.041 became 0.29 after sampling settings, and the
  model called `calculator("1")`. The gate is part of the target distribution, so self-spec stays exact.
- **`tool_choice: "none"`** keeps the tools in the rendered prompt (so the prompt cache still matches the
  turns that may call them) and bans the `<tool_call>` token for that reply, in AR and self-spec alike.

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
  is available (`bench --skip-rows`) but measured slower, so it is off. (N = 4.)
- **Tool calls.** The Trida-2.0-4B-1006 model card reports lower BFCL scores for self-spec than AR on
  the SGLang path (unclosed `</tool_call>`). Here greedy self-spec reproduces AR token for token and
  sampled self-spec keeps the AR distribution, so that gap should not appear — but measure it; and
  `--mode causal` (or `"trida_mode": "causal"` per request) is the fallback.
- **Vision scope.** Still images only (no video), no deepstack features (the released checkpoints
  have none); images inside system or assistant messages are dropped by the chat template contract.
- The GGUF quants (`mradermacher/Trida2.0-4B-*GGUF`) run in llama.cpp as plain AR only; self-spec
  there would need the canvas mask and the block-end GDN readout in llama.cpp's qwen35 graph.
