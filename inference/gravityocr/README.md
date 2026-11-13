<div align="center">

# GravityOCR nano-inference

**Crop-level document OCR with [`trillionlabs/GravityOCR`](https://huggingface.co/trillionlabs/GravityOCR), served with self-speculative decoding on upstream SGLang.**
One layout region per request, behind an OpenAI-compatible API: a launcher, a one-crop client and a pinned SGLang patch.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/backend-SGLang%20v0.5.10-8A2BE2.svg)](#install)
[![GPU](https://img.shields.io/badge/GPU-H100-76B900.svg)](#requirements)

[Requirements](#requirements) · [Install](#install) · [Serve](#serve) · [Verify](#verify) · [Requests](#request-format) · [What self-spec reproduces](#what-self-spec-reproduces) · [Security](#security) · [Known limits](#known-limits) · [Tests](#tests)

</div>

---

GravityOCR is [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) (a CogViT vision encoder and a 0.5B GLM text decoder) fine-tuned as a block-diffusion model: the text decoder learns to fill a block of 32 masked tokens in parallel, and an AR auxiliary loss keeps its causal head. The architecture is unchanged, so the checkpoint loads as a stock `GlmOcrForConditionalGeneration`. The weights were trained outside this repo; this folder only serves them.

Self-speculative decoding uses both heads of that one checkpoint. Each round runs two forward passes. The **draft** pass reads the boundary token followed by 32 masks, bidirectional within that window, and proposes the next block. The **verify** pass runs the causal path over the draft, accepts the longest prefix the AR head agrees with (its greedy choices; with sampling, speculative sampling against its distribution), and adds the AR head's own next token. A round commits up to K + 1 = 34 tokens (K = `DRAFT` = 33). [What self-spec reproduces](#what-self-spec-reproduces) says exactly how the result relates to causal decoding.

| `MODE` | what it does |
|---|---|
| **`self-spec`** (default) | block-diffusion draft, AR verify, inside SGLang's speculative-decoding loop (`--speculative-algorithm SELFSPEC_DIFFUSION --speculative-num-draft-tokens 33 --page-size 1`) |
| `causal` | the same weights decoded autoregressively, one token per forward pass: the reference |

The model reads **one layout region per request**: a text block, a table or a formula crop. Page layout detection and page assembly are out of scope for this folder; a full page needs a layout detector in front of the server.

This is the second nano-inference next to [`../zoro-1/`](../zoro-1/). The two do not share a backend. This one is upstream SGLang v0.5.10 plus an Apache-2.0 patch; zoro-1's SGLang backend is a different, noncommercial-licensed fork. Build each in its own environment.

## Requirements

- **Linux x86_64**, `git` and [`uv`](https://docs.astral.sh/uv/). Install and first start need network access to GitHub, PyPI, the PyTorch package index and the Hugging Face Hub. The model is public; no token is needed.
- **An NVIDIA Hopper GPU (compute capability 9.0).** Validated on H100 80 GB with driver 570 (CUDA 12.8). The FlashInfer and FA3 attention paths the patch uses were validated only there, and `serve.sh` refuses other GPUs unless `SKIP_PREFLIGHT=1`.
- **A CUDA 12.x toolkit at `CUDA_HOME`** (default `/usr/local/cuda-12.8`). FlashInfer and Triton compile kernels when the server starts, so `nvcc` must be present. Use a toolkit no newer than your driver supports.
- **Python 3.12** (uv fetches it if needed).
- **A separate environment.** The server venv lives inside your SGLang checkout and is used only by `serve.sh`. Do not install it into the repo's root environment or into zoro-1's backend environment; the pins clash. `sglang/requirements-lock.txt` pins, among others: torch 2.9.1 (the PyPI build, CUDA 12.8 runtime) with torchvision, torchaudio, torchao and torchcodec as `+cu128` builds from the PyTorch cu128 index (they must match torch's CUDA 12.8: torchaudio refuses to import otherwise), transformers 5.3.0, flashinfer-python 0.6.7.post2 and sglang-kernel 0.4.1, and SGLang's build backend.

## Install

```bash
SGLANG_SRC=$HOME/src/sglang-gravityocr bash inference/gravityocr/sglang/setup.sh
```

`SGLANG_SRC` must be outside this repository and be missing, empty, or an SGLang checkout. `setup.sh` is safe to re-run; a finished build is only re-verified. It:

1. checks for `git`, `uv` (the lock was generated with 0.11.27) and a CUDA 12.x `nvcc` at `CUDA_HOME`. `SKIP_PREFLIGHT=1` skips the `nvcc` check, for example to build on a machine without the toolkit;
2. fetches exactly the pinned base commit of [`sgl-project/sglang`](https://github.com/sgl-project/sglang) from `sglang/BASE_COMMIT.txt` (`1519acf37c23f2189adb93f57ca9cd2db1bebf18`, tag v0.5.10) into an empty `SGLANG_SRC` with a shallow fetch, and verifies that `HEAD` is that commit. In a clean SGLang checkout at another commit it fetches the base commit without making the clone shallow and checks it out as a detached `HEAD`, leaving the branches alone. A checkout with local changes, or a git repository that is not SGLang, is refused;
3. applies `sglang/gravityocr_selfspec.patch` (skipped when already applied), copies the seven new files from `sglang/python/` into the checkout, and then checks the checkout byte for byte: every tracked file equals the base commit with the patch applied, the new files equal the committed ones, and `python/` holds no other untracked file (build output such as `__pycache__/` aside). Any other difference stops the install and is listed;
4. creates `$SGLANG_SRC/.venv-sglang` (Python 3.12) and installs `sglang/requirements-lock.txt` as is, with `--no-deps --no-build` and nothing re-resolved: the PyPI pins first, from `https://pypi.org/simple` only, then the four `+cu128` builds from the PyTorch cu128 index only. uv config files and uv's index variables (`UV_INDEX`, `UV_EXTRA_INDEX_URL`, `UV_FIND_LINKS`, ...) are ignored. The pins carry no hashes;
5. installs the patched SGLang itself, editable, as version 0.5.10, offline and without build isolation: the build backend (setuptools, setuptools-scm, wheel and their dependencies) is the locked one from step 4, so the build fetches nothing;
6. checks the installed packages against the lock and prints the `VBIN=...` line that `serve.sh` needs.

The lock is reproducible from the repository: its header has the resolution command, which uses the committed `sglang/validated-freeze.txt` (the GPU-validated environment) as constraints and a fixed `--exclude-newer` date.

Other settings, as environment variables or `KEY=VALUE` arguments: `PY` (venv Python, default 3.12, the version the lock is resolved for), `CUDA_HOME` (default `/usr/local/cuda-12.8`) and `SGLANG_REPO` (the upstream URL, for a mirror). `--help` prints them.

## Serve

```bash
export VBIN=$SGLANG_SRC/.venv-sglang/bin                         # the line setup.sh printed
bash inference/gravityocr/serve.sh                               # self-spec, trillionlabs/GravityOCR, GPU 0, 127.0.0.1:30000
MODE=causal PORT=30001 bash inference/gravityocr/serve.sh        # autoregressive reference, same weights
CKPT=/models/my-gravityocr GPU=1 bash inference/gravityocr/serve.sh   # a local checkpoint on GPU 1
DP=2 GPU=0,1 bash inference/gravityocr/serve.sh                  # two data-parallel replicas
bash inference/gravityocr/serve.sh MODE=causal GPU=1             # KEY=VALUE arguments work too (not for API_KEY)
```

Before it starts the server, `serve.sh` prints one line, for example `[serve] MODE=self-spec FAST=1 GPU=0 DP=1 SERVE_HOST=127.0.0.1 PORT=30000 SERVED_NAME=glm-ocr CKPT=<local snapshot>`. The first start compiles FlashInfer and Triton kernels and captures CUDA graphs, so it is slower than later starts; poll `/v1/models` until it answers. In self-spec mode the server log also has a `[selfspec-diffusion] K=33 mask_id=59282 (from ...)` line. Check that the mask id comes from your checkpoint's `block_diffusion.json` ([`MODEL_CONTRACT.md`](MODEL_CONTRACT.md)). `bash inference/gravityocr/serve.sh --help` prints the main settings.

| env | default | meaning |
|---|---|---|
| `MODE` | `self-spec` | `self-spec` or `causal`; anything else exits with status 2. `causal` also drops any `SELFSPEC_*` variables inherited from your environment |
| `CKPT` | `$GRAVITYOCR_MODEL`, else `trillionlabs/GravityOCR` | local checkpoint directory or HF repo id. A repo id is downloaded first with `snapshot_download` (the only step with Hub access, `HF_HUB_OFFLINE=0`); the server then runs offline from the local snapshot. A value that looks like a path (`/`, `./`, `../`, `~`) but is not a directory exits with status 2; a failed download exits 1 |
| `VBIN` | *(required)* | `bin/` of the patched SGLang venv, as printed by `sglang/setup.sh` |
| `GPU` | `0` | `CUDA_VISIBLE_DEVICES` list, comma-separated, one device per replica: exactly `DP` entries (else exit 2) |
| `DP` | `1` | data-parallel replicas behind one port; above 1, passes `--dp-size $DP` |
| `PORT` | `30000` | server port |
| `SERVE_HOST` | `127.0.0.1` | bind address. A non-loopback address requires `API_KEY` (else exit 2). Named so that a `HOST` exported by your shell has no effect |
| `API_KEY` | *(unset)* | bearer token the server then requires on every endpoint except SGLang's `/health*` and `/metrics`. Every tokenizer worker checks it, so it works with any `TOKENIZER_WORKERS`. Printable ASCII without spaces, not starting with `-`. Set it in the environment; as a `KEY=VALUE` argument it is refused (exit 2). It reaches SGLang through a FIFO named by `--config`, never on a command line or in a file ([Security](#security)) |
| `SERVED_NAME` | `glm-ocr` | model id on `/v1/models`; clients send it as `model` |
| `CONTEXT` | `16384` | `--context-length`: image tokens + prompt + `max_tokens` must fit ([Request format](#request-format)) |
| `MEMFRAC` | `0.6` | `--mem-fraction-static` |
| `MAXREQ` | *(unset)* | `--max-running-requests`. Unset, the patch uses 48 in self-spec mode; causal mode keeps SGLang's default |
| `GRAPHBS` | *(SGLang default)* | `--cuda-graph-max-bs`; lower it if CUDA-graph capture overflows the FlashInfer workspace at a large `MAXREQ` in self-spec mode |
| `DRAFT` | `33` | self-spec draft window K = `bd_size` + 1 (`--speculative-num-draft-tokens`) |
| `FAST` | `1` | the runtime optimizations in the patch. Both modes: GLM ViT as CUDA graphs ([below](#vit-cuda-graphs)), bf16 pixel transport, single-threaded CPU ops in the server. Self-spec: mask-free attention plans, M-RoPE cache fix, GPU-side commit, host-side lengths, device-side positions, next draft launched ahead. `0` serves the unoptimized runtime; other values exit 2. Both give the same outputs |
| `TOKENIZER_WORKERS` | `1` | `--tokenizer-worker-num`, with `FAST=1` too. `2` or more is an opt-in, for a host whose users you trust; `serve.sh` then prints a note at startup. Every worker enforces `API_KEY` and the admin key ([Security](#security)) |
| `OMP_NUM_THREADS_SERVER` | `1` | with `FAST=1`: OpenMP / MKL / torch threads in the server |
| `CUDA_HOME` | `/usr/local/cuda-12.8` | CUDA toolkit used for kernel compilation |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache |
| `EXTRA_ARGS` | *(empty)* | extra `sglang.launch_server` options from a fixed list of logging and capacity options, split on whitespace (no shell quoting), appended last; anything else exits 2 ([Security](#security)) |
| `SKIP_PREFLIGHT` | *(unset)* | `1` skips the startup checks: `$CUDA_HOME/bin/nvcc` reports release 12.x, `ninja` is in `VBIN`, `nvidia-smi` is on `PATH`, and every selected GPU has compute capability 9.0 (a failed check exits 1) |

`PORT`, `CONTEXT`, `DRAFT`, `MAXREQ`, `GRAPHBS`, `TOKENIZER_WORKERS` and `OMP_NUM_THREADS_SERVER` must be positive integers and `MEMFRAC` a fraction such as `0.6`; `SERVED_NAME` and `SERVE_HOST` must not start with `-` or contain spaces. Any other value exits 2.

Self-spec mode also exports these, each overridable from the environment:

- `SELFSPEC_FULLDRAFT=1`: all mask logits are draft candidates and a fully accepted draft commits the bonus token. `0` is the older round of at most K tokens.
- `SELFSPEC_POS_OFFSET=1`: every generated token, in the draft and verify windows and in the KV cache, sits one RoPE position later than causal decoding, and the model's training, put it. GravityOCR's published self-spec results were produced with this layout, so keep it to reproduce them. `0` uses the causal layout, under which verification computes what causal decoding computes, up to floating point; it is not yet validated for GravityOCR on this build's optimized path beyond the correctness tests described in [What self-spec reproduces](#what-self-spec-reproduces).
- `SELFSPEC_MASK_ID=59282`, only when `CKPT` has no `block_diffusion.json`.

`FAST=1` sets `SS_GLM_VIT_GRAPH_MAXM=2048` and `SS_GLM_PV_BF16=1`, and in self-spec mode `SELFSPEC_FASTATTN`, `SELFSPEC_MROPE_FIX`, `SELFSPEC_FAST_COMMIT`, `SELFSPEC_HOSTLEN`, `SELFSPEC_DEV_POS` and `SELFSPEC_DRAFT_AHEAD` to `1`. Request sampling settings, penalties and grammars need no switch.

`serve.sh` also handles four setup problems: it puts `VBIN` and `$CUDA_HOME/bin` on `PATH` (CUDA-graph capture shells out to `ninja`), runs `$VBIN/python` directly rather than through `uv run`, sets `VLLM_USE_FLASHINFER_SAMPLER=0` (that sampler's JIT path crashes without its own build setup), and serves with `HF_HUB_OFFLINE=1` after the one download. It also clears `PYTHONOPTIMIZE`, which would strip SGLang's `assert` checks, and sets `PYTHONSAFEPATH=1`, so Python never imports from the current directory: run from `inference/gravityocr/`, the recipe's own `sglang/` folder would otherwise shadow the installed SGLang (`setup.sh` sets it too). It does not pass `--trust-remote-code`, and `EXTRA_ARGS` cannot add it; GLM-OCR is implemented natively in transformers 5.x and SGLang.

### ViT CUDA graphs

With `FAST=1` the GLM-OCR vision encoder runs as one CUDA graph per patch count, captured when the model loads. Three settings control it, in both modes:

| env | `FAST=1` | meaning |
|---|---|---|
| `SS_GLM_VIT_GRAPH_MAXM` | `2048` | largest patch count per image that gets a graph; one graph per multiple of 4 up to it. Decimal digits only, rounded down to a multiple of 4, at most 16384; unset, empty or `0` turns the graphs off (the `FAST=0` setting) |
| `SS_GLM_VIT_EAGER_LIST` | `1` | how calls without a graph run (a crop above `MAXM`, several images): `1` runs the eager encoder with its sequence offsets passed once as a list, instead of a host copy and a sync per block; `0` runs the original eager forward. Only `0` or `1` |
| `SS_GLM_VIT_GRAPH_CHECK` | `1` | `1` compares every captured graph, and a few shapes of the list path, bitwise with the original eager forward at startup and drops any that differ; `0` skips that check. Only `0` or `1` |

A malformed value (`2k`, `-4`, `true`, ...) stops the server at startup with an error that names the variable. The server log shows what was parsed, `[ss-vit-graph] settings: ...`, then either `[ss-vit-graph] ON: ...` (graph count, capture time, memory, sizes dropped by the check) or `[ss-vit-graph] OFF: ...`. If the capture fails, for example because the GPU runs out of memory, the server logs `[ss-vit-graph] OFF: capture failed (...)`, releases what it had captured, and continues with the original eager forward, which gives the same outputs. The graphs change speed only; `FAST=0` and `FAST=1` give the same outputs. Their memory cost is under [Known limits](#known-limits).

## Verify

```bash
curl -s http://127.0.0.1:30000/v1/models                                     # lists "glm-ocr" once ready
"$VBIN/python" inference/gravityocr/ocr_request.py --synthetic 'Hello GravityOCR'
"$VBIN/python" inference/gravityocr/ocr_request.py table.png --task table
```

`ocr_request.py` takes either an image file or `--synthetic TEXT`, which renders `TEXT` into a small PNG with Pillow, so the first check needs no sample image; the reply should read back the same text. It prints the recognized text on stdout and the token usage on stderr. It exits 1, with a one-line message, on an HTTP error, a refused or dropped connection, a reply slower than `--timeout`, or a reply without `choices[0].message.content`. Apart from `--synthetic` it uses only the standard library and runs on any Python 3.10+. Its flags: `--task text|table|formula` (default `text`; the prompts are in [Request format](#request-format)) or `--prompt` with a prompt of your own, `--url` (default `http://127.0.0.1:30000`), `--model` (default `glm-ocr`), `--max-tokens` (default 4096) and `--timeout` (default 300 s). It reads the key from `$API_KEY` only, never from its command line. It sends the request to `--url` and nowhere else: it refuses redirects (urllib would resend the `Authorization` header to the host a redirect names) and ignores proxy variables such as `https_proxy` (the proxy would receive the key).

With a key, give `curl` the header on standard input, so the key stays off its command line:

```bash
printf 'Authorization: Bearer %s\n' "$API_KEY" | curl -s -H @- http://127.0.0.1:30000/v1/models
```

## Request format

A standard OpenAI chat completion, `POST /v1/chat/completions`, with the image first and the task prompt second:

```json
{
  "model": "glm-ocr",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo..."}},
    {"type": "text", "text": "Text Recognition:"}
  ]}],
  "max_tokens": 4096,
  "temperature": 0
}
```

| prompt | `--task` | region | output |
|---|---|---|---|
| `Text Recognition:` | `text` | a text block (paragraph, title, caption, list) | plain text in reading order |
| `Table Recognition:` | `table` | one table | HTML table |
| `Formula Recognition:` | `formula` | one display formula | LaTeX |

- The reply text is `choices[0].message.content`; `usage` has the token counts.
- Send images as `data:` URLs (see [Security](#security)). Greedy decoding (`temperature: 0`) is the setting GravityOCR's results use and the one OCR wants; sampling settings, penalties, `logit_bias`, `min_tokens` and structured output (`response_format`) are served too, and so are log probabilities (`logprobs`, `top_logprobs`), as [What self-spec reproduces](#what-self-spec-reproduces) describes. Keep `max_tokens` at 4096 or below.
- `CONTEXT` (16384) must hold the image tokens, the prompt and `max_tokens`. The image processor downscales every crop to at most 6,144 image tokens (its `longest_edge` of 9,633,792 pixels counts two temporal frames, so 4,816,896 pixels at 28 × 28 pixels per token; a 4000 × 4000 crop becomes about 6,100 prompt tokens), so any crop fits with `max_tokens` 4096. If you lower `CONTEXT` or raise `max_tokens`, SGLang checks the total when a request arrives and rejects one that cannot fit with HTTP 400 ("Requested token count exceeds the model's maximum context length"). Self-spec mode also reserves K + 1 KV slots per round, and a request within K + 1 tokens of `CONTEXT` has not been exercised on a GPU, so keep that margin. A higher `CONTEXT` costs KV-cache memory per request.

## What self-spec reproduces

Self-spec commits a token only where the AR head, with the same request settings and given the tokens committed before it, would have produced it, at the position layout described under 1. below. The draft decides how many tokens a round commits, not which ones. For every request parameter SGLang's causal decoding applies, self-spec applies the same processing at every position it verifies:

- **Greedy decoding** (`temperature: 0`): every committed token is the AR head's highest-scoring token given the tokens before it.
- **Sampling** (`temperature`, `top_k`, `top_p`, `min_p`): the first token of each round is drawn by SGLang's own sampler from the request's settings. Each drafted token is then accepted with the probability that AR sampling gives it, and at the first rejection the token is drawn from the AR distribution with the rejected token removed (speculative sampling against a deterministic draft). Each committed token therefore follows the AR sampling distribution for its prefix, with the filters SGLang's sampler applies, `min_p` and the joint top-k / top-p rule included. This holds with SGLang's speculative acceptance thresholds at their default of 1.0; the server refuses to start with any other value (`EXTRA_ARGS` cannot set them anyway).
- **Penalties and token constraints** (`repetition_penalty`, `frequency_penalty`, `presence_penalty`, `min_tokens`, `logit_bias`, stop tokens and stop strings): applied at every position with that position's own prefix, tokens drafted earlier in the same round included, and every committed token is counted, as causal decoding counts it.
- **Structured output** (`response_format` with a JSON schema, `regex`, `ebnf`): every position is masked by the grammar state after its prefix. The default grammar backend, xgrammar, and llguidance are supported; a backend without rollback is refused at startup. A token the grammar still rejects ends that request with an abort, as in causal decoding, and the server keeps running.
- **Custom logit processors** are applied per position with that position's prefix as well, but they need `--enable-custom-logit-processor`, which `serve.sh` refuses ([Security](#security)).
- **Log probabilities** (`logprobs` and `top_logprobs` in the chat API, `return_logprob` in `/generate`, streamed or not): one entry per generated token, as causal decoding records them, with the request's temperature. The first token of each round gets the value SGLang's sampler computes for it; the other committed tokens get theirs from the verify pass, after the processing above. Prompt log probabilities come from the prefill, which is the same in both modes.

A batch that is greedy and uses none of these options skips all of this: it runs the plain greedy round, with no extra kernels, synchronizations or host round trips.

What still separates the two modes:

1. **The position layout.** At the default `SELFSPEC_POS_OFFSET=1`, every generated token sits one RoPE position later than causal decoding puts it ([Serve](#serve)). Self-spec then commits the AR head's choices for that shifted layout, which is a slightly different computation from `MODE=causal`. `SELFSPEC_POS_OFFSET=0` removes this difference.
2. **Floating point.** The verify pass scores K positions in one forward, with other kernels and batch shapes than causal decoding's one-token steps. Under bf16, a token whose two best candidates are within a rounding step of each other can come out differently, and the outputs diverge from that token on.
3. **SGLang's own run-to-run behaviour.** SGLang runs without deterministic inference by default, so the same mode can resolve such a near-tie differently from one server run to the next: in two `DP=2` causal runs over the same 100 crops, one crop came out differently in one run (it stopped before a trailing `\tag`). With `repetition_penalty` together with `frequency_penalty` or `presence_penalty`, SGLang applies the penalizers in an order that differs between server processes, so two causal servers can differ as well; self-spec follows the order of its own process. And SGLang's overlap scheduler, which `MODE=causal` uses by default, counts each generated token for the penalties one step late; self-spec counts it at once, as causal decoding with `EXTRA_ARGS=--disable-overlap-schedule` does. Compare penalties against that.

Checks on H100 with GravityOCR, greedy unless stated. At the default layout, `FAST=1` and `FAST=0` give identical outputs on 100 OCR crops, and self-spec and causal decoding agree on 92 of them; GravityOCR's own record at this layout is 96.6% of 8,922 crops. At `SELFSPEC_POS_OFFSET=0`, against causal decoding without the overlap scheduler, on 40 crops: outputs are identical on all 40, with `FAST=1` and with `FAST=0`. With options, most stay identical (39 or 40 of 40 with each of `repetition_penalty` 1.3, `frequency_penalty`, `presence_penalty`, `logit_bias` and `min_tokens`; 10 or 12 of 12 for each grammar type), no output breaks its constraint, and every remaining difference, with the penalizers in the same order, is a near-tie of at most 0.25 nats in which the self-spec token was one of causal decoding's two best candidates. With sampling, also at offset 0 (temperature 1.0, alone and with top-k, top-p and min-p), chi-square tests of the token counts at the first positions against the causal server's probabilities pass, as they do for the causal server's own samples. Log probabilities, also at offset 0, match the causal server's on the tokens both modes produce up to bf16 rounding, and in greedy decoding asking for them changes no output token. Compare the modes on token ids over many crops, not on one.

## Security

- **Loopback by default.** The server binds `127.0.0.1`. Binding any other address with `SERVE_HOST` requires `API_KEY`; without it `serve.sh` exits with status 2. Clients then send `Authorization: Bearer <key>` ([Verify](#verify) shows how to keep it off `curl`'s command line).
- **Every tokenizer worker checks the keys.** Upstream SGLang v0.5.10 checks keys only when it runs a single tokenizer worker: with more, its HTTP workers accept every request, and they refuse to start if a key is set. The patch installs the key check in every worker. Each worker receives the keys through the pipe it is started with, not through shared memory or the environment, and a worker without them answers 503. So `API_KEY` also holds with `TOKENIZER_WORKERS=2`, an opt-in; the default is one worker. Each worker logs whether it holds the keys (`api_key set`, `admin_api_key set`), never the keys.
- **Management endpoints.** SGLang serves management endpoints next to the OpenAI API: `/update_weights_from_disk` (which loads weights from a path the caller chooses), the other weight-update routes, `/flush_cache`, `/start_profile`, `/load_lora_adapter`, `/set_internal_state`, `/set_trace_level` and more (upstream v0.5.10 leaves `/set_trace_level` at the normal level; the patch makes it a management endpoint). On every start `serve.sh` gives SGLang a fresh random admin key that it never prints, so these endpoints refuse every caller, holders of `API_KEY` included, whatever the number of tokenizer workers. `/server_info` and `/get_server_info` stay at SGLang's normal level (open without `API_KEY`, else with it) and return the server's configuration; the patch blanks `api_key`, `admin_api_key` and `ssl_keyfile_password` there (upstream v0.5.10 returns them), so `API_KEY` does not reveal the admin key. `/load_lora_adapter_from_tensors` is also at the normal level: it refuses because LoRA is off (`serve.sh` never turns it on, and `EXTRA_ARGS` refuses the `--lora*` options), not because of the admin key.
- **Keys stay off command lines and off the disk.** `serve.sh` takes `API_KEY` only from the environment. It hands both keys to SGLang through `--config`, which names a FIFO (mode 0600) in a private temporary directory: SGLang reads it once, at startup, so the keys pass through a kernel pipe, are written to no file, and appear in no command line and not in the server's environment. Until SGLang has read the FIFO, the short-lived writer and its guard, which are forks of `serve.sh`, still show an `API_KEY` exported by the caller in their `/proc/<pid>/environ`; only the same user can read that, and that user can open the FIFO anyway. A writer outside the server's process tree removes the directory as soon as SGLang has read the FIFO, or within about a second after a server exits without reading it, also when the server kills its own process tree. If that writer is killed too (for example by a `SIGKILL` to the whole process group), a `gravityocr-serve.*` directory holding only the empty FIFO can remain under `$TMPDIR` (default `/tmp`); it contains no key.
- **No keys in logs or shared memory.** The patch hides the three secret fields from the server-arguments line SGLang logs at startup and from its request and crash dumps. With two or more tokenizer workers SGLang passes its arguments to them in `/dev/shm/multi_tokenizer_args_<pid>`; the patch leaves the keys out of it, creates it with mode 0600 and refuses one that another user created first (the server then does not start), and never unpickles a segment that is not its own. Upstream v0.5.10 opened such a segment if it already existed, which let another local user read it and replace its contents with data the workers then load.
- **Do not expose the port directly, even with a key.** SGLang loads `image_url` values on the server: an `http(s)://` URL makes the server fetch it, from wherever the host can reach, and a `file://` URL or a bare path makes it read a local image file. `/health*` and `/metrics` answer without a key. Put the server behind a reverse proxy that authenticates callers, forwards only `/v1/chat/completions` and `/v1/models` (so not `/server_info`, `/get_server_info`, `/health*`, `/metrics` or the management endpoints), and only requests whose image URLs are `data:` URLs. `ocr_request.py` sends `data:` URLs only.
- **Other listening sockets.** `SERVE_HOST` and `PORT` are the HTTP listener. Each replica also opens torch.distributed's TCP store on a random free port. The patch rendezvouses a single-node server on `127.0.0.1` and binds that store to `127.0.0.1` only, whatever `SERVE_HOST` is (upstream v0.5.10 connects the rendezvous to `--host`, and torch binds the store on every interface). torch.distributed's other sockets are not affected: gloo listens on the address the host name resolves to, and NCCL opens sockets while it starts. `--dist-init-addr`, refused in `EXTRA_ARGS`, would move the rendezvous off loopback, switch image tensors to the cross-node CPU transport and make data-parallel replicas collide on one port; `SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE`, if set in the environment, also replaces the loopback rendezvous. On a host reachable from untrusted networks, set `SERVE_HOST` to one interface's address rather than the wildcard address, and firewall every inbound port except `PORT`.
- **`EXTRA_ARGS` takes a fixed list.** Every `EXTRA_ARGS` token that starts with `-` must be one of these exact option names, bare or as `--name=value`; abbreviations and anything else exit 2:

  `--log-level` `--log-level-http` `--log-requests` `--log-requests-level` `--enable-metrics` `--enable-cache-report` `--decode-log-interval` `--show-time-cost` `--enable-request-time-stats-logging` `--max-total-tokens` `--max-prefill-tokens` `--chunked-prefill-size` `--max-queued-requests` `--schedule-policy` `--schedule-conservativeness` `--num-continuous-decode-steps` `--stream-interval` `--cuda-graph-bs` `--disable-cuda-graph` `--disable-cuda-graph-padding` `--disable-radix-cache` `--disable-overlap-schedule` `--skip-server-warmup` `--watchdog-timeout` `--soft-watchdog-timeout` `--random-seed`

  Options that `serve.sh` sets from its own variables (`--host`, `--port`, `--context-length`, `--tokenizer-worker-num`, ...) are refused with the variable to use instead. Among the rest, these matter for security:

  | refused | why |
  |---|---|
  | `--config` | merges arbitrary options from a YAML file. `serve.sh` adds its own `--config` for the keys, after every check |
  | `--host`, `--api-key`, `--admin-api-key` | the bind address and keys are set only through `SERVE_HOST`, `API_KEY` and the launcher's own admin key |
  | `--enable-custom-logit-processor` | makes the server deserialize client-supplied logit processors with `dill`: remote code execution, with no upstream fix |
  | `--enable-lora`, `--lora-paths` and the other `--lora*` options | LoRA is not part of this recipe. `--lora-paths` alone turns it on, and the LoRA manager has a known request-triggered crash |
  | `--grpc-mode`, `--encoder-only`, `--use-ray` | other front-ends that serve on `SERVE_HOST:PORT` without SGLang's key check |
  | `--enable-dp-attention` | moves SGLang's internal pickle-based IPC from unix sockets to TCP |
  | `--dist-init-addr`, `--nnodes`, `--node-rank` | multi-node serving. `--dist-init-addr` also moves the torch.distributed rendezvous off loopback and switches image tensors to the cross-node CPU transport (see Other listening sockets) |
  | `--trust-remote-code`, `--load-format` | would run code from the checkpoint, or load its weights with a pickle-based loader |

  For anything beyond the list, run `sglang.launch_server` yourself.
- **Known advisories.** The pinned server environment carries advisories that have no fix on this base (in SGLang v0.5.10, torch 2.9.1, transformers 5.3.0 and diskcache). Each one is recorded in [`docs/supply-chain/exceptions.md`](../../docs/supply-chain/exceptions.md), with whether it is reachable here and the mitigation this launcher relies on. Report vulnerabilities through [`SECURITY.md`](../../SECURITY.md), not a public issue.

## Known limits

- **Hardware.** Validated only on H100 (compute capability 9.0) with driver 570 and a CUDA 12.8 toolkit.
- **Pinned base.** The patch is a diff against SGLang v0.5.10 (`1519acf`) and does not apply to later releases without porting. FlashInfer must be exactly 0.6.7.post2: the fast attention path calls its planning internals and checks the version.
- **Self-spec and causal can differ.** By the position layout at the default `SELFSPEC_POS_OFFSET=1` and by bf16 near-ties; [What self-spec reproduces](#what-self-spec-reproduces) has the details and the checks.
- **Crops only.** No page layout, no PDF input, no page assembly.
- **Several tokenizer workers on a busy host.** With `TOKENIZER_WORKERS=2` or more, uvicorn restarts any worker that does not answer its 5-second health check, and a new worker can take longer than that to import SGLang while the host's CPUs are busy. Start-up can then take minutes. The default, one worker, has no such step. This is upstream SGLang's multi-worker start-up, unchanged by the patch.
- **Batch caps.** Self-spec reserves K + 1 KV slots per running request each round. At a large `MAXREQ`, CUDA-graph capture can overflow the FlashInfer workspace; lower `MAXREQ` or `GRAPHBS`.
- **ViT graph memory.** The ViT CUDA graphs are captured while the model loads, before SGLang sizes the KV cache, and take about 3.3 GB of GPU memory from it. On H100 80 GB the KV pool holds about 7.7% fewer tokens at the default `MEMFRAC=0.6` (646,525 instead of 700,253) and 17% fewer at `MEMFRAC=0.3`. The cost does not shrink with the card, so on a GPU with less memory it is a larger share of the KV cache (every GPU this recipe is validated on has 80 GB, and `serve.sh` refuses non-Hopper GPUs). `SS_GLM_VIT_GRAPH_MAXM=0`, or `FAST=0`, gives it back. The graphs also lengthen startup. After a failed capture the server continues without them; the CUDA driver keeps a few MB per graph it had already built and reuses that memory for the model's own CUDA graphs.
- **Sampling memory.** A batch that samples (`temperature` > 0 with `top_k` or `top_p`) or uses a penalty builds vocabulary-sized float tensors for every verify position, K = 33 rows per request, and the top-k / top-p filter sorts them. They are allocated after SGLang has sized the KV cache, so a large sampled batch at a high `MEMFRAC` can run the scheduler out of GPU memory. The default `MEMFRAC=0.6` leaves room for the default 48 running requests; above it, lower `MAXREQ` for sampled traffic. Greedy batches without options allocate none of this.
- **No GPU coverage in CI.** The tests below check files, launcher arguments, request payloads and the patch's host-side logic; the weekly fetch-recipe job applies the patch at its pinned commit and compiles the result. Nothing in CI builds the server environment or starts a server, so a change to the patch or the lock needs a manual GPU check. The patch and launcher were validated on H100 in the environment frozen in the GravityOCR repository. `sglang/requirements-lock.txt` raises 11 pins above that environment for security fixes and installs torchvision, torchaudio, torchao and torchcodec as `+cu128` builds of the same versions; a fresh `setup.sh` run with it on H100 (2026-11-12) gave the same greedy outputs as that environment on 100 OCR crops, in self-spec and causal mode.

## Tests

```bash
pytest -q inference/gravityocr/      # CPU only, no downloads; also runs under `pytest -q` from the repo root
```

| file | checks |
|---|---|
| `test_gravityocr_patch.py` | `sglang/BASE_COMMIT.txt` is a full commit SHA, and the pins quoted in this README, `sglang/setup.sh` and `NOTICE` match it; the patch covers exactly 23 files (22 upstream sources plus `python/pyproject.toml`); the seven new files parse as Python 3.10 and carry the licence header, with SGLang's copyright notice exactly in the ones that adapt SGLang code; `sglang/setup.sh` is valid bash and installs with `--no-deps --no-config` from explicit indexes, and builds SGLang offline without build isolation; `/server_info` blanks every secret and a failed ViT graph capture falls back to the eager encoder; the lock names the base tag, holds exactly the four `+cu128` builds, pins the build backend and the versions `setup.sh` checks, and agrees with `validated-freeze.txt`; `setup.sh` sets `PYTHONSAFEPATH`, and its fetch, patch and verify steps run against a small synthetic repository, accept a re-run and refuse every kind of local change |
| `test_gravityocr_launcher.py` | runs `serve.sh` against a stub `VBIN` and a local checkpoint directory: mode flags, the `SELFSPEC_*` environment, `DP`, value checks, the `SERVE_HOST` / `API_KEY` rule, one tokenizer worker by default and the note for more, with and without a key, an admin key on every start, `PYTHONSAFEPATH`, no `--grammar-backend` override, the key FIFO (mode 0600, delivered once, nothing left behind on eight exit paths, including a server that kills its own process tree), the `--config` guard, the `EXTRA_ARGS` allowlist, no rendezvous options or override, the named mask-id fallback, and the preflight checks with stub `nvcc` / `nvidia-smi` |
| `test_gravityocr_request.py` | `ocr_request.py` with the network call replaced: the payload (`[image_url, text]` order, temperature 0, a `data:` URL that decodes back to the input bytes), `--task` and `--prompt`, CLI defaults and argument checks, the key taken from `$API_KEY` only, HTTP errors, refused and dropped connections, timeouts and malformed replies (exit 1), and `--synthetic` rendering a PNG; on real local sockets: a request succeeds, every redirect is refused without the key reaching the redirect target, and proxy variables are ignored |
| `test_gravityocr_selfspec_cpu.py` | the self-spec commit rules against SGLang's per-token `check_finished` loop on thousands of random cases; the in-block penalty rows and penalty counting against causal decoding's step-by-step result, bitwise, in every penalizer order; the sampling target distributions (joint top-k / top-p, `min_p`); custom-processor rows; the grammar mask rows against a matcher fed token by token; a grammar rejection aborting the request; a greedy batch running only the greedy kernel; with the patched SGLang, the full-draft round's log probabilities (the first token's entry, then the verified tokens', also when every request ends at its first token) |
| `test_gravityocr_auth.py` | needs the patched SGLang: `/server_info`, `/get_server_info`, the scheduler state, the logged arguments and the dumps carry no key; every auth level with every key; real uvicorn worker processes all enforcing the keys; the private shared-memory segment |
| `test_gravityocr_gating.py` | the patch leaves SGLang's diffusion-LLM and EAGLE draft-runner modules untouched and adds no code that depends on diffusion-LLM state; with the patched SGLang: the shared functions it changes behave exactly as v0.5.10 for every server that is not self-spec (differential tests against verbatim copies), EAGLE-family CUDA-graph capture, `--dllm-algorithm` refused together with self-spec, and nothing left of the removed `SELFSPEC_NOMASK` path |
| `test_gravityocr_dist_init.py` | the single-node torch.distributed rendezvous: the patch's store function, run on CPU torch, listens on loopback only (both TCPStore backends, IPv4 and IPv6), serves two ranks, closes its port with the store and leaves every other init method to torch; with the patched SGLang: a single-node server rendezvouses on `127.0.0.1` whatever `--host` is, the override and `--dist-init-addr` keep their precedence, and `init_distributed_environment` with gloo (one and two ranks) runs its collectives over a loopback-only store |
| `test_gravityocr_vit_graph.py` | the ViT graph settings parser, the replay / eager dispatch, the single caller thread, and the fallback after a failed capture (memory released after the exception is handled, eager forward kept); with the patched SGLang: the runner's copy of the encoder forward in sync with the model's, and the model keeping its eager forward after a failed capture |

`test_gravityocr_auth.py`, and the parts of `test_gravityocr_selfspec_cpu.py`, `test_gravityocr_gating.py`, `test_gravityocr_dist_init.py` and `test_gravityocr_vit_graph.py` that need the patched SGLang, skip without it, as in CI. To run them, use the server's Python with the checkout first on the path (otherwise `inference/gravityocr/sglang/` shadows the editable install) and pytest from a separate directory, so the server venv stays as locked:

```bash
PYT=$(mktemp -d) && uv pip install -q --python "$VBIN/python" --target "$PYT" --no-deps pytest pluggy iniconfig
CUDA_VISIBLE_DEVICES= PYTHONPATH="$SGLANG_SRC/python:$PYT" "$VBIN/python" -m pytest -q -p no:cacheprovider \
  inference/gravityocr/test_gravityocr_{selfspec_cpu,auth,gating,dist_init,vit_graph}.py
```

On a GPU, serve both modes and send each the same crops. For an exact comparison, start self-spec with `SELFSPEC_POS_OFFSET=0` and causal with `EXTRA_ARGS=--disable-overlap-schedule`:

```bash
bash inference/gravityocr/serve.sh &                               # self-spec on GPU 0, port 30000
MODE=causal GPU=1 PORT=30001 bash inference/gravityocr/serve.sh &   # causal on GPU 1, port 30001
"$VBIN/python" inference/gravityocr/ocr_request.py crop.png --url http://127.0.0.1:30000
"$VBIN/python" inference/gravityocr/ocr_request.py crop.png --url http://127.0.0.1:30001
```

## Files

```
inference/gravityocr/
  README.md, MODEL_CONTRACT.md, NOTICE
  serve.sh                   launcher: MODE=self-spec | causal, env-driven
  ocr_request.py             one-crop OpenAI-compatible client (stdlib; Pillow for --synthetic)
  test_gravityocr_*.py       CPU tests
  sglang/
    BASE_COMMIT.txt            upstream sgl-project/sglang pin (tag v0.5.10)
    gravityocr_selfspec.patch  diff against that commit
    python/sglang/srt/...      the seven new files the patch needs
    setup.sh                   fetch the pin -> patch -> verify -> venv from the lock; prints VBIN
    requirements.in, overrides.txt, validated-freeze.txt, requirements-lock.txt   the server environment
```

## Provenance and license

- **Code.** Apache-2.0, like the rest of this repo ([`LICENSE`](../../LICENSE)). It comes from [`trillion-labs/GravityOCR`](https://github.com/trillion-labs/GravityOCR), which Trillion Labs also publishes. There, the SGLang patch and the seven new SGLang files are likewise Apache-2.0, and the launcher and client are MIT; Trillion Labs contributes them here under Apache-2.0.
- **Upstream SGLang.** [`sgl-project/sglang`](https://github.com/sgl-project/sglang), Apache-2.0, at `1519acf37c23f2189adb93f57ca9cd2db1bebf18` (tag v0.5.10). It is not vendored: `setup.sh` fetches it. This folder carries only the patch, which modifies 22 upstream source files and `python/pyproject.toml`, and the seven new files. Five of the new files adapt SGLang code and carry SGLang's copyright notice; [`NOTICE`](NOTICE) lists them with their sources.
- **Weights.** [`trillionlabs/GravityOCR`](https://huggingface.co/trillionlabs/GravityOCR) is MIT-licensed and fine-tuned from [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) (MIT). Neither is redistributed here; `serve.sh` downloads the checkpoint from the Hub.
- **Design notes.** Comments in the new SGLang files mention "natcache", `src/fast_sampling.py` and `docs/SGLANG_SELFSPEC_IMPL.md`. These are the in-process reference decoder and the implementation notes in the [GravityOCR repository](https://github.com/trillion-labs/GravityOCR), which is the place to read about the design. Some of those comments also say that a path matches causal decoding; they record checks made while the patch was developed, on test sets and at a given position layout, not a guarantee for every input. [What self-spec reproduces](#what-self-spec-reproduces) states what holds. The files keep those comments so that they stay identical to GravityOCR's, and the header of `selfspec_diffusion_worker.py` points here.

Attribution for every component is in [`NOTICE`](NOTICE); see also the repo's [`NOTICE`](../../NOTICE) and [`COMPLIANCE.md`](../../COMPLIANCE.md). What a checkpoint needs in order to be servable is in [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md).
