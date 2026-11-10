<div align="center">

# GravityOCR nano-inference

**Crop-level document OCR with [`trillionlabs/GravityOCR`](https://huggingface.co/trillionlabs/GravityOCR), served with self-speculative decoding on upstream SGLang.**
One layout region per request, behind an OpenAI-compatible API: a launcher, a one-crop client and a pinned SGLang patch.

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](../../LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Backend](https://img.shields.io/badge/backend-SGLang%20v0.5.10-8A2BE2.svg)](#install)
[![GPU](https://img.shields.io/badge/GPU-H100-76B900.svg)](#requirements)

[Requirements](#requirements) · [Install](#install) · [Serve](#serve) · [Verify](#verify) · [Requests](#request-format) · [Security](#security) · [Known limits](#known-limits) · [Tests](#tests)

</div>

---

GravityOCR is [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) (a CogViT vision encoder and a 0.5B GLM text decoder) fine-tuned as a block-diffusion model with a single output head that decodes both ways: with bidirectional attention over a block of 32 masked tokens it fills them in parallel, and with causal attention it predicts the next token, a skill an AR auxiliary loss keeps. The architecture is unchanged, so the checkpoint loads as a stock `GlmOcrForConditionalGeneration`. The weights were trained outside this repo; this folder only serves them.

Self-speculative decoding uses both modes of that one model. Each round runs two forward passes. The **draft** pass reads the boundary token followed by 32 masks, bidirectional within that window, and proposes the next block. The **verify** pass runs the causal path over the draft, accepts the longest prefix that matches the model's greedy AR choices, and adds its own next AR token. A round commits up to K + 1 = 34 tokens (K = `DRAFT` = 33).

| `MODE` | what it does |
|---|---|
| **`self-spec`** (default) | block-diffusion draft, AR verify, inside SGLang's speculative-decoding loop (`--speculative-algorithm SELFSPEC_DIFFUSION --speculative-num-draft-tokens 33 --page-size 1`) |
| `causal` | the same weights decoded autoregressively, one token per forward pass: the reference |

The model reads **one layout region per request**: a text block, a table or a formula crop. Page layout detection and page assembly are out of scope for this folder; a full page needs a layout detector in front of the server. Section 5 of the [tutorial notebook](#tutorial-notebook) shows one: [PP-DocLayoutV3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) (Apache-2.0, the detector GLM-OCR and the Gravity app use) finds typed regions in reading order, and each crop goes to the server with its task prompt.

This is the second nano-inference next to [`../zoro-1/`](../zoro-1/). The two do not share a backend. This one is upstream SGLang v0.5.10 plus an Apache-2.0 patch; zoro-1's SGLang backend is a different, noncommercial-licensed fork. Build each in its own environment.

## Requirements

- **Linux x86_64**, `git` and [`uv`](https://docs.astral.sh/uv/). Install and first start need network access to GitHub, PyPI, the PyTorch package index and the Hugging Face Hub. The model is public; no token is needed.
- **An NVIDIA Hopper GPU (compute capability 9.0).** Validated on H100 80 GB with driver 570 (CUDA 12.8). The FlashInfer and FA3 attention paths the patch uses were validated only there, and `serve.sh` refuses other GPUs unless `SKIP_PREFLIGHT=1`.
- **A CUDA 12.x toolkit at `CUDA_HOME`** (default `/usr/local/cuda-12.8`). FlashInfer and Triton compile kernels when the server starts, so `nvcc` must be present. Use a toolkit no newer than your driver supports.
- **Python 3.12** (uv fetches it if needed).
- **A separate environment.** The server venv lives inside your SGLang checkout and is used only by `serve.sh`. Do not install it into the repo's root environment or into zoro-1's backend environment; the pins clash. `sglang/requirements-lock.txt` pins, among others: torch 2.9.1 (the PyPI build, CUDA 12.8 runtime) with torchvision, torchaudio, torchao and torchcodec as `+cu129` builds from the PyTorch cu129 index, transformers 5.3.0, flashinfer-python 0.6.7.post2 and sglang-kernel 0.4.1.

## Install

```bash
SGLANG_SRC=$HOME/src/sglang-gravityocr bash inference/gravityocr/sglang/setup.sh
```

`SGLANG_SRC` must be outside this repository and be missing, empty, or an SGLang checkout. `setup.sh` is safe to re-run; a finished build is only re-verified. It:

1. checks for `git`, `uv` (the lock was generated with 0.11.27) and a CUDA 12.x `nvcc` at `CUDA_HOME`. `SKIP_PREFLIGHT=1` skips the `nvcc` check, for example to build on a machine without the toolkit;
2. fetches exactly the pinned base commit of [`sgl-project/sglang`](https://github.com/sgl-project/sglang) from `sglang/BASE_COMMIT.txt` (`1519acf37c23f2189adb93f57ca9cd2db1bebf18`, tag v0.5.10) into an empty `SGLANG_SRC` with a shallow fetch, and verifies that `HEAD` is that commit. In a clean SGLang checkout at another commit it fetches the base commit without making the clone shallow and checks it out as a detached `HEAD`, leaving the branches alone. A checkout with local changes, or a git repository that is not SGLang, is refused;
3. applies `sglang/gravityocr_selfspec.patch` (skipped when already applied), checks that the changed files are exactly the patch's, and copies the four new files from `sglang/python/` into the checkout;
4. creates `$SGLANG_SRC/.venv-sglang` (Python 3.12) and installs `sglang/requirements-lock.txt` as is, with `--no-deps --no-build` and nothing re-resolved: the PyPI pins first, from `https://pypi.org/simple` only, then the four `+cu129` builds from the PyTorch cu129 index only. uv config files and uv's index variables (`UV_INDEX`, `UV_EXTRA_INDEX_URL`, `UV_FIND_LINKS`, ...) are ignored. The pins carry no hashes;
5. installs the patched SGLang itself, editable, as version 0.5.10;
6. checks the installed packages against the lock and prints the `VBIN=...` line that `serve.sh` needs.

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
| `API_KEY` | *(unset)* | bearer token the server then requires on every endpoint except SGLang's `/health*` and `/metrics`. SGLang v0.5.10 checks keys only with one tokenizer worker, so a key means `--tokenizer-worker-num 1`, also with `FAST=1`; `TOKENIZER_WORKERS` above 1 together with a key exits 2. Printable ASCII without spaces, not starting with `-`. Set it in the environment; as a `KEY=VALUE` argument it is refused (exit 2). It reaches SGLang in a mode-0600 YAML file through `--config`, never on the command line ([Security](#security)) |
| `SERVED_NAME` | `glm-ocr` | model id on `/v1/models`; clients send it as `model` |
| `CONTEXT` | `16384` | `--context-length`: image tokens + prompt + `max_tokens` must fit ([Request format](#request-format)) |
| `MEMFRAC` | `0.6` | `--mem-fraction-static` |
| `MAXREQ` | *(unset)* | `--max-running-requests`. Unset, the patch uses 48 in self-spec mode; causal mode keeps SGLang's default |
| `GRAPHBS` | *(SGLang default)* | `--cuda-graph-max-bs`; lower it if CUDA-graph capture overflows the FlashInfer workspace at a large `MAXREQ` in self-spec mode |
| `DRAFT` | `33` | self-spec draft window K = `bd_size` + 1 (`--speculative-num-draft-tokens`) |
| `FAST` | `1` | the runtime optimizations in the patch. Both modes: GLM ViT as CUDA graphs, bf16 pixel transport, single-threaded CPU ops in the server, two tokenizer workers (one when `API_KEY` is set). Self-spec: mask-free attention plans, M-RoPE cache fix, GPU-side commit, host-side lengths, device-side positions, next draft launched ahead. `0` serves the unoptimized runtime; other values exit 2 |
| `TOKENIZER_WORKERS` | `2` with `FAST=1` and no `API_KEY`, else `1` | `--tokenizer-worker-num`. With one worker SGLang enforces keys and the launcher locks the management endpoints; with more, SGLang enforces no keys at all ([Security](#security)) |
| `OMP_NUM_THREADS_SERVER` | `1` | with `FAST=1`: OpenMP / MKL / torch threads in the server |
| `CUDA_HOME` | `/usr/local/cuda-12.8` | CUDA toolkit used for kernel compilation |
| `HF_HOME` | `~/.cache/huggingface` | Hugging Face cache |
| `EXTRA_ARGS` | *(empty)* | extra `sglang.launch_server` options from a fixed list of logging and capacity options, split on whitespace (no shell quoting), appended last; anything else exits 2 ([Security](#security)) |
| `SKIP_PREFLIGHT` | *(unset)* | `1` skips the startup checks: `$CUDA_HOME/bin/nvcc` reports release 12.x, `ninja` is in `VBIN`, `nvidia-smi` is on `PATH`, and every selected GPU has compute capability 9.0 (a failed check exits 1) |

`PORT`, `CONTEXT`, `DRAFT`, `MAXREQ`, `GRAPHBS`, `TOKENIZER_WORKERS` and `OMP_NUM_THREADS_SERVER` must be positive integers and `MEMFRAC` a fraction such as `0.6`; `SERVED_NAME` and `SERVE_HOST` must not start with `-` or contain spaces. Any other value exits 2.

Self-spec mode also exports these, each overridable from the environment: `SELFSPEC_FULLDRAFT=1` (all mask logits are draft candidates and a fully accepted draft commits the bonus token; `0` is the older round of at most K tokens), `SELFSPEC_POS_OFFSET=1` (the position layout GravityOCR's published results were produced with, one position after where causal decoding places the same token; `0` is the AR-aligned layout, which measured about the same draft acceptance during development), and `SELFSPEC_MASK_ID=59282` only when `CKPT` has no `block_diffusion.json`. `FAST=1` sets `SS_GLM_VIT_GRAPH_MAXM=2048` and `SS_GLM_PV_BF16=1`, and in self-spec mode `SELFSPEC_FASTATTN`, `SELFSPEC_MROPE_FIX`, `SELFSPEC_FAST_COMMIT`, `SELFSPEC_HOSTLEN`, `SELFSPEC_DEV_POS` and `SELFSPEC_DRAFT_AHEAD` to `1`. `SELFSPEC_EXACT_PENALTY` stays `0` unless you set it ([Known limits](#known-limits)).

`serve.sh` also handles four setup problems: it puts `VBIN` and `$CUDA_HOME/bin` on `PATH` (CUDA-graph capture shells out to `ninja`), runs `$VBIN/python` directly rather than through `uv run`, sets `VLLM_USE_FLASHINFER_SAMPLER=0` (that sampler's JIT path crashes without its own build setup), and serves with `HF_HUB_OFFLINE=1` after the one download. It also clears `PYTHONOPTIMIZE`, which would strip SGLang's `assert` checks. It does not pass `--trust-remote-code`, and `EXTRA_ARGS` cannot add it; GLM-OCR is implemented natively in transformers 5.x and SGLang.

## Verify

```bash
curl -s http://127.0.0.1:30000/v1/models                                     # lists "glm-ocr" once ready
"$VBIN/python" inference/gravityocr/ocr_request.py --synthetic 'Hello GravityOCR'
"$VBIN/python" inference/gravityocr/ocr_request.py table.png --prompt 'Table Recognition:'
```

`ocr_request.py` takes either an image file or `--synthetic TEXT`, which renders `TEXT` into a small PNG with Pillow, so the first check needs no sample image; the reply should read back the same text. It prints the recognized text on stdout and the token usage on stderr. It exits 1, with a one-line message, on an HTTP error, a refused or dropped connection, a reply slower than `--timeout`, or a reply without `choices[0].message.content`. Apart from `--synthetic` it uses only the standard library and runs on any Python 3.10+. Its flags: `--url` (default `http://127.0.0.1:30000`), `--model` (default `glm-ocr`), `--prompt` (default `Text Recognition:`), `--max-tokens` (default 4096), `--timeout` (default 300 s), and `--api-key` (default `$API_KEY`). With a key, add `-H "Authorization: Bearer $API_KEY"` to `curl`.

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

| prompt | region | output |
|---|---|---|
| `Text Recognition:` | a text block (paragraph, title, caption, list) | plain text in reading order |
| `Table Recognition:` | one table | HTML table |
| `Formula Recognition:` | one display formula | LaTeX |

- The reply text is `choices[0].message.content`; `usage` has the token counts.
- Send images as `data:` URLs (see [Security](#security)) and use `temperature: 0`, the setting this recipe was validated with. Keep `max_tokens` at 4096 or below.
- `CONTEXT` (16384) must hold the image tokens, the prompt and `max_tokens`. The image processor downscales every crop to at most 6,144 image tokens (its `longest_edge` of 9,633,792 pixels counts two temporal frames, so 4,816,896 pixels at 28 × 28 pixels per token; a 4000 × 4000 crop becomes about 6,100 prompt tokens), so any crop fits with `max_tokens` 4096. If you lower `CONTEXT` or raise `max_tokens`, SGLang checks the total when a request arrives and rejects one that cannot fit with HTTP 400 ("Requested token count exceeds the model's maximum context length"). Self-spec mode also reserves K + 1 KV slots per round, and a request within K + 1 tokens of `CONTEXT` has not been exercised on a GPU, so keep that margin. A higher `CONTEXT` costs KV-cache memory per request.

## Security

- **Loopback by default.** The server binds `127.0.0.1`. Binding any other address with `SERVE_HOST` requires `API_KEY`; without it `serve.sh` exits with status 2. Clients then send `Authorization: Bearer <key>`.
- **Keys need one tokenizer worker.** SGLang v0.5.10 installs its key check only when it runs a single tokenizer worker. With more, its HTTP workers accept every request and refuse to start if a key is set. `serve.sh` therefore runs one worker whenever `API_KEY` is set (also with `FAST=1`) and exits 2 if `TOKENIZER_WORKERS` asks for more.
- **Management endpoints.** SGLang serves management endpoints next to the OpenAI API: `/update_weights_from_disk` (which loads weights from a path the caller chooses), the other weight-update routes, `/flush_cache`, `/start_profile`, `/load_lora_adapter`, `/set_internal_state` and more. Whenever the server runs one tokenizer worker (`API_KEY` set, `FAST=0`, or `TOKENIZER_WORKERS=1`), `serve.sh` also gives SGLang a random admin key that it never prints, so these endpoints refuse every caller; `API_KEY` does not unlock them. With the `FAST=1` default of two workers and no key, SGLang enforces no keys and these endpoints are open to every user on the host, which `serve.sh` points out when it starts. On a shared machine set `API_KEY`, or `TOKENIZER_WORKERS=1` if loopback clients need no key.
- **Keys stay off the command line.** `serve.sh` takes `API_KEY` only from the environment and writes it, with the admin key, to a mode-0600 YAML file in a private temporary directory, which it passes with `--config`; neither key shows up in `ps` or `/proc/<pid>/cmdline`. A background watcher normally removes the directory when the server exits; if the watcher is killed as well (for example by a `SIGKILL` to the whole process group), delete the leftover `gravityocr-serve.*` directory under `$TMPDIR` (default `/tmp`) yourself. SGLang does log its full server arguments at startup, keys included, so keep the server log private or add `EXTRA_ARGS="--log-level warning"`.
- **Do not expose the port directly, even with a key.** SGLang loads `image_url` values on the server: an `http(s)://` URL makes the server fetch it, from wherever the host can reach, and a `file://` URL or a bare path makes it read a local image file. `/health*` and `/metrics` answer without a key. Put the server behind a reverse proxy that authenticates callers, forwards only `/v1/chat/completions` and `/v1/models`, and only requests whose image URLs are `data:` URLs. `ocr_request.py` sends `data:` URLs only.
- **`EXTRA_ARGS` takes a fixed list.** Every `EXTRA_ARGS` token that starts with `-` must be one of these exact option names, bare or as `--name=value`; abbreviations and anything else exit 2:

  `--log-level` `--log-level-http` `--log-requests` `--log-requests-level` `--enable-metrics` `--enable-cache-report` `--decode-log-interval` `--show-time-cost` `--enable-request-time-stats-logging` `--max-total-tokens` `--max-prefill-tokens` `--chunked-prefill-size` `--max-queued-requests` `--schedule-policy` `--schedule-conservativeness` `--num-continuous-decode-steps` `--stream-interval` `--cuda-graph-bs` `--disable-cuda-graph` `--disable-cuda-graph-padding` `--disable-radix-cache` `--disable-overlap-schedule` `--skip-server-warmup` `--watchdog-timeout` `--soft-watchdog-timeout` `--random-seed`

  Options that `serve.sh` sets from its own variables (`--host`, `--port`, `--context-length`, `--tokenizer-worker-num`, ...) are refused with the variable to use instead. Among the rest, these matter for security:

  | refused | why |
  |---|---|
  | `--config` | merges arbitrary options from a YAML file. `serve.sh` adds its own `--config` for the keys, after every check |
  | `--host`, `--api-key`, `--admin-api-key` | the bind address and keys are set only through `SERVE_HOST` and `API_KEY` |
  | `--enable-custom-logit-processor` | makes the server deserialize client-supplied logit processors with `dill`: remote code execution, with no upstream fix |
  | `--enable-lora`, `--lora-paths` and the other `--lora*` options | LoRA is not part of this recipe. `--lora-paths` alone turns it on, and the LoRA manager has a known request-triggered crash |
  | `--grpc-mode`, `--encoder-only`, `--use-ray` | other front-ends that serve on `SERVE_HOST:PORT` without SGLang's key check |
  | `--enable-dp-attention`, `--dist-init-addr`, `--nnodes`, `--node-rank` | move SGLang's internal pickle-based IPC from unix sockets to TCP |
  | `--trust-remote-code`, `--load-format` | would run code from the checkpoint, or load its weights with a pickle-based loader |

  For anything beyond the list, run `sglang.launch_server` yourself.
- **Known advisories.** The pinned server environment carries advisories that have no fix on this base (in SGLang v0.5.10, torch 2.9.1, transformers 5.3.0 and diskcache). Each one is recorded in [`docs/supply-chain/exceptions.md`](../../docs/supply-chain/exceptions.md), with whether it is reachable here and the mitigation this launcher relies on. Report vulnerabilities through [`SECURITY.md`](../../SECURITY.md), not a public issue.

## Known limits

- **Hardware.** Validated only on H100 (compute capability 9.0) with driver 570 and a CUDA 12.8 toolkit.
- **Pinned base.** The patch is a diff against SGLang v0.5.10 (`1519acf`) and does not apply to later releases without porting. FlashInfer must be exactly 0.6.7.post2: the fast attention path calls its planning internals and checks the version.
- **Self-spec and causal can differ.** Self-spec checks every draft token against the model's AR prediction, so each committed token is its greedy AR choice given the tokens committed before it. The draft and verify passes, though, run other kernels and batch shapes than causal decoding. Under bf16, a near-tie between two tokens can then resolve differently, and the outputs diverge from that token on. Compare the modes on token ids over many crops, not on one. The same mode can also resolve such a near-tie differently from one server run to the next, since SGLang runs without deterministic inference by default: in two `DP=2` causal runs over the same 100 crops, one crop came out differently in one run (it stopped before a trailing `\tag`).
- **Repetition penalty.** In speculative verification SGLang penalizes every position of a block using only the tokens committed before the block, so a token repeated inside the block escapes the penalty that causal decoding would apply. With `repetition_penalty` other than 1, set `SELFSPEC_EXACT_PENALTY=1`, which adds the in-block part for `repetition_penalty` (not for frequency or presence penalties). It is off by default.
- **Sampling.** Validated with greedy decoding only (`temperature: 0`).
- **Crops only.** No page layout, no PDF input, no page assembly.
- **Batch caps.** Self-spec reserves K + 1 KV slots per running request each round. At a large `MAXREQ`, CUDA-graph capture can overflow the FlashInfer workspace; lower `MAXREQ` or `GRAPHBS`. With `FAST=1` the ViT CUDA graphs lengthen startup and take some GPU memory from the KV cache.
- **No GPU coverage in CI.** The tests below check files, launcher arguments and request payloads. Nothing in CI builds the patched SGLang or starts a server, so a change to the patch or the lock needs a manual GPU check. The patch and launcher were validated on H100 in the environment frozen in the GravityOCR repository. `sglang/requirements-lock.txt` raises 11 pins above that environment for security fixes; a fresh `setup.sh` run with it on H100 (2026-10-30) gave the same greedy outputs as that environment on 100 OCR crops, in self-spec and causal mode.

## Tests

```bash
pytest -q inference/gravityocr/      # CPU only, no downloads; also runs under `pytest -q` from the repo root
```

| file | checks |
|---|---|
| `test_gravityocr_patch.py` | `sglang/BASE_COMMIT.txt` is a full commit SHA, and the pins quoted in this README, `sglang/setup.sh` and `NOTICE` match it; the patch covers 25 files (24 upstream sources plus `python/pyproject.toml`); the four new files parse as Python 3.10 and carry the licence header; `sglang/setup.sh` is valid bash and installs with `--no-deps --no-config` from explicit indexes; the lock names the base tag, holds exactly the four `+cu129` builds, and pins the versions `setup.sh` checks |
| `test_gravityocr_launcher.py` | runs `serve.sh` against a stub `VBIN` and a local checkpoint directory: mode flags, the `SELFSPEC_*` environment, `DP`, value checks, the `SERVE_HOST` / `API_KEY` rule, one tokenizer worker with a key, the admin key and the private key file, the `EXTRA_ARGS` allowlist, and the preflight checks with stub `nvcc` / `nvidia-smi` |
| `test_gravityocr_request.py` | `ocr_request.py` with `urlopen` replaced: the payload (`[image_url, text]` order, temperature 0, a `data:` URL that decodes back to the input bytes), CLI defaults and argument checks, an `Authorization` header only when a key is given, HTTP errors, refused and dropped connections, timeouts and malformed replies (exit 1), and `--synthetic` rendering a PNG |
| `test_gravityocr_mac.py` | `serve_mac.py` with a stub engine: the three task prompts, the payload `ocr_request.py` sends, refusal of non-`data:` image URLs and other bad requests (400), an engine error answered with 500 while the server keeps running, and refusal of a non-loopback `--host` or a missing Gravity app |

On a GPU, serve both modes and send each the same crops:

```bash
bash inference/gravityocr/serve.sh &                               # self-spec on GPU 0, port 30000
MODE=causal GPU=1 PORT=30001 bash inference/gravityocr/serve.sh &   # causal on GPU 1, port 30001
"$VBIN/python" inference/gravityocr/ocr_request.py crop.png --url http://127.0.0.1:30000
"$VBIN/python" inference/gravityocr/ocr_request.py crop.png --url http://127.0.0.1:30001
```

## Tutorial notebook

[`getting_started.ipynb`](getting_started.ipynb) serves GravityOCR, reads a text, a table and a formula
crop, and compares self-spec with causal decoding, in about 5–10 minutes after a one-time setup. It
runs on Linux with an L40S, L40 or H100 (`serve.sh`) or on an Apple Silicon Mac (`serve_mac.py`, see
[On a Mac](#on-a-mac)), and is driven by environment variables:

| variable | default | meaning |
|---|---|---|
| `BACKEND` | `auto` | `sglang` (Linux) or `mac`; `auto` picks `mac` on an Apple Silicon Mac |
| `GPU`, `CAUSAL_GPU` | `auto` | the least-loaded L40S / L40 / H100, and a second one for the causal server if there is one (else they take turns) |
| `PAGE_LAYOUT` | `1` | `0` skips section 5 (a whole page with PP-DocLayoutV3; installs `torch` CPU, `transformers` ≥ 5.1 and OpenCV into the kernel) |
| `COMPARE_CAUSAL` | `1` | `0` skips section 6 |
| `SGLANG_SRC`, `VBIN` | `~/src/sglang-gravityocr` | where `sglang/setup.sh` built the server environment |
| `GRAVITYOCR_MODEL` | `trillionlabs/GravityOCR` | Hub id or local checkpoint |
| `GRAVITY_APP` | `/Applications/Gravity.app` | the Gravity app, for `BACKEND=mac` |
| `PORT`, `CAUSAL_PORT` | `30000`, `30001` | the two servers' ports |
| `CUDA_HOME` | `$CUDA_HOME`, else `/usr/local/cuda-12.8` or `/usr/local/cuda`, whichever has `nvcc` | a CUDA 12.x toolkit |
| `API_KEY` | as for `serve.sh` | |

The first code cell checks the kernel: Python 3.10 or newer, with Pillow and matplotlib. It installs
whichever is missing into that kernel, with uv if there is one, so any Python ≥ 3.10 kernel works,
including an environment that has no `pip`.

**On an L40S or L40.** `serve.sh` accepts only compute capability 9.0 (H100); the L40S and the L40
are 8.9. For them the notebook runs `serve.sh`'s other startup checks itself (a CUDA 12.x `nvcc`,
`ninja` in `VBIN`) and passes `SKIP_PREFLIGHT=1`. The patch's vision-encoder CUDA graphs need FA3, which
is Hopper-only, so they switch themselves off; the other `FAST=1` paths run. On an L40S (Brev
`l40s-48gb.1x`, CUDA 12.6 at `/usr/local/cuda`) the whole notebook ran in 2.7 minutes, the first server
start took 90 s, and self-spec was 3.5–5.7× faster than causal with identical text on all four crops.
The L40 was not tested; it has the same compute capability and memory.

### With a coding agent

Settings come from the environment and every server stops when the kernel exits, even after a failed
cell, so an agent can run the notebook headless. From `inference/gravityocr/`:

```bash
uv run --no-project --python 3.12 --with jupyter --with pillow --with matplotlib \
  jupyter nbconvert --to notebook --execute getting_started.ipynb --output-dir /tmp/gravityocr-run
```

`--no-project` keeps `uv` from installing the repository's own dependencies, and `--output-dir` keeps
the executed copy out of the repository.

- **Claude Code** (`claude`): approve the commands it asks to run. If you turned on its sandbox, let
  this command run outside it.
- **Codex**: start it with `codex -s danger-full-access -a on-request` (it still asks before commands
  that need approval). Its default `workspace-write` sandbox blocks the network, including the
  servers on `127.0.0.1`, and writes to `~/.cache`. On a Mac it also blocks the GPU: with network
  access and `~/.cache` allowed, the Mac backend still fails with *No Metal device available* (tested
  with Codex CLI 0.166).

## On a Mac

`serve_mac.py` answers the same requests as `serve.sh` on an Apple Silicon Mac, for trying the model
and running [`getting_started.ipynb`](getting_started.ipynb) without an NVIDIA GPU. It is not a
second serving recipe. It runs the MLX engine of the [Gravity](https://gravityocr.trillionlabs.co) app,
which must be installed, with Gravity's bundled Python, and downloads `trillionlabs/GravityOCR` once
into `~/.cache/gravityocr-mac/` (pinned to a commit, SHA-256 checked):

```bash
/Applications/Gravity.app/Contents/Resources/runtime/bin/python -I -B inference/gravityocr/serve_mac.py
python3 inference/gravityocr/ocr_request.py --synthetic 'Hello GravityOCR'
```

Its numbers are not this recipe's: MLX instead of SGLang, text layers quantized to 6 bits, a 16-token
draft block instead of 32, crops downscaled to a 1344 px long side, and Gravity's own per-task token
limits (`max_tokens` is ignored). `--mode causal` swaps in the autoregressive decoder that ships with
Gravity. It binds loopback only, accepts `data:` image URLs only, and imports Gravity's internal
`ocr.engine` package, which is not a public API: it was written against Gravity 0.3.4 and warns on
other versions.

## Files

```
inference/gravityocr/
  README.md, MODEL_CONTRACT.md, NOTICE
  serve.sh                   launcher: MODE=self-spec | causal, env-driven
  ocr_request.py             one-crop OpenAI-compatible client (stdlib; Pillow for --synthetic)
  serve_mac.py               Mac option: the same API on the Gravity app's MLX engine (see below)
  getting_started.ipynb      5-10 minute tutorial; runs on an L40S / L40 / H100 or on a Mac
  test_gravityocr_*.py       CPU tests
  sglang/
    BASE_COMMIT.txt            upstream sgl-project/sglang pin (tag v0.5.10)
    gravityocr_selfspec.patch  diff against that commit
    python/sglang/srt/...      the four new files the patch needs
    setup.sh                   fetch the pin -> patch -> venv from the lock; prints VBIN
    requirements.in, overrides.txt, requirements-lock.txt   the server environment
```

## Provenance and license

- **Code.** Apache-2.0, like the rest of this repo ([`LICENSE`](../../LICENSE)). It comes from [`trillion-labs/GravityOCR`](https://github.com/trillion-labs/GravityOCR), which Trillion Labs also publishes. There, the SGLang patch and the four new SGLang files are likewise Apache-2.0, and the launcher and client are MIT; Trillion Labs contributes them here under Apache-2.0.
- **Upstream SGLang.** [`sgl-project/sglang`](https://github.com/sgl-project/sglang), Apache-2.0, at `1519acf37c23f2189adb93f57ca9cd2db1bebf18` (tag v0.5.10). It is not vendored: `setup.sh` fetches it. This folder carries only the patch, which modifies 24 upstream source files and `python/pyproject.toml`, and the four new files.
- **Weights.** [`trillionlabs/GravityOCR`](https://huggingface.co/trillionlabs/GravityOCR) is MIT-licensed and fine-tuned from [`zai-org/GLM-OCR`](https://huggingface.co/zai-org/GLM-OCR) (MIT). Neither is redistributed here; `serve.sh` downloads the checkpoint from the Hub.
- **Design notes.** Comments in the new SGLang files mention "natcache", `src/fast_sampling.py` and `docs/SGLANG_SELFSPEC_IMPL.md`. These are the in-process reference decoder and the implementation notes in the [GravityOCR repository](https://github.com/trillion-labs/GravityOCR), which is the place to read about the design. Some of those comments also record the checks used while the patch was developed, such as matching causal decoding token for token on a test set; they describe how the code was tested, not a guarantee for every input (see [Known limits](#known-limits)). The header of `selfspec_diffusion_worker.py` says the same.

Attribution for every component is in [`NOTICE`](NOTICE); see also the repo's [`NOTICE`](../../NOTICE) and [`COMPLIANCE.md`](../../COMPLIANCE.md). What a checkpoint needs in order to be servable is in [`MODEL_CONTRACT.md`](MODEL_CONTRACT.md).
