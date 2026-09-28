# SGLang diffusion backend — clone → patch → install → shape

`inference/serve.py` is a thin wrapper: it maps `--mode` to a decode algorithm +
config and launches an SGLang server. The heavy lifting (batching, KV cache, the
two-stream decode kernels) lives in an SGLang backend that is **not vendored in
this repo**. This directory is the reproducible recipe for building that backend
from its public upstream, so `--mode diffusion` / `self-spec` / `causal` work
against a two-stream Qwen3.5 checkpoint.

## What the patch adds

`two_stream_diffusion.patch` is a diff against the public
[`yuchen-zhu-zyc/HybridDiffusion`](https://github.com/yuchen-zhu-zyc/HybridDiffusion)
serving stack (a modified SGLang runtime). It adds the **two-stream Gated DeltaNet
block-end readout** and the **block-diffusion decode** algorithm needed to serve a
Qwen3.5 two-stream checkpoint — i.e. the `LowConfidenceShiftHybridDiffusion` and
`HybridDiffusionSelfSpec` decode paths that `serve.py`'s `diffusion` and
`self-spec` modes target. It touches 14 files under `eval/sglang/srt/…` (config
registration, the dllm decode algorithms + scheduler mixin, the GDN attention
backend, the Qwen3.5 model/logits path, and the CUDA-graph / KV-cache plumbing).

## Recipe

### 1. Clone the pinned upstream

```bash
git clone https://github.com/yuchen-zhu-zyc/HybridDiffusion
cd HybridDiffusion
git checkout 6ca547a          # "Initial public release" — the pinned base commit
```

### 2. Apply the patch

```bash
git apply /path/to/inference/sglang/two_stream_diffusion.patch
```

Verify before applying with `git apply --check <patch>` (exit 0 = applies
cleanly). The patch is a diff against this exact commit; apply it from the repo
root.

### 3. Install per upstream's instructions

Follow the upstream `eval/` install (validated on Linux / Python 3.10 / NVIDIA
CUDA / H100, uv-based). From the patched `eval/` directory:

```bash
cd eval
HYBRID_DIFFUSION_CACHE_ROOT=/persistent/hybrid-diffusion-cache \
  bash scripts/setup_eval_env.sh
```

This builds a venv (default
`$HYBRID_DIFFUSION_CACHE_ROOT/venvs/hybrid-diffusion-eval`) with the patched
SGLang runtime and its matching FlashInfer source installed. That venv's
`bin/python` is the interpreter that has the patched backend importable — it is
what `serve.py` needs as `SGLANG_PYTHON` below. See the upstream `eval/README.md`
for env knobs and checkpoint download.

### 4. Shape the checkpoint for SGLang

SGLang loads a two-stream Qwen3.5 checkpoint through the
`Qwen3_5DLLMForConditionalGeneration` architecture (the text backbone nested in a
vision-language wrapper — the vision tower is built but never used for text). A
*raw* training checkpoint instead ships a flat `Qwen3_5ForCausalLM` /
`model_type: qwen3_5_text` config, which this SGLang build has no implementation
for: serving it raw fails at load (`causal`) or at the first decode block
(`diffusion`, with `missing the decode seed`).

`tools/shape_for_sglang.py` fixes this by writing a lightweight *shim* directory:
a rewritten `config.json` plus symlinks of the (unchanged) weights and tokenizer.
Run it once per checkpoint:

```bash
python inference/sglang/tools/shape_for_sglang.py <RAW_CKPT> [<OUT>] [--copy]
# default <OUT> is <RAW_CKPT>_sglang; --copy instead of symlink for a portable dir
```

This is an SGLang-only step — vLLM registers the raw architecture directly and
needs no shaping.

### 5. Point nano-inference at the patched backend

`serve.py` launches `python -m sglang.launch_server` using `$SGLANG_PYTHON`
(defaulting to the current `python`). Set it to the venv interpreter from step 3
and serve the **shaped** directory from step 4:

```bash
export SGLANG_PYTHON=/persistent/hybrid-diffusion-cache/venvs/hybrid-diffusion-eval/bin/python

# <CKPT> = the shaped directory from step 4 (…_sglang)
python inference/serve.py <CKPT> --mode diffusion  --port 30000
python inference/serve.py <CKPT> --mode self-spec  --port 30000
python inference/serve.py <CKPT> --mode causal     --port 30000
```

Add `--dry-run` to print the launch command without starting the server.

Once it is up, confirm it generates (SGLang serves the same OpenAI-compatible API).
SGLang uses the checkpoint you launched with as the model id, so pass `<CKPT>` as `model`:

```bash
CKPT=<same repo-id-or-path you served>
curl -s http://localhost:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$CKPT\",\"messages\":[{\"role\":\"user\",\"content\":\"What is 17 * 24?\"}],\"max_tokens\":64}" \
  | python -m json.tool
```

A JSON reply with `choices[0].message.content` means the backend is serving.
(The model id the server exposes is listed at `GET /v1/models`.)

The `diffusion` and `self-spec` modes reference the shipped decode configs in
[`../configs/`](../configs/) — `trida_diffusion_b4.yaml` and
`trida_self_spec_b7_g4.yaml` respectively (override with `--config`). `causal`
needs no dllm config. See [`../README.md`](../README.md) for what each mode does
and [`../MODEL_CONTRACT.md`](../MODEL_CONTRACT.md) for what makes a checkpoint
compatible.

## Provenance

- **Base:** `yuchen-zhu-zyc/HybridDiffusion` @ `6ca547a` ("Initial public
  release").
- **Patch:** `two_stream_diffusion.patch` — pure code, no vendored weights.

See [`NOTICE`](NOTICE) for attribution and licensing.
