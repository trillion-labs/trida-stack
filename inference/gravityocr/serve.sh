#!/bin/bash
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# Serve trillionlabs/GravityOCR (a GLM-OCR checkpoint with a block-diffusion head) over the
# OpenAI-compatible HTTP API of upstream SGLang v0.5.10 plus the GravityOCR self-speculative patch
# (sglang/). One request is one layout region (an image crop) plus a task prompt; see ocr_request.py.
#
# Build the patched server environment once with sglang/setup.sh; it prints the VBIN to use here.
#     VBIN=<venv>/bin bash inference/gravityocr/serve.sh                  # self-spec (default), GPU 0
#     VBIN=<venv>/bin MODE=causal bash inference/gravityocr/serve.sh      # plain autoregressive decoding
#     VBIN=<venv>/bin DP=2 GPU=0,1 bash inference/gravityocr/serve.sh     # 2 data-parallel replicas, 1 port
#     bash inference/gravityocr/serve.sh VBIN=<venv>/bin PORT=30001       # KEY=VALUE arguments work too
#
# Everything is env-driven; KEY=VALUE arguments are exported first. Main knobs:
#   MODE          self-spec (default) | causal
#   CKPT          local checkpoint dir or Hub id (default: $GRAVITYOCR_MODEL, else trillionlabs/GravityOCR)
#   VBIN          bin/ directory of the patched SGLang venv (required)
#   GPU, DP       CUDA_VISIBLE_DEVICES list (default 0); data-parallel replicas (default 1, one GPU each)
#   SERVE_HOST    bind address (default 127.0.0.1); a non-loopback address requires API_KEY
#   PORT          default 30000
#   SERVED_NAME   model name clients send and /v1/models reports (default glm-ocr)
#   CONTEXT       --context-length (default 16384)
#   MEMFRAC       --mem-fraction-static (default 0.6)
#   API_KEY       optional bearer key; set it in the environment, never as an argument. With a key
#                 the server runs one tokenizer worker, the only mode in which SGLang checks keys
#   EXTRA_ARGS    extra SGLang logging / capacity options from a fixed list, appended last
#   SKIP_PREFLIGHT=1   skip the CUDA toolkit / ninja / GPU checks
#
# Verify:  curl -s http://127.0.0.1:30000/v1/models
#          "$VBIN/python" inference/gravityocr/ocr_request.py --synthetic "Hello GravityOCR"
set -euo pipefail

die() { local code=$1; shift; echo "[serve] error: $*" >&2; exit "$code"; }
usage() { sed -n '/^# Serve /,/^set -euo pipefail$/{/^set /d;s/^# \{0,1\}//;p;}' "$0"; }

for kv in "$@"; do
  case "$kv" in
    -h|--help) usage; exit 0 ;;
    API_KEY=*) die 2 "set API_KEY in the environment, not as an argument (arguments show up in the process list)" ;;
  esac
  [[ $kv =~ ^[A-Za-z_][A-Za-z0-9_]*= ]] || die 2 "arguments must be KEY=VALUE (got '$kv'); see --help"
  export "${kv?}"
done

# ---- config (env-driven) ---------------------------------------------------------------------
MODE="${MODE:-self-spec}"
case "$MODE" in
  self-spec|causal) ;;
  *) die 2 "MODE must be self-spec | causal (got '$MODE')" ;;
esac
CKPT="${CKPT:-${GRAVITYOCR_MODEL:-trillionlabs/GravityOCR}}"
VBIN="${VBIN:-}"
[ -n "$VBIN" ] || die 2 "set VBIN=<patched SGLang venv>/bin (sglang/setup.sh prints it)"
[ -x "$VBIN/python" ] || die 2 "VBIN=$VBIN has no executable python"
GPU="${GPU:-0}"
DP="${DP:-1}"
PORT="${PORT:-30000}"
SERVE_HOST="${SERVE_HOST:-127.0.0.1}"   # deliberately not HOST: some shells export HOST=<hostname>
SERVED_NAME="${SERVED_NAME:-glm-ocr}"
# Image tokens + prompt + max_tokens must fit. Very large crops with a large max_tokens can still
# exceed 16384: raise CONTEXT or lower max_tokens.
CONTEXT="${CONTEXT:-16384}"
MEMFRAC="${MEMFRAC:-0.6}"
DRAFT="${DRAFT:-33}"
MAXREQ="${MAXREQ:-}"
GRAPHBS="${GRAPHBS:-}"
TOKENIZER_WORKERS="${TOKENIZER_WORKERS:-}"
SERVER_THREADS="${OMP_NUM_THREADS_SERVER:-1}"
FAST="${FAST:-1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
API_KEY="${API_KEY:-}"
CUDA_HOME_DIR="${CUDA_HOME:-/usr/local/cuda-12.8}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"

# ---- value checks: these values end up on SGLang's command line --------------------------------
case "$FAST" in 0|1) ;; *) die 2 "FAST must be 0 or 1 (got '$FAST')" ;; esac
for v in DP PORT CONTEXT DRAFT SERVER_THREADS MAXREQ GRAPHBS TOKENIZER_WORKERS; do
  val="${!v:-}"
  case "$v" in MAXREQ|GRAPHBS|TOKENIZER_WORKERS) [ -n "$val" ] || continue ;; esac
  [ "$v" = SERVER_THREADS ] && v=OMP_NUM_THREADS_SERVER
  [[ $val =~ ^[1-9][0-9]{0,8}$ ]] || die 2 "$v must be a positive integer (got '$val')"
done
[ "$PORT" -le 65535 ] || die 2 "PORT must be at most 65535 (got '$PORT')"
[[ $MEMFRAC =~ ^(0?\.[0-9]+|1(\.0+)?)$ ]] || die 2 "MEMFRAC must be a fraction such as 0.6 (got '$MEMFRAC')"
[[ $SERVED_NAME =~ ^[A-Za-z0-9][A-Za-z0-9._/:+-]*$ ]] \
  || die 2 "SERVED_NAME must start with a letter or digit and hold no spaces (got '$SERVED_NAME')"
[[ $SERVE_HOST =~ ^[A-Za-z0-9:][A-Za-z0-9.:%_-]*$ ]] || die 2 "SERVE_HOST must be a host name or IP address (got '$SERVE_HOST')"
case "$CKPT" in -*) die 2 "CKPT must not start with '-' (got '$CKPT')" ;; esac
IFS=, read -r -a GPUS <<< "$GPU"
for g in "${GPUS[@]}"; do
  [[ $g =~ ^[A-Za-z0-9][A-Za-z0-9:_-]*$ ]] || die 2 "GPU must be a comma-separated device list without spaces (got '$GPU')"
done
[ "${#GPUS[@]}" -eq "$DP" ] \
  || die 2 "GPU=$GPU lists ${#GPUS[@]} device(s) but DP=$DP: list exactly one GPU per data-parallel replica"

# ---- network exposure and tokenizer workers ----------------------------------------------------
# The default bind is loopback. Any other address requires API_KEY. SGLang v0.5.10 checks keys only
# with a single tokenizer worker: with more, its HTTP workers install no auth middleware and refuse
# to start when a key is set. So a key means one worker; FAST=1 otherwise runs two.
is_loopback() { [[ $1 =~ ^127\.[0-9]+\.[0-9]+\.[0-9]+$ || $1 == localhost || $1 == ::1 ]]; }
if ! is_loopback "$SERVE_HOST" && [ -z "$API_KEY" ]; then
  die 2 "SERVE_HOST=$SERVE_HOST is not a loopback address: set API_KEY so every request must authenticate, or keep SERVE_HOST=127.0.0.1"
fi
if [ -n "$API_KEY" ]; then
  key_re='^[!-~]+$'   # printable ASCII, no spaces (checked in the C locale)
  ( LC_ALL=C; [[ $API_KEY =~ $key_re && $API_KEY != -* ]] ) \
    || die 2 "API_KEY must be printable ASCII without spaces and must not start with '-'"
fi
if [ "$FAST" = 1 ] && [ -z "$API_KEY" ]; then WORKERS="${TOKENIZER_WORKERS:-2}"; else WORKERS="${TOKENIZER_WORKERS:-1}"; fi
if [ -n "$API_KEY" ] && [ "$WORKERS" != 1 ]; then
  die 2 "API_KEY needs a single tokenizer worker (got TOKENIZER_WORKERS=$WORKERS): SGLang v0.5.10 checks keys only then"
fi

# ---- EXTRA_ARGS: logging and capacity options only ----------------------------------------------
# Every EXTRA_ARGS token that starts with '-' must be one of these exact option names, bare or as
# --name=value; abbreviations are refused. Everything else stays out: what this launcher sets (bind
# address, keys, port, model, mode flags), --config files, other front-ends and transports (gRPC,
# encoder-only, Ray, DP attention, multi-node), model and weight loading, LoRA, grammar backends and
# custom logit processors. For anything beyond this list, run sglang.launch_server yourself.
EXTRA_OK=(--log-level --log-level-http --log-requests --log-requests-level --enable-metrics
          --enable-cache-report --decode-log-interval --show-time-cost --enable-request-time-stats-logging
          --max-total-tokens --max-prefill-tokens --chunked-prefill-size --max-queued-requests
          --schedule-policy --schedule-conservativeness --num-continuous-decode-steps --stream-interval
          --cuda-graph-bs --disable-cuda-graph --disable-cuda-graph-padding --disable-radix-cache
          --disable-overlap-schedule --skip-server-warmup --watchdog-timeout --soft-watchdog-timeout
          --random-seed)
read -r -d '' -a EXTRA <<< "$EXTRA_ARGS" || true   # whitespace split, no glob expansion
for tok in ${EXTRA[@]+"${EXTRA[@]}"}; do
  case "$tok" in -*) ;; *) continue ;; esac
  name="${tok%%=*}"
  case " ${EXTRA_OK[*]} " in *" $name "*) continue ;; esac
  case "$name" in
    --host) hint=SERVE_HOST ;; --port) hint=PORT ;; --api-key) hint=API_KEY ;; --model-path) hint=CKPT ;;
    --served-model-name) hint=SERVED_NAME ;; --context-length) hint=CONTEXT ;; --mem-fraction-static) hint=MEMFRAC ;;
    --max-running-requests) hint=MAXREQ ;; --cuda-graph-max-bs) hint=GRAPHBS ;; --dp-size) hint=DP ;;
    --tokenizer-worker-num) hint=TOKENIZER_WORKERS ;; --speculative-num-draft-tokens) hint=DRAFT ;;
    *) hint="" ;;
  esac
  die 2 "EXTRA_ARGS may not contain '$tok'${hint:+ (set $hint instead)}. It accepts only: ${EXTRA_OK[*]}"
done

# ---- environment gotchas (each one cost a debugging cycle) -------------------------------------
# 1. VBIN on PATH: FlashInfer's CUDA-graph capture shells out to `ninja`, which lives in the venv;
#    without it capture dies with "FileNotFoundError: ninja". Run $VBIN/python directly, not
#    `uv run`, which can pick up a different (nested) venv that lacks the dependencies.
# 2. VLLM_USE_FLASHINFER_SAMPLER=0: otherwise the FlashInfer sampler JIT path needs ninja and a
#    build config and crashes.
# 3. CUDA_HOME: FlashInfer and Triton compile kernels at startup with nvcc, so a CUDA 12.x toolkit
#    must be installed. The wheels are CUDA 12 builds (validated with driver 570 / CUDA 12.8).
# 4. HF_HUB_OFFLINE=1: a Hub id is fetched once below, then the server loads the local snapshot, so
#    replicas never hit Hub rate limits.
# PYTHONOPTIMIZE is cleared: SGLang guards some invariants with assert statements.
export PATH="$VBIN:$CUDA_HOME_DIR/bin:$PATH"
export VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_HOME="$CUDA_HOME_DIR" HF_HUB_OFFLINE=1 HF_HOME="$HF_CACHE"
unset PYTHONOPTIMIZE

# ---- preflight (SKIP_PREFLIGHT=1 to bypass) ----------------------------------------------------
# Fails fast on boxes where the server would otherwise die minutes into startup. The patch's
# FlashInfer / FA3 paths are validated only on Hopper (compute capability 9.0). GPU ids are checked
# with nvidia-smi, which numbers devices in PCI order (set CUDA_DEVICE_ORDER=PCI_BUS_ID on mixed boxes).
if [ "${SKIP_PREFLIGHT:-0}" != 1 ]; then
  nvcc="$CUDA_HOME_DIR/bin/nvcc"
  [ -x "$nvcc" ] || die 1 "no nvcc at $nvcc: set CUDA_HOME to a CUDA 12.x toolkit (kernels are JIT-compiled at startup)"
  nvcc_out=$("$nvcc" --version 2>&1) || die 1 "'$nvcc --version' failed"
  [[ $nvcc_out =~ release\ ([0-9]+)\.([0-9]+) ]] || die 1 "could not read the CUDA release from '$nvcc --version'"
  [ "${BASH_REMATCH[1]}" = 12 ] \
    || die 1 "$nvcc is CUDA ${BASH_REMATCH[1]}.${BASH_REMATCH[2]}; this server needs a CUDA 12.x toolkit (point CUDA_HOME at one)"
  [ -x "$VBIN/ninja" ] || die 1 "no ninja in VBIN=$VBIN: FlashInfer needs it for CUDA-graph capture (rerun sglang/setup.sh)"
  command -v nvidia-smi >/dev/null 2>&1 || die 1 "nvidia-smi not found: cannot check the GPUs (SKIP_PREFLIGHT=1 to bypass)"
  for g in "${GPUS[@]}"; do
    cap=$(nvidia-smi -i "$g" --query-gpu=compute_cap --format=csv,noheader 2>/dev/null) || cap=""
    cap="${cap%%$'\n'*}"; cap="${cap//[[:space:]]/}"
    [ "$cap" = 9.0 ] \
      || die 1 "GPU $g has compute capability '${cap:-unknown}'; this recipe is validated only on Hopper (9.0) GPUs"
  done
fi

# ---- checkpoint --------------------------------------------------------------------------------
# A Hub id is fetched once here (nothing is downloaded when it is already cached) and served from the
# local snapshot, so the server itself stays offline (gotcha 4) and block_diffusion.json is found.
if [ ! -d "$CKPT" ]; then
  case "$CKPT" in /*|./*|../*|'~'*) die 2 "CKPT=$CKPT is not a directory" ;; esac
  CKPT=$(HF_HUB_OFFLINE=0 "$VBIN/python" -c \
    'import sys; from huggingface_hub import snapshot_download; print(snapshot_download(sys.argv[1]))' "$CKPT") \
    || die 1 "could not fetch $CKPT from the Hugging Face Hub (set CKPT to a local checkpoint dir to serve offline)"
  CKPT="${CKPT##*$'\n'}"
  [ -d "$CKPT" ] || die 1 "the Hub download did not return a directory (got '$CKPT')"
fi

# ---- server arguments --------------------------------------------------------------------------
# No --trust-remote-code: GLM-OCR (model_type glm_ocr) is built into transformers 5.x, so no code from
# the checkpoint runs, and EXTRA_ARGS cannot add the flag.
ARGS=(-m sglang.launch_server
      --model-path "$CKPT" --served-model-name "$SERVED_NAME" --host "$SERVE_HOST" --port "$PORT"
      --attention-backend flashinfer --mem-fraction-static "$MEMFRAC" --context-length "$CONTEXT"
      --tokenizer-worker-num "$WORKERS")
# MAXREQ: --max-running-requests (self-spec defaults to 48, causal to SGLang's own default).
if [ -n "$MAXREQ" ]; then ARGS+=(--max-running-requests "$MAXREQ"); fi
# GRAPHBS (opt-in): --cuda-graph-max-bs; unset keeps SGLang's default. In self-spec mode a sweep at
# MAXREQ=256 died during CUDA-graph capture with a FlashInfer workspace overflow (the K=33 draft window
# needs more prefill scratch than causal decoding). Lowering only the capture batch keeps the running
# concurrency (MAXREQ).
if [ -n "$GRAPHBS" ]; then ARGS+=(--cuda-graph-max-bs "$GRAPHBS"); fi
# DP>1: SGLang data parallelism, one replica per listed GPU behind one port.
if [ "$DP" -gt 1 ]; then ARGS+=(--dp-size "$DP"); fi

if [ "$MODE" = self-spec ]; then
  # Block-diffusion self-speculative decoding: the diffusion head drafts K = bd_size + 1 = 33 tokens per
  # round and the AR path verifies them; needs the SELFSPEC_DIFFUSION choice from the patch and page
  # size 1. Under bf16 kernels a near-tie can resolve differently from causal decoding. With
  # repetition_penalty != 1, export SELFSPEC_EXACT_PENALTY=1.
  ARGS+=(--speculative-algorithm SELFSPEC_DIFFUSION --speculative-num-draft-tokens "$DRAFT" --page-size 1)
  # SELFSPEC_POS_OFFSET=1 is the position layout GravityOCR's published results were produced with
  # (0 is the AR-aligned layout). A checkpoint dir with block_diffusion.json supplies the mask id
  # itself; otherwise GLM-OCR's <|mask|> (59282) is used.
  export SELFSPEC_FULLDRAFT="${SELFSPEC_FULLDRAFT:-1}" SELFSPEC_POS_OFFSET="${SELFSPEC_POS_OFFSET:-1}"
  if [ ! -f "$CKPT/block_diffusion.json" ]; then export SELFSPEC_MASK_ID="${SELFSPEC_MASK_ID:-59282}"; fi
else
  # causal: a plain autoregressive server; drop any inherited self-spec switches.
  while IFS= read -r var; do unset "$var"; done < <(compgen -e SELFSPEC_ || true)
fi

# FAST=1: runtime optimizations from the patch; they are meant to change speed only, not outputs
# (compare against FAST=0 if in doubt). Any single switch can still be overridden from the environment.
#   both modes: GLM ViT as CUDA graphs per patch count, bf16 pixel transport, single-threaded CPU ops
#               in the server, two tokenizer worker processes (one when API_KEY is set; see above)
#   self-spec:  attention without custom masks, M-RoPE delta cache fix, GPU-side commit, host-side
#               lengths, positions on the device, next round's draft launched ahead of the bookkeeping
if [ "$FAST" = 1 ]; then
  export SS_GLM_VIT_GRAPH_MAXM="${SS_GLM_VIT_GRAPH_MAXM:-2048}" SS_GLM_PV_BF16="${SS_GLM_PV_BF16:-1}"
  export OMP_NUM_THREADS="$SERVER_THREADS" MKL_NUM_THREADS="$SERVER_THREADS" TORCH_NUM_THREADS="$SERVER_THREADS"
  if [ "$MODE" = self-spec ]; then
    export SELFSPEC_FASTATTN="${SELFSPEC_FASTATTN:-1}" SELFSPEC_MROPE_FIX="${SELFSPEC_MROPE_FIX:-1}" \
           SELFSPEC_FAST_COMMIT="${SELFSPEC_FAST_COMMIT:-1}" SELFSPEC_HOSTLEN="${SELFSPEC_HOSTLEN:-1}" \
           SELFSPEC_DEV_POS="${SELFSPEC_DEV_POS:-1}" SELFSPEC_DRAFT_AHEAD="${SELFSPEC_DRAFT_AHEAD:-1}"
  fi
fi
ARGS+=(${EXTRA[@]+"${EXTRA[@]}"})
# SGLang merges a YAML file wherever a literal '--config' token appears in its argv. The checks above
# keep it out; refuse anyway if one got through, before the launcher adds its own.
for a in "${ARGS[@]}"; do
  [ "$a" != --config ] || die 2 "'--config' reached the server arguments through a launcher value; refusing"
done

# ---- keys --------------------------------------------------------------------------------------
# With one tokenizer worker SGLang enforces keys. The launcher then always sets a random admin key
# that it never prints, so SGLang's management endpoints (weight reloads from a caller-chosen path,
# profiling, cache control, LoRA loading, ...) refuse every caller; API_KEY, when set, guards the
# other endpoints. Both reach SGLang in a mode-0600 YAML file read through --config, written with shell
# builtins only, so neither appears on a command line. A detached watcher deletes the file's private
# temp dir when the server process exits. SGLang logs its full server_args, keys included, at INFO on
# startup: keep the server log private, or add EXTRA_ARGS="--log-level warning".
if [ "$WORKERS" = 1 ]; then
  admin=$(od -vAn -N32 -tx1 /dev/urandom) || die 1 "could not read /dev/urandom for the admin key"
  admin="${admin//[[:space:]]/}"
  [[ $admin =~ ^[0-9a-f]{64}$ ]] || die 1 "could not generate the admin key"
  keydir=$(umask 077 && mktemp -d "${TMPDIR:-/tmp}/gravityocr-serve.XXXXXX")
  ( umask 077
    { printf "admin-api-key: '%s'\n" "$admin"
      if [ -n "$API_KEY" ]; then printf "api-key: '%s'\n" "${API_KEY//\'/\'\'}"; fi
    } > "$keydir/server.yaml" )
  unset admin API_KEY
  ARGS+=(--config "$keydir/server.yaml")
  ( trap '' HUP INT TERM; while kill -0 "$$" 2>/dev/null; do sleep 2; done; rm -rf -- "$keydir" ) \
    </dev/null >/dev/null 2>&1 &
  if ! is_loopback "$SERVE_HOST"; then
    echo "[serve] note: listening on $SERVE_HOST; every request must send 'Authorization: Bearer <API_KEY>'." \
         "Authenticated clients can still make the server fetch http(s) image URLs and read local image" \
         "files, so put a proxy that only forwards data: image URLs in front if clients are not trusted." >&2
  fi
else
  echo "[serve] note: $WORKERS tokenizer workers: SGLang enforces no keys in this mode, so its management" \
       "endpoints are open to every user on this host. On a shared host set API_KEY or TOKENIZER_WORKERS=1." >&2
fi

echo "[serve] MODE=$MODE FAST=$FAST GPU=$GPU DP=$DP SERVE_HOST=$SERVE_HOST PORT=$PORT SERVED_NAME=$SERVED_NAME CKPT=$CKPT"
export CUDA_VISIBLE_DEVICES="$GPU"
exec "$VBIN/python" "${ARGS[@]}"
