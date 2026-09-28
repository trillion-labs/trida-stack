#!/bin/bash
# Serve trillionlabs/Trida2.0-4B (or any two-stream block-diffusion checkpoint) on vLLM 0.27.x
# via the out-of-tree plugin (arch: Qwen3_5ForBlockDiffusion). No fork, no SGLang, no shaping.
#
# Requires a plain `vllm` on PATH and the plugin installed:
#     pip install vllm==0.27.*
#     pip install -e inference/vllm
#
# Everything is env-driven. The checkpoint is CKPT, falling back to TRIDA_MODEL, then to the
# default trillionlabs/Trida2.0-4B (private on HF during preview: your account needs access,
# then `huggingface-cli login` or HF_TOKEN).
#     bash inference/vllm/serve_diffusion.sh                        # self-spec (default), one stream
#     MAX_NUM_SEQS=8 bash inference/vllm/serve_diffusion.sh         # self-spec, batched (PIECEWISE graphs)
#     MODE=causal bash inference/vllm/serve_diffusion.sh            # plain autoregressive, stock vLLM
#     MODE=diffusion bash inference/vllm/serve_diffusion.sh         # pure block-diffusion (other families)
set -euo pipefail

# ---- config (env-driven) ---------------------------------------------------
CKPT="${CKPT:-${TRIDA_MODEL:-trillionlabs/Trida2.0-4B}}"   # HF repo id or local path
MODE="${MODE:-self-spec}"             # self-spec | causal | diffusion
PORT="${PORT:-8000}"
SERVED_NAME="${SERVED_NAME:-trida}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"  # prompt + max_tokens must fit (eval.py defaults to 8192 output tokens)
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"     # concurrent sequences per replica (1 = latency mode)
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.8}"
MASK_ID="${MASK_ID:-248077}"          # mask token id of the checkpoint
LOG="${LOG:-}"                        # optional: also write server output here (eval.py --server-log)
# self-spec: N = draft block (gen) size; the canvas is 2N-1 slots (= the model's block of 8 for N=4;
# the config value is 2N-2 because vLLM adds the bonus token itself)
SELFSPEC_N="${SELFSPEC_N:-4}"
[ "$SELFSPEC_N" -ge 2 ] 2>/dev/null || { echo "SELFSPEC_N must be an integer >= 2 (got '$SELFSPEC_N'); N=4 is the model's block of 8" >&2; exit 2; }
SS_THRESHOLD="${SS_THRESHOLD:-0.90}"  # passed through as confidence_threshold; the self-spec path does not gate on it
# diffusion (pure denoising; not supported by the Trida2.0-4B full-mask recipe)
CANVAS="${CANVAS:-3}"                 # block canvas length (hf canvas_length): 3 = block B=4
THRESHOLD="${THRESHOLD:-0.95}"        # per-step commit confidence gate
MAX_STEPS="${MAX_STEPS:-8}"           # max denoising steps per block

# The plugin's per-step counters ("DiffusionDecoding metrics: ... Committed: N tokens") are logged at
# INFO; eval.py --server-log parses them for tokens-per-forward, so don't let a quieter ambient
# setting silently drop them.
export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-INFO}"

# ---- cuda graphs -----------------------------------------------------------
# self-spec: FULL_AND_PIECEWISE at every concurrency (lossless at C>1 since the int32 causal-buffer fix;
# byte-identical to PIECEWISE under greedy at C=1/2/4/8). diffusion: PIECEWISE. Override with CUDAGRAPH=.
if [ -z "${CUDAGRAPH:-}" ]; then
  if [ "$MODE" = "self-spec" ]; then CUDAGRAPH=FULL_AND_PIECEWISE; else CUDAGRAPH=PIECEWISE; fi
fi

[ -n "$LOG" ] && exec > >(tee -a "$LOG") 2>&1
COMMON=(--port "$PORT" --served-model-name "$SERVED_NAME" --tensor-parallel-size 1
        --max-model-len "$MAX_MODEL_LEN" --max-num-seqs "$MAX_NUM_SEQS"
        --gpu-memory-utilization "$GPU_MEM_UTIL" --trust-remote-code)

case "$MODE" in
  causal)
    # stock vLLM, no plugin: the checkpoint is a plain Qwen3.5 causal LM
    export VLLM_PLUGINS=""
    exec vllm serve "$CKPT" "${COMMON[@]}" ;;
  self-spec)
    export VLLM_PLUGINS="${VLLM_PLUGINS:-trida_diffusion}"   # register_trida() runs in EVERY vLLM process
    export TRIDA_SELFSPEC_N="$SELFSPEC_N"
    # TRIDA_SS_SAMPLE=1 (default): speculative rejection sampling with the request's temperature/top_k/top_p
    # (matches SGLang's self-spec). TRIDA_SS_SAMPLE=0: exact-greedy verify, ignores sampling fields.
    export TRIDA_SS_SAMPLE="${TRIDA_SS_SAMPLE:-1}"
    CL=$((2 * SELFSPEC_N - 2))
    exec vllm serve "$CKPT" "${COMMON[@]}" \
      --hf-overrides "{\"architectures\":[\"Qwen3_5ForBlockDiffusion\"],\"canvas_length\":$CL,\"mask_id\":$MASK_ID,\"confidence_threshold\":$SS_THRESHOLD}" \
      --diffusion-config "{\"canvas_length\":$CL,\"max_denoising_steps\":8}" \
      --compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH\"}" ;;
  diffusion)
    export VLLM_PLUGINS="${VLLM_PLUGINS:-trida_diffusion}"
    exec vllm serve "$CKPT" "${COMMON[@]}" \
      --hf-overrides "{\"architectures\":[\"Qwen3_5ForBlockDiffusion\"],\"canvas_length\":$CANVAS,\"mask_id\":$MASK_ID,\"confidence_threshold\":$THRESHOLD}" \
      --diffusion-config "{\"canvas_length\":$CANVAS,\"max_denoising_steps\":$MAX_STEPS}" \
      --compilation-config "{\"cudagraph_mode\":\"$CUDAGRAPH\"}" ;;
  *) echo "MODE must be self-spec | causal | diffusion (got '$MODE')" >&2; exit 2 ;;
esac
