#!/bin/bash
# One SGLang server (recent upstream build with DFlash) for the comparison. env: MODE=ar|dflash NAME PORT GPU BLOCK MAXSEQS(16)
# Blackwell-only flags from the model card (trtllm_mha, fa4) are omitted for H100.
set -u
L=$SCRATCH; VENV=$L/env/sglang-dflash
# git-main SGLang ships a torch built for CUDA 13; the box driver (570) needs the cuda-compat-13 libs first on the path
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
export LD_LIBRARY_PATH=$COMPAT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
export PATH=$VENV/bin:$PATH CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 SGLANG_ENABLE_OVERLAP_PLAN_STREAM=1 SGLANG_DISABLE_CUDNN_CHECK=1 FLASHINFER_DISABLE_VERSION_CHECK=1   # cuDNN check trips on this box; git-main flashinfer 0.6.18 has no matching cubin wheel on PyPI (falls back to JIT)
TARGET=$L/models/dflash/Qwen3.5-4B; DRAFT=$L/models/dflash/Qwen3.5-4B-DFlash
COMMON="--model-path $TARGET --served-model-name $NAME --trust-remote-code --tp-size 1 --host 0.0.0.0 --port $PORT --max-running-requests ${MAXSEQS:-16} --cuda-graph-max-bs-decode ${MAXSEQS:-16} --mem-fraction-static 0.8 --context-length 8192"
case $MODE in
  ar)     exec python -m sglang.launch_server $COMMON ;;
  # the model card also passes --mamba-scheduler-strategy extra_buffer; that flag does not exist in this build
  dflash) exec python -m sglang.launch_server $COMMON --speculative-algorithm DFLASH --speculative-draft-model-path $DRAFT --speculative-dflash-block-size $BLOCK \
             --linear-attn-prefill-backend flashinfer --linear-attn-decode-backend flashinfer ${SGL_EXTRA:-} ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
