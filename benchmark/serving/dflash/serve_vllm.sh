#!/bin/bash
# One vLLM server for the DFlash comparison. env: MODE=ar|dflash|selfspec  NAME  PORT  GPU  BLOCK (dflash draft
# tokens or self-spec N)  MAXSEQS(16). Box layout: $SCRATCH/{env,models,scripts}.
set -u
L=$SCRATCH; ROOT=$L/env/vllm-uv27
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
export LD_LIBRARY_PATH=$COMPAT:$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH} PATH=$ROOT/.venv/bin:$PATH
export CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1
TARGET=$L/models/dflash/Qwen3.5-4B; DRAFT=$L/models/dflash/Qwen3.5-4B-DFlash
COMMON="--port $PORT --served-model-name $NAME --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs ${MAXSEQS:-16} --gpu-memory-utilization 0.8 --trust-remote-code"
case $MODE in
  ar)      exec vllm serve $TARGET $COMMON ;;
  dflash)  exec vllm serve $TARGET $COMMON --speculative-config "{\"method\":\"dflash\",\"model\":\"$DRAFT\",\"num_speculative_tokens\":$BLOCK}" ;;
  selfspec) export PYTHONPATH=$L/code/vllm-native-dev9 VLLM_PLUGINS=trida_diffusion
           exec env PORT=$PORT CL=$((2*BLOCK-2)) THRESH=0.90 MAXSTEPS=8 TRIDA_SELFSPEC_N=$BLOCK COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json \
               EXTRA_ARGS="--served-model-name $NAME --max-num-seqs ${MAXSEQS:-16} --gpu-memory-utilization 0.8" bash $L/scripts/serve_diff_cl.sh ;;
  *) echo "unknown MODE $MODE"; exit 2 ;;
esac
