#!/bin/bash
# NOTE: not runnable from a clean checkout. This script drives lm-eval through eval_dinfer.py,
# part of the vendored dInfer harness that is not published in this repository. Kept as a record of
# how the single-turn numbers were produced; serve from inference/ and use lm-evaluation-harness
# directly for a runnable path.
source $SHARED/miniconda3/etc/profile.d/conda.sh
conda activate dinfer

# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
# Save Hugging Face downloads under dInfer (override if you want a different path)
export HUGGINGFACE_HUB_CACHE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.hf_cache"
export NUMEXPR_MAX_THREADS=64
# Debug logging: Set TRIDA_DEBUG=1 to enable debug prints and file logging (default: disabled for performance)
# export TRIDA_DEBUG=1  # Uncomment to enable debug logging
# Set PYTHONPATH for accelerate processes (includes dInfer/python for Trida model classes).
# NOTE: the Trida model classes come from that EXTERNAL dInfer checkout, not from this repo
# (the in-repo Trida-7B model definition was removed) — so this script is unaffected.
export PYTHONPATH=$SHARED/lm-evaluation-harness:$SHARED/dInfer/python:$SHARED/miniconda3/envs/dinfer/lib/python3.12/site-packages:$PYTHONPATH

# Model configuration
# - 'trida'        : direct TridaForDLM.generate() path (uses Trida's internal cache, no dInfer cache)
# - 'trida_dinfer' : TridaModelLM wrapped for dInfer BlockDiffusionLLM
#                    * dInfer drives block schedule + external KV cache
#                    * TridaModelLM preserves Trida's eval_mask + RoPE semantics internally
model_type='trida_dinfer'
model_path='trillionlabs/Trida-7B-Preview'  # Hugging Face; or set to local path e.g. /path/to/hf_eso/0000010000

# Generation configuration
length=1024           # generate length
block_length=32        # block length (Trida was trained with block_size=4)

# For 'trida_dinfer', these directly configure dInfer's BlockDiffusionLLM + TridaThresholdDecoder.
# For 'trida', they are forwarded into TridaForDLM.generate().
threshold=0.9        # threshold for unmasking (1.0 = greedy)
temperature=0
top_p=0.95

# Cache configuration
# For 'trida_dinfer':
#   - cache='prefix' : use dInfer's prefix KV cache (standard external cache, recommended)
#   - cache='dual'   : use dInfer cache with Trida-style decoding over all blocks
# For 'trida':
#   - cache is mapped inside eval_dinfer.py to TridaForDLM.generate(cache_mode=...)
cache='prefix'
use_compile=True      # enable torch.compile (attention mask and SDPA are excluded to preserve Trida semantics)
# Note: use_incremental_cache is not a parameter in eval_dinfer.py
# Trida's generate() method supports use_incremental_cache, but it's not exposed through eval_dinfer.py yet

# Parallelism configuration
tp_size=4             # tensor parallel size (set >1 for multi-GPU TP)
gpus='0;1;2;3;'        # gpus to use (semicolon-separated for DP mode)
parallel='tp'         # 'tp' for tensor parallel or 'dp' for data parallel
master_port="23457"   # master port
num_processes=1        # number of processes for DP mode (each GPU gets 1 sample)

# Output configuration
output_dir='./outputs/trida_dinfer_tp'
save_samples=True

# Task to run (use existing task - gsm8k_trida doesn't exist, use gsm8k_llada1.5 or gsm8k)
task='gsm8k_trida'  # or 'gsm8k' for standard gsm8k task

# Limit number of examples for debugging (set to empty string '' for full set)
limit='10'  # Set to desired number (e.g., 10 for quick testing, '' for full set)

# Batch size for inference
batch_size=4

if [ "$parallel" == 'tp' ]; then
    gpus='0;1;2;3'
    tp_size=4
    output_path=${output_dir}/${task}
    python eval_dinfer.py --tasks ${task} \
        --confirm_run_unsafe_code --model dInfer_eval \
        --model_args model_path=${model_path},add_bos_token=True,gen_length=${length},block_length=${block_length},threshold=${threshold},temperature=${temperature},top_p=${top_p},show_speed=True,save_dir=${output_path},cache=${cache},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},gpus=${gpus},model_type=${model_type},master_port=${master_port},save_samples=${save_samples} \
        --output_path ${output_path} --include_path ./tasks --apply_chat_template \
        --batch_size ${batch_size} \
        ${limit:+--limit ${limit}}
elif [ "$parallel" == 'dp' ]; then
    # Use accelerate to enable multi-gpu data parallel inference
    output_path=${output_dir}/${task}
    tp_size=1  # Set tp_size for DP mode (not used, but prevents undefined variable error)
    $SHARED/miniconda3/envs/dinfer/bin/python3 -m accelerate.commands.launch --num_processes=${num_processes} --main_process_port=$((master_port + 1000)) eval_dinfer.py --tasks ${task} \
        --confirm_run_unsafe_code --model dInfer_eval \
        --model_args model_path=${model_path},add_bos_token=True,gen_length=${length},block_length=${block_length},threshold=${threshold},temperature=${temperature},top_p=${top_p},show_speed=True,save_dir=${output_path},cache=${cache},use_compile=${use_compile},tp_size=${tp_size},parallel=${parallel},gpus=${gpus},model_type=${model_type},master_port=${master_port},save_samples=${save_samples} \
        --output_path ${output_path} --include_path ./tasks --apply_chat_template \
        --batch_size ${batch_size} \
        ${limit:+--limit ${limit}}
else
    echo "parallel must be tp or dp"
fi
