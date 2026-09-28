#!/bin/bash
# NOTE: not runnable from a clean checkout. This script drives lm-eval through eval_dinfer.py,
# part of the vendored dInfer harness that is not published in this repository. Kept as a record of
# how the single-turn numbers were produced; serve from inference/ and use lm-evaluation-harness
# directly for a runnable path.
# Run Trida eval with vLLM (CPU build) on Mac: single process, gloo backend.
# Requires vLLM installed for CPU (e.g. from vllm_source with uv as in README).
# Usage: from dInfer/evaluations/, run: ./eval_trida_vllm_cpu.sh
# Default model: trillionlabs/Trida-7B-Preview. Override with TRIDA_MODEL_PATH.

set -e
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export NUMEXPR_MAX_THREADS=8

# PYTHONPATH: dInfer package + optional venv with vLLM
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DINFER_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$DINFER_DIR/.." && pwd)"
export PYTHONPATH="${DINFER_DIR}/python:${PYTHONPATH:-}"

# Activate project venv if present (has vLLM CPU build)
if [ -f "${PROJECT_ROOT}/.venv/bin/activate" ]; then
  source "${PROJECT_ROOT}/.venv/bin/activate"
fi

# Save Hugging Face downloads under dInfer (instead of ~/.cache/huggingface/hub)
export HUGGINGFACE_HUB_CACHE="${DINFER_DIR}/.hf_cache"

# Model path: Hugging Face id or local path (override with TRIDA_MODEL_PATH)
model_path="${TRIDA_MODEL_PATH:-trillionlabs/Trida-7B-Preview}"

model_type='trida_dinfer'
task='gsm8k_trida'
length=256
block_length=32
threshold=0.9
temperature=0
top_p=0.95
cache='prefix'
use_compile=False
use_cudagraph=False
limit='1'
batch_size=1
output_dir='./outputs/trida_vllm_cpu'
save_samples=True
parallel='vllm_cpu'

output_path="${output_dir}/${task}"
mkdir -p "$output_path"

cd "$SCRIPT_DIR"
python eval_dinfer.py --tasks "$task" \
  --confirm_run_unsafe_code --model dInfer_eval \
  --model_args model_path=${model_path},add_bos_token=True,gen_length=${length},block_length=${block_length},threshold=${threshold},temperature=${temperature},top_p=${top_p},show_speed=True,save_dir=${output_path},cache=${cache},use_compile=${use_compile},use_cudagraph=${use_cudagraph},parallel=${parallel},model_type=${model_type},save_samples=${save_samples} \
  --output_path "${output_path}" --include_path ./tasks --apply_chat_template \
  --batch_size ${batch_size} \
  --limit ${limit}

echo "Done. Results in ${output_path}"
