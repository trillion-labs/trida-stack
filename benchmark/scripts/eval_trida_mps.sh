#!/bin/bash
# NOTE: not runnable from a clean checkout. This script drives lm-eval through eval_dinfer.py,
# part of the vendored dInfer harness that is not published in this repository. Kept as a record of
# how the single-turn numbers were produced; serve from inference/ and use lm-evaluation-harness
# directly for a runnable path.
# Run Trida eval on a single device: MPS (Mac) or CPU. No vLLM, no multi-GPU.
# Updated for the trida-stack layout: first-party `trida/` + vendored `dinfer` under vendor/.
# Usage (from anywhere): bash benchmark/scripts/eval_trida_mps.sh
# Default model: mock (tiny in-memory Trida, no download). Override with TRIDA_MODEL_PATH,
# e.g. TRIDA_MODEL_PATH=trillionlabs/Trida-7B-Preview bash benchmark/scripts/eval_trida_mps.sh

set -e
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=1
export TRANSFORMERS_TRUST_REMOTE_CODE=1
export NUMEXPR_MAX_THREADS=8   # No CUDA_VISIBLE_DEVICES; PyTorch uses MPS or CPU

# Repo root = two levels up from this script (benchmark/scripts/ -> repo root).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# First-party `trida` (repo root) + vendored `dinfer` (vendor/) must both be importable.
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/vendor:${PYTHONPATH:-}"

EVAL_PY="${REPO_ROOT}/vendor/dinfer/evaluations/eval_dinfer.py"
TASKS_DIR="${REPO_ROOT}/benchmark/tasks"

# Use TRIDA_MODEL_PATH=mock for a tiny in-memory Trida (no download, for a smoke test).
model_path="${TRIDA_MODEL_PATH:-mock}"

model_type='trida_dinfer'
task='gsm8k_trida'
length=256
block_length=32
threshold=0.9
temperature=0
top_p=0.95
cache='prefix'
use_compile=False
limit='1'
batch_size=1
output_dir="${REPO_ROOT}/outputs/trida_mps"
save_samples=True
parallel='single'   # single-device path: picks MPS if available, else CPU

output_path="${output_dir}/${task}"
mkdir -p "$output_path"

python "$EVAL_PY" --tasks "$task" \
  --confirm_run_unsafe_code --model dInfer_eval \
  --model_args model_path=${model_path},add_bos_token=True,gen_length=${length},block_length=${block_length},threshold=${threshold},temperature=${temperature},top_p=${top_p},show_speed=True,save_dir=${output_path},cache=${cache},use_compile=${use_compile},parallel=${parallel},model_type=${model_type},save_samples=${save_samples} \
  --output_path "${output_path}" --include_path "$TASKS_DIR" --apply_chat_template \
  --batch_size ${batch_size} \
  --limit ${limit}

echo "Done. Results in ${output_path}"
