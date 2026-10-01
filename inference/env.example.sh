# nano-inference environment (example) — source before serving in a clean env.
#   source env.example.sh
#
# serve.py deliberately stays thin and does NOT set backend plumbing, so a fresh
# machine needs a few env vars for the diffusion-serving SGLang backend. Adjust
# CACHE_ROOT to a fast local/scratch disk with room for weights + JIT artifacts.

# Checkpoint served when serve.py / serve_diffusion.sh get no explicit model.
# The reference model is public on Hugging Face -- no authentication needed. For a
# private checkpoint of your own, authenticate once with `huggingface-cli login`,
# or export HF_TOKEN=hf_... here.
export TRIDA_MODEL="${TRIDA_MODEL:-trillionlabs/Trida2.0-4B}"
# export HF_TOKEN=hf_...

# Interpreter that has the diffusion-serving SGLang backend installed.
# (defaults to the current `python` if unset)
export SGLANG_PYTHON="${SGLANG_PYTHON:-$(command -v python)}"

# The backend venv's bin must be on PATH — the JIT kernel build needs `ninja`
# (and other build tools) at server startup.
export PATH="$(dirname "$SGLANG_PYTHON"):$PATH"

CACHE_ROOT="${CACHE_ROOT:-$HOME/.cache/nano-inference}"
export HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"                 # model downloads
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$CACHE_ROOT/flashinfer}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-$CACHE_ROOT/torch_extensions}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$CACHE_ROOT/triton}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
mkdir -p "$HF_HOME" "$FLASHINFER_WORKSPACE_BASE" "$TORCH_EXTENSIONS_DIR" "$TRITON_CACHE_DIR"

# If the backend build needs CUDA stubs (libcuda.so), point CUDA_HOME at your
# CUDA install; see the backend's install docs.
# export CUDA_HOME=/usr/local/cuda
