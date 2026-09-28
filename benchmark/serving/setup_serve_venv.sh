#!/bin/bash
# Build the shared Trida serving venv (.venv-serve) if missing. Idempotent.
#
# Pinned to transformers 4.57.1 + torch 2.11 (cu128): the released trillionlabs/Trida-7B-Preview
# remote code targets transformers 4.x and breaks on 5.x. Sourced/called by every register_*.sh so
# all benchmarks share one model-serving environment.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UV="${UV:-uv}"
SERVE_VENV="${SERVE_VENV:-$REPO/.venv-serve}"; SERVE_PY="$SERVE_VENV/bin/python"

if [ ! -x "$SERVE_PY" ]; then
    echo "[serve-venv] creating $SERVE_VENV"
    "$UV" venv --python 3.12 "$SERVE_VENV" >/dev/null
    "$UV" pip install --python "$SERVE_PY" --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" >/dev/null
fi
echo "[serve-venv] ensuring server deps (transformers 4.57.1 + fastapi/uvicorn/httpx)"
"$UV" pip install --python "$SERVE_PY" \
    "transformers==4.57.1" accelerate safetensors sentencepiece einops fastapi uvicorn httpx >/dev/null
echo "[serve-venv] ready: $SERVE_PY"
