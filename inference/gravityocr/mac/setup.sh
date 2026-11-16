#!/usr/bin/env bash
# Copyright 2026 Trillion Labs
# SPDX-License-Identifier: Apache-2.0
#
# Build the Python env that serve_mac.py runs on an Apple Silicon Mac: the GravityOCR MLX engine in
# inference/gravityocr/mac/ocr plus the packages it needs. No Gravity app is needed.
#
#     bash inference/gravityocr/mac/setup.sh                          # venv at inference/gravityocr/mac/.venv
#     MAC_VENV=$HOME/.cache/gravityocr-mac/venv bash inference/gravityocr/mac/setup.sh
#
# Steps. Re-running is safe: an existing venv is re-synced to the lock.
#   1. Preflight: macOS on arm64 (Apple Silicon), uv on PATH.
#   2. Create $MAC_VENV with Python 3.12 (the Gravity app's runtime is 3.12).
#   3. Install requirements-lock.txt as is: --no-deps, wheels only, hashes checked, from PyPI only.
#      These are the package versions bundled in Gravity.app 0.3.4. mlx-vlm comes without its
#      dependencies, as in the app: the engine imports only its glm_ocr model (no transformers or torch).
#   4. Check: the engine imports, and MLX sees a Metal GPU.
#
# Then serve with:  $MAC_VENV/bin/python -I inference/gravityocr/serve_mac.py
#
# Env, or KEY=VALUE arguments:
#   MAC_VENV   venv directory (default inference/gravityocr/mac/.venv)
#   PY         Python version for the venv (default 3.12)
set -euo pipefail

for kv in "$@"; do
  case "$kv" in
    -h|--help) sed -n '/^# Build /,/^set -euo pipefail$/{/^set /d;s/^# \{0,1\}//;p;}' "$0"; exit 0 ;;
    MAC_VENV=*|PY=*) export "${kv?}" ;;
    *) echo "mac/setup.sh: unknown argument: $kv" >&2; exit 2 ;;
  esac
done

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
die() { echo "mac/setup.sh: $*" >&2; exit 1; }
log() { echo "[mac/setup] $*"; }
MAC_VENV=${MAC_VENV:-$HERE/.venv}
PY=${PY:-3.12}

# ── 1. preflight ─────────────────────────────────────────────────────────────────────────────────
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || die "needs an Apple Silicon Mac (MLX runs on the Mac GPU)"
command -v uv > /dev/null || die "uv not found on PATH (https://docs.astral.sh/uv/getting-started/installation/)"

# ── 2. venv ──────────────────────────────────────────────────────────────────────────────────────
[ -x "$MAC_VENV/bin/python" ] || { log "creating $MAC_VENV (Python $PY)"; uv venv -q --python "$PY" "$MAC_VENV"; }

# ── 3. packages, from the lock only ──────────────────────────────────────────────────────────────
log "installing requirements-lock.txt (the package versions of Gravity.app 0.3.4)"
env -u UV_INDEX -u UV_INDEX_URL -u UV_EXTRA_INDEX_URL -u UV_FIND_LINKS -u UV_DEFAULT_INDEX \
  uv pip sync -q --no-config --python "$MAC_VENV/bin/python" --index-url https://pypi.org/simple \
  --only-binary :all: --require-hashes "$HERE/requirements-lock.txt"

# ── 4. check ─────────────────────────────────────────────────────────────────────────────────────
"$MAC_VENV/bin/python" -I - "$HERE" <<'PY' || die "the engine does not import (see above)"
import sys
sys.dont_write_bytecode = True
sys.path.insert(0, sys.argv[1])
import mlx.core as mx
from ocr.engine import pipeline  # noqa: F401
from ocr.recognition.model import decode_ar_mlx_from_prefix  # noqa: F401
if not mx.metal.is_available():
    sys.exit("MLX sees no Metal GPU")
print(f"[mac/setup] engine imports; MLX {mx.__version__} on {mx.default_device()}")
PY

log "done. Serve with:"
log "  $MAC_VENV/bin/python -I $(cd "$HERE/.." && pwd)/serve_mac.py"
