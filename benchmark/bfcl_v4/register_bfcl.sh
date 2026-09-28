#!/bin/bash
# Install the BFCL v4 package and register Trida-7B into it (idempotent).
#
# BFCL has no plugin system: --model is resolved against MODEL_CONFIG_MAPPING built at import in
# bfcl_eval/constants/model_config.py, and the handler class must be importable inside bfcl_eval. So
# we (1) editable-install BFCL, (2) symlink our handler into the package, (3) append a ModelConfig
# registration to model_config.py. Our source of truth stays here under benchmark/bfcl_v4/.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BFCL_DIR="${BFCL_DIR:-$SCRATCH/gorilla/berkeley-function-call-leaderboard}"
UV="${UV:-uv}"
MODEL_ID="trillionlabs/Trida-7B-Preview"
# Three isolated venvs (dep worlds don't mix):
#   .venv        trainer (transformers 5.x)          — untouched here
#   .venv-serve  Trida inference server (tf 4.57.1)   — the released remote code needs 4.x, not 5.x
#   .venv-bfcl   the bfcl CLI (BFCL's own deps)
SERVE_VENV="${SERVE_VENV:-$REPO/.venv-serve}"; SERVE_PY="$SERVE_VENV/bin/python"
BFCL_VENV="${BFCL_VENV:-$REPO/.venv-bfcl}";     BFCL_PY="$BFCL_VENV/bin/python"

[ -d "$BFCL_DIR" ] || { echo "BFCL dir not found: $BFCL_DIR (set BFCL_DIR)"; exit 1; }

# --- server venv: transformers 4.57.1 (matches the released model) + torch 2.11 cu128 ---
if [ ! -x "$SERVE_PY" ]; then
    echo "[register] creating server venv at $SERVE_VENV"
    "$UV" venv --python 3.12 "$SERVE_VENV" >/dev/null
    "$UV" pip install --python "$SERVE_PY" --index-url https://download.pytorch.org/whl/cu128 "torch==2.11.0" >/dev/null
fi
echo "[register] installing server deps into $SERVE_VENV (transformers 4.57.1 + einops + fastapi/uvicorn/httpx)"
"$UV" pip install --python "$SERVE_PY" "transformers==4.57.1" accelerate safetensors sentencepiece einops fastapi uvicorn httpx >/dev/null

# --- BFCL CLI venv ---
if [ ! -x "$BFCL_PY" ]; then
    echo "[register] creating BFCL venv at $BFCL_VENV"
    "$UV" venv --python 3.12 "$BFCL_VENV" >/dev/null
fi
echo "[register] editable-installing BFCL (base deps) into $BFCL_VENV"
"$UV" pip install --python "$BFCL_PY" -e "$BFCL_DIR" >/dev/null
# model_config.py eagerly imports every handler at import; the Qwen API handler pulls qwen_agent
# -> soundfile. Install it so the registry (and our Trida entry) import cleanly.
"$UV" pip install --python "$BFCL_PY" soundfile >/dev/null

# 1) symlink the handler into the BFCL package
LOCAL_DIR="$BFCL_DIR/bfcl_eval/model_handler/local_inference"
ln -sf "$HERE/trida_handler.py" "$LOCAL_DIR/trida.py"
echo "[register] linked handler -> $LOCAL_DIR/trida.py"

# 2) append the registration to model_config.py (idempotent via marker)
CFG="$BFCL_DIR/bfcl_eval/constants/model_config.py"
MARKER="# --- Trida BFCL registration (added by trida-stack/benchmark/bfcl_v4) ---"
if grep -qF "$MARKER" "$CFG"; then
    echo "[register] model_config.py already has the Trida entry; skipping"
else
    cat >> "$CFG" <<PYEOF

$MARKER
from bfcl_eval.model_handler.local_inference.trida import TridaFCHandler as _TridaFCHandler

_trida_cfg = ModelConfig(
    model_name="$MODEL_ID",
    display_name="Trida-7B (FC)",
    url="https://huggingface.co/$MODEL_ID",
    org="Trillion Labs",
    license="see model card",
    model_handler=_TridaFCHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
)
local_inference_model_map["$MODEL_ID"] = _trida_cfg
MODEL_CONFIG_MAPPING["$MODEL_ID"] = _trida_cfg
PYEOF
    echo "[register] appended Trida ModelConfig to $CFG"
fi

echo "[register] verifying (in BFCL venv)..."
"$BFCL_PY" - <<PYEOF
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING
cfg = MODEL_CONFIG_MAPPING["$MODEL_ID"]
print("  OK:", cfg.display_name, "| handler:", cfg.model_handler.__name__, "| is_fc:", cfg.is_fc_model)
PYEOF
echo "[register] done. BFCL CLI: $BFCL_VENV/bin/bfcl"
