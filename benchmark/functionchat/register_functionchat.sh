#!/bin/bash
# Set up FunctionChat-Bench for Trida eval (idempotent). Unlike BFCL, FunctionChat needs no package
# patching — it takes the target model via CLI (--model inhouse --base_url ...). So this just builds
# the shared serving venv and FunctionChat's own deps venv.
#
# The LLM-as-judge is FunctionChat's own config/openai.cfg (OpenRouter gpt-4.1) — left as-is; make
# sure its api_key is valid before running (scoring calls it).
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UV="${UV:-uv}"
FCBENCH_DIR="${FCBENCH_DIR:-$SCRATCH/FunctionChat-Bench}"

[ -d "$FCBENCH_DIR" ] || { echo "FunctionChat-Bench not found: $FCBENCH_DIR (set FCBENCH_DIR)"; exit 1; }

# 1) shared Trida serving venv (.venv-serve)
bash "$REPO/benchmark/serving/setup_serve_venv.sh"

# 2) FunctionChat deps venv (its own dir; imports openai/mistralai/vertexai/qwen_agent at module top)
FC_VENV="$FCBENCH_DIR/.venv"; FC_PY="$FC_VENV/bin/python"
if [ ! -x "$FC_PY" ]; then
    echo "[register] creating FunctionChat venv at $FC_VENV"
    "$UV" venv --python 3.12 "$FC_VENV" >/dev/null
fi
echo "[register] installing FunctionChat requirements into $FC_VENV"
"$UV" pip install --python "$FC_PY" -r "$FCBENCH_DIR/requirements.txt" >/dev/null

echo "[register] verifying imports..."
"$FC_PY" - <<'PY'
import openai, click, pydantic  # core; heavy provider libs imported lazily by our run path
print("  OK: openai", openai.__version__)
PY

# 3) judge config reminder
CFG="$FCBENCH_DIR/config/openai.cfg"
if [ -f "$CFG" ]; then
    echo "[register] judge config: $CFG (api_version = judge model; ensure api_key is valid)"
else
    echo "[register] WARNING: judge config missing at $CFG — scoring will fail without it"
fi
echo "[register] done. FunctionChat venv: $FC_PY"
