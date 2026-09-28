#!/bin/bash
# Set up Ko-AgentBench for Trida eval (idempotent). No package patching — the target model is
# selected via LiteLLM's OpenAI provider (--model openai/<name> + OPENAI_API_BASE). This builds the
# shared serving venv and syncs Ko-AgentBench's own env.
#
# Judge uses Ko-AgentBench's OPENROUTER_API_KEY (its .env); the run script routes the judge through
# the openrouter/ prefix so scoring hits OpenRouter, not the local Trida endpoint.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KOAB_DIR="${KOAB_DIR:-$SCRATCH/Ko-AgentBench}"

[ -d "$KOAB_DIR" ] || { echo "Ko-AgentBench not found: $KOAB_DIR (set KOAB_DIR)"; exit 1; }

# 1) shared Trida serving venv (.venv-serve)
bash "$REPO/benchmark/serving/setup_serve_venv.sh"

# 2) Ko-AgentBench env (its own .venv via uv sync; python 3.10, litellm/jsonschema)
echo "[register] uv sync Ko-AgentBench env at $KOAB_DIR"
( cd "$KOAB_DIR" && uv sync >/dev/null )

echo "[register] verifying (litellm import)..."
( cd "$KOAB_DIR" && uv run python -c "import litellm, jsonschema; print('  OK: litellm', litellm.__version__)" )

# 3) judge key reminder
if [ -f "$KOAB_DIR/.env" ] && grep -q "OPENROUTER_API_KEY" "$KOAB_DIR/.env"; then
    echo "[register] judge: OPENROUTER_API_KEY present in $KOAB_DIR/.env"
else
    echo "[register] WARNING: OPENROUTER_API_KEY not found in $KOAB_DIR/.env — judging will fail"
fi
echo "[register] done."
