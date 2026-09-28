#!/bin/bash
# Set up tau2-bench (sierra-research/tau2-bench) for Trida eval (idempotent). No package patching —
# the agent model is selected via LiteLLM (--agent-llm openai/<name> + api_base in --agent-llm-args).
# This builds the shared serving venv and syncs tau2-bench's own uv env.
#
# The user simulator runs on OpenRouter (OPENROUTER_API_KEY). tau2's NL-assertion judge is hardcoded
# to gpt-4.1 (OpenAI) but only fires for tasks whose reward_basis includes NL assertions; airline/
# retail/telecom are mostly DB/action-graded. Set OPENAI_API_KEY if you hit NL-assertion tasks.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
UV="${UV:-uv}"
TAU2_DIR="${TAU2_DIR:-$SCRATCH/tau2-bench}"

# 1) shared Trida serving venv (.venv-serve)
bash "$REPO/benchmark/serving/setup_serve_venv.sh"

# 2) tau2-bench checkout (clone stock upstream if absent) + uv sync
if [ ! -d "$TAU2_DIR" ]; then
    echo "[register] cloning sierra-research/tau2-bench -> $TAU2_DIR"
    git clone --depth 1 https://github.com/sierra-research/tau2-bench "$TAU2_DIR"
fi
echo "[register] uv sync tau2-bench at $TAU2_DIR (python 3.12; tau2 core + LiteLLM)"
( cd "$TAU2_DIR" && "$UV" sync )

echo "[register] verifying tau2 CLI + LiteLLM OpenRouter routing"
( cd "$TAU2_DIR" && "$UV" run tau2 --help >/dev/null && \
  "$UV" run python -c "from litellm import get_llm_provider; m,p,_,_=get_llm_provider('openrouter/openai/gpt-4o-mini'); print('  OK: OpenRouter routing ->', m, 'via', p)" )

# 3) key reminders
KEYSRC=""
[ -f "$REPO/.env" ] && grep -q OPENROUTER_API_KEY "$REPO/.env" 2>/dev/null && KEYSRC="$REPO/.env"
if [ -n "$KEYSRC" ] || [ -n "${OPENROUTER_API_KEY:-}" ]; then
    echo "[register] user-simulator judge key: OPENROUTER_API_KEY available (${KEYSRC:-env})"
else
    echo "[register] WARNING: OPENROUTER_API_KEY not found — the user simulator will fail. Add it to $REPO/.env"
fi
echo "[register] done. TAU2_DIR=$TAU2_DIR"
