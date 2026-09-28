#!/bin/bash
# End-to-end Ko-AgentBench eval for Trida-7B: launch the shared Trida server pool (chat route),
# point Ko-AgentBench's LiteLLM adapter at it, run the agent trajectories, then score.
#
# Ko-AgentBench is SEQUENTIAL (one model call at a time), so one replica suffices (NUM_GPUS default
# 1). Tools are cache-backed (--cache-mode read → no external API keys). The judge is routed through
# the openrouter/ prefix so it hits OpenRouter (not the local Trida endpoint).
#
# Env knobs: NUM_GPUS (1), MODEL, PORT, BLOCK_SIZE, THRESHOLD, LEVELS, SERVED, JUDGE, KOAB_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
KOAB_DIR="${KOAB_DIR:-$SCRATCH/Ko-AgentBench}"

cd "$REPO"
set -a; [ -f ./.env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

export NUM_GPUS="${NUM_GPUS:-8}"       # 8 replicas (data-parallel); tasks dispatched concurrently below
export MODEL="${MODEL:-trillionlabs/Trida-7B-Preview}"
export PORT="${PORT:-8000}"
export BLOCK_SIZE="${BLOCK_SIZE:-32}"
export THRESHOLD="${THRESHOLD:-0.9}"
export POOL_LOG_DIR="$HERE"
LEVELS="${LEVELS:-L1,L2,L3,L4,L5,L6,L7}"
SERVED="${SERVED:-trida-7b-preview}"   # LiteLLM sends this as the model id (server ignores it); no gpt-5 substring
JUDGE="${JUDGE:-openrouter/openai/gpt-4.1-mini}"  # openrouter/ prefix -> OpenRouter (uses OPENROUTER_API_KEY)
DATE="$(date +%Y%m%d)"

[ -d "$KOAB_DIR/.venv" ] || { echo "Ko-AgentBench env missing; run benchmark/ko_agentbench/register_ko_agentbench.sh"; exit 1; }

source "$REPO/benchmark/serving/pool.sh"
launch_pool                            # exports REMOTE_OPENAI_BASE_URL
trap cleanup_pool EXIT
echo "[run] chat self-check..."; pool_selfcheck_chat

# Stage 1 — run trajectories. OPENAI_API_BASE/KEY are scoped to THIS command so only the agent model
# (openai/ provider) hits the local server; the judge in stage 2 is unaffected.
echo "[run] === stage 1: run trajectories (levels=$LEVELS) ==="
( cd "$KOAB_DIR" && OPENAI_API_KEY=EMPTY OPENAI_API_BASE="$REMOTE_OPENAI_BASE_URL" \
    uv run run_benchmark_with_logging.py --levels "$LEVELS" --model "openai/$SERVED" --cache-mode read \
      --timeout "${KOAB_TIMEOUT:-1200}" --concurrency "${KOAB_CONCURRENCY:-8}" )   # 8-way concurrent -> uses 8 replicas; high per-call timeout for the slow reasoning model

# Stage 2 — score (judge via OpenRouter; no OPENAI_API_BASE here so openrouter/ routing is used).
echo "[run] === stage 2: evaluate (judge=$JUDGE) ==="
( cd "$KOAB_DIR" && uv run evaluate_model_run.py --date "$DATE" --model "openai/$SERVED" \
    --judge-models "$JUDGE" --levels "$LEVELS" --format all )

# Collect the report.
SANITIZED="openai_${SERVED}"
SRC="$KOAB_DIR/reports/${SANITIZED}_${DATE}"
mkdir -p "$HERE/reports"
if [ -d "$SRC" ]; then
    cp -r "$SRC" "$HERE/reports/"
    echo "[run] report collected under $HERE/reports/${SANITIZED}_${DATE}/ :"
    ls "$HERE/reports/${SANITIZED}_${DATE}/" || true
else
    echo "[run] WARNING: expected report dir not found: $SRC (check $KOAB_DIR/reports/)"
    ls "$KOAB_DIR/reports/" 2>/dev/null | tail -5 || true
fi
echo "[run] done."
