#!/bin/bash
# End-to-end tau2-bench eval for Trida-7B: launch the shared Trida server pool (chat route) as the
# AGENT, run the user-simulated multi-turn conversations (user-sim on OpenRouter), collect rewards.
#
# tau2 runs --max-concurrency conversations in parallel, each issuing serial agent calls, so the
# agent pool benefits from data-parallel replicas — CONCURRENCY defaults to NUM_GPUS.
#
# Env knobs: NUM_GPUS (8), MODEL, PORT, BLOCK_SIZE, THRESHOLD, DOMAINS ("airline retail telecom"),
#            NUM_TRIALS (1), CONCURRENCY (=NUM_GPUS), NUM_TASKS (cap; optional), TASK_IDS (optional),
#            USER_LLM (openrouter/openai/gpt-4o-mini), SERVED, TAU2_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
TAU2_DIR="${TAU2_DIR:-$SCRATCH/tau2-bench}"

cd "$REPO"
set -a; [ -f ./.env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

export NUM_GPUS="${NUM_GPUS:-8}"
export MODEL="${MODEL:-trillionlabs/Trida-7B-Preview}"
export PORT="${PORT:-8000}"
export BLOCK_SIZE="${BLOCK_SIZE:-32}"
export THRESHOLD="${THRESHOLD:-0.9}"
export POOL_LOG_DIR="$HERE"
DOMAINS="${DOMAINS:-airline retail telecom}"
NUM_TRIALS="${NUM_TRIALS:-1}"
CONCURRENCY="${CONCURRENCY:-$NUM_GPUS}"
USER_LLM="${USER_LLM:-openrouter/openai/gpt-4o-mini}"   # user simulator on OpenRouter
SERVED="${SERVED:-trida-7b-preview}"                    # LiteLLM agent model label (server ignores it)

[ -d "$TAU2_DIR/.venv" ] || { echo "tau2 not set up; run benchmark/tau2/register_tau2.sh"; exit 1; }
[ -n "${OPENROUTER_API_KEY:-}" ] || { echo "OPENROUTER_API_KEY not set (needed for the user simulator); add it to $REPO/.env"; exit 1; }
export OPENROUTER_API_KEY
export TAU2_DATA_DIR="${TAU2_DATA_DIR:-$TAU2_DIR/data}"   # domains load from here; sims written here

source "$REPO/benchmark/serving/pool.sh"
launch_pool                                             # exports REMOTE_OPENAI_BASE_URL
trap cleanup_pool EXIT
echo "[run] chat self-check..."; pool_selfcheck_chat

# Agent -> local Trida server via LiteLLM's openai/ provider (api_base spread into completion()).
# max_tokens caps each turn's generation so a reasoning model's verbose <think> doesn't accumulate
# and blow past the model's context window over a long multi-turn conversation (AGENT_MAX_TOKENS).
AGENT_LLM_ARGS="{\"temperature\":0.0,\"max_tokens\":${AGENT_MAX_TOKENS:-2048},\"api_base\":\"$REMOTE_OPENAI_BASE_URL\",\"api_key\":\"EMPTY\"}"

summarize() {  # $1 = results.json
    "$TAU2_DIR/.venv/bin/python" - "$1" <<'PY'
import json, sys
sims = json.load(open(sys.argv[1])).get("simulations", [])
n = len(sims)
rw = [ (s.get("reward_info") or {}).get("reward", 0.0) or 0.0 for s in sims ]
succ = sum(1 for r in rw if r >= 0.999)
avg = sum(rw)/n if n else 0.0
print(f"    sims={n}  avg_reward={avg:.3f}  pass@1={succ}/{n} ({(succ/n if n else 0):.3f})")
PY
}

for d in $DOMAINS; do
    RUN_NAME="trida_${d}"
    # clear any stale save for this run name so --auto-resume can't conflict on a changed task set
    # ("Tasks were removed from the task set"); set TAU2_KEEP_RESUME=1 to genuinely resume instead.
    [ "${TAU2_KEEP_RESUME:-0}" = "1" ] || rm -rf "$TAU2_DATA_DIR/simulations/$RUN_NAME" 2>/dev/null || true
    echo "[run] === tau2 domain=$d  agent=openai/$SERVED  user=$USER_LLM  trials=$NUM_TRIALS conc=$CONCURRENCY ==="
    ARGS=(run --domain "$d"
          --agent-llm "openai/$SERVED" --agent-llm-args "$AGENT_LLM_ARGS"
          --user-llm "$USER_LLM" --user-llm-args '{"temperature":0.0}'
          --num-trials "$NUM_TRIALS" --max-concurrency "$CONCURRENCY"
          --timeout "${TAU2_SIM_TIMEOUT:-900}"
          --save-to "$RUN_NAME" --auto-resume)
    [ -n "${NUM_TASKS:-}" ] && ARGS+=(--num-tasks "$NUM_TASKS")
    [ -n "${TASK_IDS:-}" ] && ARGS+=(--task-ids "$TASK_IDS")
    ( cd "$TAU2_DIR" && uv run tau2 "${ARGS[@]}" )

    SRC="$TAU2_DATA_DIR/simulations/$RUN_NAME/results.json"
    mkdir -p "$HERE/results/$RUN_NAME"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$HERE/results/$RUN_NAME/results.json"
        echo "[run] $d results -> $HERE/results/$RUN_NAME/results.json"
        summarize "$SRC"
    else
        echo "[run] WARNING: no results.json for $d at $SRC"
    fi
done
echo "[run] done."
