#!/bin/bash
# End-to-end FunctionChat-Bench eval for Trida-7B: launch the shared Trida server pool (chat route),
# point FunctionChat at it (--model inhouse), run the subsets, collect scores.
#
# Env knobs: NUM_GPUS (8), MODEL, PORT, BLOCK_SIZE, THRESHOLD, SUBSETS ("dialog singlecall common"),
#            SAMPLE (small-run cap, optional), FCBENCH_DIR.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
FCBENCH_DIR="${FCBENCH_DIR:-$SCRATCH/FunctionChat-Bench}"
FC_PY="$FCBENCH_DIR/.venv/bin/python"

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
SUBSETS="${SUBSETS:-dialog singlecall common}"
SERVED="$MODEL"                      # sent as the OpenAI "model" field; FunctionChat names its output dir after it
SANITIZED="${SERVED//\//_}"
# NB: FunctionChat's own --sample flag is an unimplemented stub that SKIPS scoring, so we don't use
# it. For a cheap *scored* smoke set FC_LIMIT=N to truncate the raw input to its first N records.
FC_LIMIT="${FC_LIMIT:-}"

maybe_limit() {                      # $1 = raw data filename -> echoes the input path to use
    local raw="$FCBENCH_DIR/data/$1"
    if [ -n "$FC_LIMIT" ]; then
        local dst="$FCBENCH_DIR/output/.smoke_$1"
        head -n "$FC_LIMIT" "$raw" > "$dst"
        echo "$dst"
    else
        echo "$raw"
    fi
}

[ -x "$FC_PY" ] || { echo "FunctionChat venv missing; run benchmark/functionchat/register_functionchat.sh"; exit 1; }

source "$REPO/benchmark/serving/pool.sh"
launch_pool                          # exports REMOTE_OPENAI_BASE_URL
trap cleanup_pool EXIT
echo "[run] chat self-check..."; pool_selfcheck_chat

COMMON=(--model inhouse --base_url "$REMOTE_OPENAI_BASE_URL" --api_key EMPTY
        --served_model_name "$SERVED" --temperature 0.1 --is_batch False)

run_subset() {
    case "$1" in
        dialog)    ARGS=(dialog --input_path "$(maybe_limit FunctionChat-Dialog.jsonl)" --system_prompt_path data/system_prompt.txt) ;;
        singlecall) ARGS=(singlecall --input_path "$(maybe_limit FunctionChat-Singlecall.jsonl)" --tools_type all --system_prompt_path data/system_prompt.txt) ;;
        common)    ARGS=(common --input_path "$(maybe_limit FunctionChat-CallDecision.jsonl)") ;;
        *) echo "[run] unknown subset: $1"; return 1 ;;
    esac
    echo "[run] === FunctionChat $1 ==="
    ( cd "$FCBENCH_DIR" && "$FC_PY" evaluate.py "${ARGS[@]}" "${COMMON[@]}" )
}

for s in $SUBSETS; do run_subset "$s"; done

# Collect: copy FunctionChat's per-model output (incl. FunctionChat-<model>.eval_score.json) here.
SRC="$FCBENCH_DIR/output/$SANITIZED"
mkdir -p "$HERE/output"
if [ -d "$SRC" ]; then
    cp -r "$SRC" "$HERE/output/"
    echo "[run] scores collected under $HERE/output/$SANITIZED/ :"
    ls "$HERE/output/$SANITIZED/" | grep -i eval_score || true
    "$FC_PY" -c "import json,glob,sys;[print(f) or print(json.dumps(json.load(open(f)),ensure_ascii=False,indent=2)[:1200]) for f in glob.glob('$HERE/output/$SANITIZED/*eval_score.json')]" || true
else
    echo "[run] WARNING: expected output dir not found: $SRC"
fi
echo "[run] done."
