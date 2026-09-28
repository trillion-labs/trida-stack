#!/bin/bash
# End-to-end BFCL v4 eval for Trida-7B: launch the shared data-parallel Trida server pool + router
# (benchmark/serving/), point BFCL at it, generate + evaluate, collect scores. Runs interactively
# on a GPU node or from sbatch.
#
# Env knobs: NUM_GPUS (default 8), CATEGORIES, MODEL, THRESHOLD, BLOCK_SIZE, PORT.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
BFCL="${BFCL:-$REPO/.venv-bfcl/bin/bfcl}"       # the bfcl CLI (separate venv; see register_bfcl.sh)

cd "$REPO"
set -a; [ -f ./.env ] && . ./.env; set +a
export HF_HOME="${HF_HOME:-$REPO/.hf_cache}"
export TOKENIZERS_PARALLELISM=false

NUM_GPUS="${NUM_GPUS:-8}"
MODEL="${MODEL:-trillionlabs/Trida-7B-Preview}"   # BFCL-registered key + served-name
CKPT="${CKPT:-$MODEL}"                            # weights to load (a local BD checkpoint dir)
SERVE_PY="${SERVE_PY:-$HERE/serve_trida_openai.py}"
# "all scoring except web_search": non_live (simple/multiple/parallel/irrelevance) + live + multi_turn + agentic memory
CATEGORIES="${CATEGORIES:-non_live,live,multi_turn,memory}"

# BFCL writes result/ and score/ under BFCL_PROJECT_ROOT — keep them under this dir.
export BFCL_PROJECT_ROOT="$HERE"
export REMOTE_OPENAI_API_KEY="EMPTY"
export REMOTE_OPENAI_TOKENIZER_PATH="${REMOTE_OPENAI_TOKENIZER_PATH:-$CKPT}"

source "$REPO/benchmark/serving/pool.sh"
launch_pool                                     # exports REMOTE_OPENAI_BASE_URL
trap cleanup_pool EXIT

echo "[run] launching $NUM_GPUS Trida replicas + router on :$PORT"
UPSTREAMS=""
for i in $(seq 0 $((NUM_GPUS-1))); do
    rport=$((PORT + 1 + i))
    CUDA_VISIBLE_DEVICES=$i "$PY" "$SERVE_PY" \
        --model "$CKPT" --served-name "$MODEL" --device cuda:0 --port "$rport" \
        --block-size "$BLOCK_SIZE" --threshold "$THRESHOLD" ${EXTRA_SERVE_ARGS:-} \
        > "$HERE/replica_${i}.log" 2>&1 &
    PIDS+=($!)
    UPSTREAMS="${UPSTREAMS:+$UPSTREAMS,}http://localhost:${rport}"
done
"$PY" "$HERE/router.py" --port "$PORT" --upstreams "$UPSTREAMS" > "$HERE/router.log" 2>&1 &
PIDS+=($!)

echo "[run] waiting for replicas to load (model init ~minutes)..."
for i in $(seq 0 $((NUM_GPUS-1))); do
    rport=$((PORT + 1 + i))
    pid=${PIDS[$i]}
    ready=0
    for _ in $(seq 1 240); do   # up to ~20 min
        if ! kill -0 "$pid" 2>/dev/null; then
            echo "replica $i (pid $pid) died during startup; log tail:"; tail -30 "$HERE/replica_${i}.log"; exit 1
        fi
        curl -sf "http://localhost:${rport}/health" >/dev/null 2>&1 && { ready=1; break; }
        sleep 5
    done
    [ "$ready" = 1 ] || { echo "replica $i not ready in time; log tail:"; tail -30 "$HERE/replica_${i}.log"; exit 1; }
    echo "[run] replica $i ready"
done
curl -sf "http://localhost:${PORT}/v1/models" >/dev/null || { echo "router not ready"; exit 1; }

# One-shot self-check before the (long) BFCL run: a tool-calling prompt must return non-empty text.
echo "[run] server self-check..."
"$PY" - "$REMOTE_OPENAI_BASE_URL" <<'PYCHK'
import sys, json, urllib.request
base = sys.argv[1]
prompt = ('<|im_start|>system\n# Tools\n<tools>\n{"name":"get_weather","parameters":{"type":"object",'
          '"properties":{"city":{"type":"string"}},"required":["city"]}}\n</tools>\nReturn calls in '
          '<tool_call></tool_call>.<|im_end|>\n<|im_start|>user\nWeather in Seoul?<|im_end|>\n<|im_start|>assistant\n')
req = urllib.request.Request(base + "/completions",
    data=json.dumps({"model":"trida","prompt":prompt,"max_tokens":64,"temperature":0.0}).encode(),
    headers={"content-type":"application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
txt = r["choices"][0]["text"]
print("  self-check output[:200]:", repr(txt[:200]))
assert txt.strip(), "empty completion"
PYCHK

echo "[run] bfcl generate — model=$MODEL categories=$CATEGORIES"
"$BFCL" generate --model "$MODEL" --test-category "$CATEGORIES" --skip-server-setup --num-threads "$((NUM_GPUS*2))"

echo "[run] bfcl evaluate"
"$BFCL" evaluate --model "$MODEL" --test-category "$CATEGORIES"

echo "[run] scores under $BFCL_PROJECT_ROOT/score/ :"
ls -R "$BFCL_PROJECT_ROOT/score" 2>/dev/null | head -40 || true
echo "[run] done."
