#!/bin/bash
# Shared Trida serving-pool helpers, sourced by each benchmark's run_eval.sh.
#
# Launches NUM_GPUS replicas of serve_trida_openai.py (one per GPU) behind router.py, waits for
# health, and exports REMOTE_OPENAI_BASE_URL=http://localhost:$PORT/v1. Reuses the .venv-serve
# built by any register_*.sh (transformers 4.57.1 — the released Trida remote code needs 4.x).
#
# Caller sets (or accepts defaults): NUM_GPUS, MODEL, PORT, BLOCK_SIZE, THRESHOLD, PY.
# Usage:
#   source "$REPO/benchmark/serving/pool.sh"
#   launch_pool; trap cleanup_pool EXIT
#   ... use "$REMOTE_OPENAI_BASE_URL" ...

SERVING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_REPO="$(cd "$SERVING_DIR/../.." && pwd)"

PY="${PY:-$_REPO/.venv-serve/bin/python}"
NUM_GPUS="${NUM_GPUS:-8}"
MODEL="${MODEL:-trillionlabs/Trida-7B-Preview}"   # served-name / BFCL-registered key
CKPT="${CKPT:-$MODEL}"                            # weights to load (a local BD checkpoint dir)
SERVE="${SERVE:-serve_trida_openai.py}"
PORT="${PORT:-8000}"
BLOCK_SIZE="${BLOCK_SIZE:-32}"
THRESHOLD="${THRESHOLD:-0.9}"
POOL_LOG_DIR="${POOL_LOG_DIR:-$SERVING_DIR}"

POOL_PIDS=()

cleanup_pool() {
    echo "[pool] shutting down server pool"
    for p in "${POOL_PIDS[@]:-}"; do kill "$p" 2>/dev/null || true; done
}

launch_pool() {
    # If an external OpenAI endpoint is already provided (e.g. a vLLM server the caller launched),
    # use it and skip the local HF replica pool entirely. Backward-compatible: unset -> normal pool.
    if [ -n "${REMOTE_OPENAI_BASE_URL:-}" ]; then
        echo "[pool] external endpoint preset: $REMOTE_OPENAI_BASE_URL — skipping local server pool"
        return 0
    fi
    [ -x "$PY" ] || { echo "[pool] serve python not found: $PY (run register_*.sh first)"; exit 1; }
    echo "[pool] launching $NUM_GPUS replicas ($SERVE) + router on :$PORT (ckpt=$CKPT served=$MODEL)"
    local UPSTREAMS="" i rport
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        rport=$((PORT + 1 + i))
        CUDA_VISIBLE_DEVICES=$i "$PY" "$SERVING_DIR/$SERVE" \
            --model "$CKPT" --served-name "$MODEL" --device cuda:0 --port "$rport" \
            --block-size "$BLOCK_SIZE" --threshold "$THRESHOLD" ${EXTRA_SERVE_ARGS:-} \
            > "$POOL_LOG_DIR/replica_${i}.log" 2>&1 &
        POOL_PIDS+=($!)
        UPSTREAMS="${UPSTREAMS:+$UPSTREAMS,}http://localhost:${rport}"
    done
    "$PY" "$SERVING_DIR/router.py" --port "$PORT" --upstreams "$UPSTREAMS" \
        > "$POOL_LOG_DIR/router.log" 2>&1 &
    POOL_PIDS+=($!)

    echo "[pool] waiting for replicas to load (model init ~minutes)..."
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        rport=$((PORT + 1 + i))
        local pid=${POOL_PIDS[$i]} ready=0
        for _ in $(seq 1 240); do   # up to ~20 min
            if ! kill -0 "$pid" 2>/dev/null; then
                echo "[pool] replica $i (pid $pid) died during startup; log tail:"
                tail -30 "$POOL_LOG_DIR/replica_${i}.log"; exit 1
            fi
            curl -sf "http://localhost:${rport}/health" >/dev/null 2>&1 && { ready=1; break; }
            sleep 5
        done
        [ "$ready" = 1 ] || { echo "[pool] replica $i not ready; log tail:"; tail -30 "$POOL_LOG_DIR/replica_${i}.log"; exit 1; }
        echo "[pool] replica $i ready"
    done
    curl -sf "http://localhost:${PORT}/v1/models" >/dev/null || { echo "[pool] router not ready"; exit 1; }
    export REMOTE_OPENAI_BASE_URL="http://localhost:${PORT}/v1"
    echo "[pool] ready at $REMOTE_OPENAI_BASE_URL"
}

# One-shot chat self-check: a one-tool request must return a tool_call or non-empty content.
pool_selfcheck_chat() {
    "$PY" - "$REMOTE_OPENAI_BASE_URL" "$MODEL" <<'PYCHK'
import sys, json, urllib.request
base, model = sys.argv[1], sys.argv[2]
body = {"model": model, "temperature": 0.0, "max_tokens": 512,
        "messages": [{"role": "user", "content": "서울 날씨 알려줘"}],
        "tools": [{"type": "function", "function": {"name": "get_weather",
                   "description": "도시의 날씨를 조회",
                   "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                                  "required": ["city"]}}}]}
req = urllib.request.Request(base + "/chat/completions", data=json.dumps(body).encode(),
                            headers={"content-type": "application/json"})
r = json.load(urllib.request.urlopen(req, timeout=600))
msg = r["choices"][0]["message"]
print("  self-check finish:", r["choices"][0]["finish_reason"])
print("  tool_calls:", json.dumps(msg.get("tool_calls"), ensure_ascii=False)[:300])
print("  content:", repr((msg.get("content") or "")[:200]))
assert msg.get("tool_calls") or (msg.get("content") or "").strip(), "empty chat completion"
print("  usage:", r["usage"])
PYCHK
}
