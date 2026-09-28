#!/bin/bash
# One-node micro-benchmark: 4 servers (AR, AR+prefix-cache, self-spec N=4, self-spec N=4+prefix-cache) on GPUs 0-3,
# each replaying the same recorded FunctionChat singlecall requests sequentially. env: RUN_DIR NREQ(40) SRC CODE_DIR
set -u
L=$SCRATCH; ROOT=$L/env/vllm-uv27; OUT=$RUN_DIR/fc_bench; mkdir -p $OUT; cd $OUT
CKPT=$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
VENV="LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 PYTHONPATH=${CODE_DIR:-$L/code/vllm-native-dev9} VLLM_PLUGINS=trida_diffusion VLLM_LOGGING_LEVEL=INFO"
PB=$((32000 + (${SLURM_JOB_ID:-$$} % 400) * 4)); SRC=${SRC:-$L/runs/fulleval_20260910/fc-causal/fc_output/FunctionChat-Singlecall.trida-causal.eval.jsonl}
CHAT="--max-model-len 32768 --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3"
declare -A PID PORT
launch(){ n=$1; g=$2; p=$3; shift 3; PORT[$n]=$p
  case $n in
    ar*) env $VENV CUDA_VISIBLE_DEVICES=$g $ROOT/.venv/bin/vllm serve "$CKPT" --port $p --served-model-name $n --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --trust-remote-code $CHAT "$@" > $OUT/serve_$n.log 2>&1 & ;;
    ss*)  env $VENV CUDA_VISIBLE_DEVICES=$g PORT=$p CL=6 THRESH=0.90 MAXSTEPS=8 TRIDA_SELFSPEC_N=4 COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json EXTRA_ARGS="$CHAT --served-model-name $n $*" bash $L/scripts/serve_diff_cl.sh > $OUT/serve_$n.log 2>&1 & ;;
  esac; PID[$n]=$!; }
launch ar     0 $PB
launch ar-pc  1 $((PB+1)) --enable-prefix-caching
launch ss4    2 $((PB+2))
launch ss4-pc 3 $((PB+3)) --enable-prefix-caching
for n in ar ar-pc ss4 ss4-pc; do for t in $(seq 1 400); do curl -s http://localhost:${PORT[$n]}/v1/models 2>/dev/null | grep -q "\"$n\"" && break; kill -0 ${PID[$n]} 2>/dev/null || break; sleep 3; done; curl -s http://localhost:${PORT[$n]}/v1/models 2>/dev/null | grep -q "\"$n\"" && echo "$n healthy" || { echo "$n NOT HEALTHY"; grep -h -E "Error|error" $OUT/serve_$n.log | grep -v "WARNING\|INFO" | tail -3 | cut -c1-200; }; done
RP=(); for n in ar ar-pc ss4 ss4-pc; do curl -s http://localhost:${PORT[$n]}/v1/models 2>/dev/null | grep -q "\"$n\"" || continue
  ( $ROOT/.venv/bin/python $L/scripts/lib/fc_replay_bench.py ${PORT[$n]} $n $SRC ${NREQ:-40} $OUT/replay_$n.json 2>&1 | tail -1 ) & RP+=($!); done
wait "${RP[@]}"
$ROOT/.venv/bin/python - $OUT <<'PY'
import json,sys,os
O=sys.argv[1]
def load(n):
    f=f"{O}/replay_{n}.json"; return json.load(open(f)) if os.path.exists(f) else None
for a,b in (("ss4","ss4-pc"),("ar","ar-pc"),("ar","ss4")):
    A,B=load(a),load(b)
    if A and B:
        same=sum(1 for x,y in zip(A,B) if x.get("tool_calls")==y.get("tool_calls") and x.get("content")==y.get("content"))
        print(f"identity {a} vs {b}: {same}/{min(len(A),len(B))}")
PY
for n in ar ar-pc ss4 ss4-pc; do grep -h -o "Prefix cache hit rate[^,]*\|prefix cache hit rate[^,]*" $OUT/serve_$n.log | tail -1 | sed "s/^/$n: /"; done
for n in ar ar-pc ss4 ss4-pc; do kill ${PID[$n]} 2>/dev/null; done; sleep 5; for n in ar ar-pc ss4 ss4-pc; do pkill -P ${PID[$n]} 2>/dev/null; done
echo "=== FC BENCH DONE ==="
