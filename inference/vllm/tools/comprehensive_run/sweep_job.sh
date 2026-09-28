#!/bin/bash
# Concurrency sweep: 1 replica each of vllm-bd4, vllm-causal, sglang-bd4, sglang-causal on 4 GPUs;
# fixed-output GSM8K bench (repo bench_gsm8k_fixed_output.py) at C in CONCS, LIMIT prompts each.
set -u
OUT=$RUN_DIR/sweep; mkdir -p $OUT; cd $OUT
L=$SCRATCH; ROOT=$L/env/vllm-uv27; HD=$L/env/HybridDiffusion
EVALPY=$HD/cache/venvs/hybrid-diffusion-eval/bin/python
export HF_HOME=$HD/cache/huggingface HF_DATASETS_CACHE=$HD/cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
CKPT=$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
VENV="LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 PYTHONPATH=${CODE_DIR:-$L/code/vllm-native-dev9} VLLM_PLUGINS=trida_diffusion VLLM_LOGGING_LEVEL=WARNING"
CONCS=${CONCS:-1 4 8 16}; LIMIT=${LIMIT:-64}; MAXSEQS=${MAXSEQS:-16}
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
PB=$((30000 + (${SLURM_JOB_ID:-$$} % 400) * 10))
declare -A PORT PID
NAMES=(vllm-bd4 vllm-causal sglang-bd4 sglang-causal)
for i in 0 1 2 3; do n=${NAMES[$i]}; g=${GPUS[$i]}; p=$((PB+i)); PORT[$n]=$p
  case $n in
    vllm-bd4)    env $VENV CUDA_VISIBLE_DEVICES=$g PORT=$p CL=4 THRESH=0.90 MAXSTEPS=8 EXTRA_ARGS="--max-num-seqs $MAXSEQS" bash $L/scripts/serve_diff_cl.sh > $OUT/serve_$n.log 2>&1 & ;;
    vllm-causal) env $VENV CUDA_VISIBLE_DEVICES=$g $ROOT/.venv/bin/vllm serve "$CKPT" --port $p --served-model-name trida-bd --tensor-parallel-size 1 --max-model-len 8192 --max-num-seqs $MAXSEQS --gpu-memory-utilization 0.55 --trust-remote-code > $OUT/serve_$n.log 2>&1 & ;;
    sglang-bd4)  ( env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER CUDA_GRAPH_BS="1 2 4 8 16" bash $L/scripts/sglang_launch.sh diffusion $p $g $L/diag/trida_b4_greedy.yaml ) > $OUT/serve_$n.log 2>&1 & ;;
    sglang-causal) ( env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER CUDA_GRAPH_BS="1 2 4 8 16" bash $L/scripts/sglang_launch.sh causal $p $g ) > $OUT/serve_$n.log 2>&1 & ;;
  esac; PID[$n]=$!
done
healthy(){ case $1 in vllm*) curl -s http://localhost:$2/v1/models 2>/dev/null | grep -q trida-bd;; *) [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$2/health 2>/dev/null)" = 200 ];; esac; }
for n in ${NAMES[@]}; do for t in $(seq 1 360); do healthy $n ${PORT[$n]} && break; kill -0 ${PID[$n]} 2>/dev/null || break; sleep 3; done; healthy $n ${PORT[$n]} && echo "$n healthy" || echo "$n NOT HEALTHY"; done
run_one(){ n=$1; healthy $n ${PORT[$n]} || return
  for C in $CONCS; do
    $EVALPY $L/scripts/lib/sweep_client.py --port ${PORT[$n]} --limit $LIMIT --concurrency $C --max_tokens 512 $( [ $n = vllm-bd4 ] && echo --no_sampling ) --save $OUT/bench_${n}_c$C.json > $OUT/bench_${n}_c$C.log 2>&1
    echo "$n C=$C rc=$? : $(tail -1 $OUT/bench_${n}_c$C.log | cut -c1-200)"
  done; }
RP=(); for n in ${NAMES[@]}; do run_one $n & RP+=($!); done; wait "${RP[@]}"   # wait on the bench runners only, not the servers
for n in ${NAMES[@]}; do kill ${PID[$n]} 2>/dev/null; done; sleep 5; for n in ${NAMES[@]}; do pkill -P ${PID[$n]} 2>/dev/null; done
echo "=== SWEEP DONE ==="
