#!/bin/bash
# One-node agentic eval job: vLLM server (data-parallel 8) for one decode mode, then the repo's
# FunctionChat or Ko-AgentBench runner against it (external-endpoint mode of benchmarks/serving/pool.sh).
# env: RUN_DIR JOB TASK=fc|koab N (0 = AR causal, else self-spec gen_block_size) CODE_DIR
#      optional: FC_LIMIT SUBSETS LEVELS KOAB_MAX_TOKENS(16384) KOAB_CONCURRENCY(8) DP(8) MAXLEN(32768) TEMPERATURE(0)
set -u
L=$SCRATCH; ROOT=$L/env/vllm-uv27; REPO=$L/code/trida-stack
OUT=$RUN_DIR/$JOB; mkdir -p $OUT; cd $OUT
CKPT=$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
VENV="LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 PYTHONPATH=${CODE_DIR:-$L/code/vllm-native-dev9} VLLM_PLUGINS=trida_diffusion VLLM_LOGGING_LEVEL=WARNING"
DP=${DP:-8}; MAXLEN=${MAXLEN:-32768}; P=$((31000 + (${SLURM_JOB_ID:-$$} % 400) * 2))
MODEL=${MODEL:-trida-$([ "${N:-0}" = 0 ] && echo causal || echo selfspec-n$N)}
CHAT="--override-generation-config {\"max_new_tokens\":${GEN_MAXTOK:-8192}} --data-parallel-size $DP --max-model-len $MAXLEN --max-num-seqs 1 --enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3 --served-model-name $MODEL"
echo "{\"job\":\"$JOB\",\"task\":\"$TASK\",\"n\":${N:-0},\"model\":\"$MODEL\",\"dp\":$DP,\"maxlen\":$MAXLEN,\"host\":\"$(hostname)\",\"slurm_job\":\"${SLURM_JOB_ID:-}\",\"code_md5\":\"$(md5sum ${CODE_DIR:-$L/code/vllm-native-dev9}/vllm_native_diffusion/qwen3_5_diffusion.py | cut -d" " -f1)\",\"start\":\"$(date -Is)\"}" > $OUT/job.json
if [ "${N:-0}" = 0 ]; then
  env $VENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $ROOT/.venv/bin/vllm serve "$CKPT" --port $P --tensor-parallel-size 1 --gpu-memory-utilization 0.55 --trust-remote-code $CHAT > $OUT/serve.log 2>&1 &
else
  env $VENV CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 PORT=$P CL=$((2*N-2)) THRESH=0.90 MAXSTEPS=8 TRIDA_SELFSPEC_N=$N COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json EXTRA_ARGS="$CHAT" bash $L/scripts/serve_diff_cl.sh > $OUT/serve.log 2>&1 &
fi
SP=$!
for i in $(seq 1 400); do curl -s http://localhost:$P/v1/models 2>/dev/null | grep -q "$MODEL" && break; kill -0 $SP 2>/dev/null || { echo "SERVER DIED"; tail -40 $OUT/serve.log; exit 1; }; sleep 3; done
curl -s http://localhost:$P/v1/models | grep -q "$MODEL" || { echo "SERVER NOT HEALTHY"; exit 1; }
echo "server healthy on $P (dp=$DP model=$MODEL)"
export REMOTE_OPENAI_BASE_URL=http://localhost:$P/v1
export PY=$ROOT/.venv/bin/python   # pool.sh self-check interpreter (no .venv-serve in this checkout)
export PATH=$HOME/.local/bin:$PATH   # uv for Ko-AgentBench
export KOAB_LLM_TIMEOUT=${KOAB_LLM_TIMEOUT:-1200}   # LiteLLM per-call timeout; the 60 s default timed out 64 AR calls
export FC_TEMPERATURE=${TEMPERATURE:-0} KOAB_TEMPERATURE=${TEMPERATURE:-0}
cd $REPO
case $TASK in
  fc)
    # FunctionChat's client is serial -> run the three subsets concurrently (distinct output files).
    RP=(); for s in ${SUBSETS:-dialog singlecall common}; do
      ( MODEL=$MODEL SUBSETS=$s POOL_LOG_DIR=$OUT FC_LIMIT=${FC_LIMIT:-} bash benchmarks/functionchat/run_eval.sh > $OUT/fc_$s.log 2>&1; echo "subset $s rc=$?" ) & RP+=($!); done
    wait "${RP[@]}"
    cp -r benchmarks/functionchat/output/$MODEL $OUT/fc_output 2>/dev/null
    ls $OUT/fc_output 2>/dev/null | grep -i "eval_score\|eval_report" ;;
  koab)
    MODEL=$MODEL SERVED=$MODEL LEVELS=${LEVELS:-L1,L2,L3,L4,L5,L6,L7} KOAB_MAX_TOKENS=${KOAB_MAX_TOKENS:-8192} KOAB_CONCURRENCY=${KOAB_CONCURRENCY:-8} POOL_LOG_DIR=$OUT bash benchmarks/ko_agentbench/run_eval.sh > $OUT/koab.log 2>&1; echo "koab rc=$?"
    cp -r benchmarks/ko_agentbench/reports/openai_${MODEL}_$(date +%Y%m%d) $OUT/koab_report 2>/dev/null
    cat $OUT/koab_report/evaluation_summary.csv 2>/dev/null | cut -d, -f1-3,21 ;;
esac
kill $SP 2>/dev/null; sleep 5; pkill -P $SP 2>/dev/null
echo "{\"end\":\"$(date -Is)\"}" > $OUT/job_end.json
echo "=== AGENTIC JOB $JOB DONE ==="
