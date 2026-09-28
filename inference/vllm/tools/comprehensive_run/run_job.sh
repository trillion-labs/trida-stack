#!/bin/bash
# Generic Slurm job body: launch NREP single-GPU replicas of ENGINE/MODE on the GPUs Slurm gave us,
# run the repo GSM8K client (eval_gsm8k.py) round-robin over the replica ports, collect stats, tear down.
# env: RUN_DIR JOB ENGINE(vllm|sglang) MODE(bd4|causal) NREP NPROB(0=full) TRACE(none|jsonl|timers) [MAXTOK=1024]
set -u
OFFSET=${OFFSET:-0}
if [ "${SLURM_NTASKS:-1}" -gt 1 ]; then
  # multi-node job: each task (one per node) runs NREP replicas on its node over its slice of items
  TOTAL=$NPROB; [ "$TOTAL" = 0 ] && TOTAL=1319
  HALF=$(( (TOTAL + SLURM_NTASKS - 1) / SLURM_NTASKS )); OFFSET=$(( SLURM_PROCID * HALF )); NPROB=$HALF
  [ $((OFFSET + NPROB)) -gt $TOTAL ] && NPROB=$((TOTAL - OFFSET))
  JOB=${JOB}-n${SLURM_PROCID}
fi
OUT=$RUN_DIR/$JOB; mkdir -p $OUT; cd $OUT
L=$SCRATCH; ROOT=$L/env/vllm-uv27; HD=$L/env/HybridDiffusion
EVALPY=$HD/cache/venvs/hybrid-diffusion-eval/bin/python
export HF_HOME=$HD/cache/huggingface HF_DATASETS_CACHE=$HD/cache/huggingface/datasets HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
CKPT=$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
VENV="LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 PYTHONPATH=${CODE_DIR:-$L/code/vllm-native-dev9} VLLM_PLUGINS=trida_diffusion VLLM_LOGGING_LEVEL=WARNING"
MAXTOK=${MAXTOK:-1024}
CL=${CL:-4}; THRESH=${THRESH:-0.90}; MAXSTEPS=${MAXSTEPS:-8}; SGL_YAML=${SGL_YAML:-$L/diag/trida_b4_greedy.yaml}
IFS=, read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
PB=$((30000 + (${SLURM_JOB_ID:-$$} % 400) * 10))
echo "{\"job\":\"$JOB\",\"engine\":\"$ENGINE\",\"mode\":\"$MODE\",\"nrep\":$NREP,\"nprob\":$NPROB,\"trace\":\"$TRACE\",\"cl\":$CL,\"thresh\":$THRESH,\"maxsteps\":$MAXSTEPS,\"sgl_yaml\":\"$(basename $SGL_YAML)\",\"gpus\":\"${CUDA_VISIBLE_DEVICES:-}\",\"host\":\"$(hostname)\",\"slurm_job\":\"${SLURM_JOB_ID:-}\",\"code_md5\":\"$(md5sum ${CODE_DIR:-$L/code/vllm-native-dev9}/vllm_native_diffusion/qwen3_5_diffusion.py | cut -c1-32)\",\"start\":\"$(date -Is)\"}" > $OUT/job.json
PIDS=(); PORTS=()
for i in $(seq 0 $((NREP-1))); do
  g=${GPUS[$((i % ${#GPUS[@]}))]}; p=$((PB+i)); PORTS+=($p)
  TENV=""
  case "$TRACE" in jsonl) TENV="TRIDA_TRACE_JSONL=$OUT/trace_rep$i.jsonl";; timers) TENV="TRIDA_TIME_PHASES=1 TRIDA_TIME_GDN=1 TRIDA_COUNT_FWD=1";; esac
  case "$ENGINE/$MODE" in
    vllm/bd4)    env $VENV $TENV CUDA_VISIBLE_DEVICES=$g PORT=$p CL=$CL THRESH=$THRESH MAXSTEPS=$MAXSTEPS EXTRA_ARGS="${EXTRA_ARGS:-}" bash $L/scripts/serve_diff_cl.sh > $OUT/serve_rep$i.log 2>&1 & ;;
    vllm/causal) env $VENV CUDA_VISIBLE_DEVICES=$g $ROOT/.venv/bin/vllm serve "$CKPT" --port $p --served-model-name trida-bd --tensor-parallel-size 1 --max-model-len ${MAXLEN:-8192} --max-num-seqs ${MAXSEQS:-1} --gpu-memory-utilization 0.55 --trust-remote-code > $OUT/serve_rep$i.log 2>&1 & ;;
    sglang/bd4)  ( env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER ${SGL_ENV:-} bash $L/scripts/sglang_launch.sh diffusion $p $g $SGL_YAML ) > $OUT/serve_rep$i.log 2>&1 & ;;
    sglang/causal) ( env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER ${SGL_ENV:-} bash $L/scripts/sglang_launch.sh causal $p $g ) > $OUT/serve_rep$i.log 2>&1 & ;;
    sglang/selfspec) ( env -i HOME=$HOME PATH=/usr/bin:/bin USER=$USER ${SGL_ENV:-} bash $L/scripts/sglang_launch.sh self-spec $p $g $SGL_YAML ) > $OUT/serve_rep$i.log 2>&1 & ;;
    *) echo "bad ENGINE/MODE $ENGINE/$MODE"; exit 2;;
  esac
  PIDS+=($!)
done
healthy(){ if [ $ENGINE = vllm ]; then curl -s http://localhost:$1/v1/models 2>/dev/null | grep -q trida-bd; else [ "$(curl -s -o /dev/null -w "%{http_code}" http://localhost:$1/health 2>/dev/null)" = 200 ]; fi; }
OK=()
for i in $(seq 0 $((NREP-1))); do
  for t in $(seq 1 360); do healthy ${PORTS[$i]} && break; kill -0 ${PIDS[$i]} 2>/dev/null || break; sleep 3; done
  healthy ${PORTS[$i]} && OK+=(${PORTS[$i]}) && echo "rep$i port ${PORTS[$i]} healthy" || echo "rep$i port ${PORTS[$i]} NOT HEALTHY"
done
[ ${#OK[@]} -gt 0 ] || { echo "no healthy replicas"; kill ${PIDS[@]} 2>/dev/null; exit 1; }
stats(){ for p in ${OK[@]}; do echo -n "\"$p\":"; curl -s http://localhost:$p/get_server_info | python3 -c "import sys,json; d=json.load(sys.stdin); print(json.dumps(d.get(\"internal_states\",[{}])[0].get(\"dllm_stats\",{})))"; echo ","; done; }
[ $ENGINE = sglang ] && { echo "{"; stats; echo "\"_\":0}"; } > $OUT/dllm_stats_before.json
T0=$(date +%s)
TEMP=0; [ "$ENGINE/$MODE" = vllm/bd4 ] && TEMP=-1   # vLLM diffusion rejects sampling params; it is greedy by construction
$EVALPY $L/scripts/lib/eval_gsm8k_greedy.py --ports ${OK[@]} --model trida-bd --num-problems $NPROB --max-tokens $MAXTOK \
  --temperature $TEMP --top-p 1.0 --top-k 1 --presence-penalty 0 $( [ "${THINK:-0}" = 1 ] || echo --disable-thinking ) --offset $OFFSET --max-workers ${#OK[@]} --timeout 3600 --output-dir $OUT --tag $JOB > $OUT/eval.log 2>&1
RC=$?; T1=$(date +%s)
[ $ENGINE = sglang ] && { echo "{"; stats; echo "\"_\":0}"; } > $OUT/dllm_stats_after.json
tail -12 $OUT/eval.log
echo "{\"eval_rc\":$RC,\"eval_wall_s\":$((T1-T0)),\"healthy_ports\":\"${OK[*]}\",\"end\":\"$(date -Is)\"}" > $OUT/job_end.json
kill ${PIDS[@]} 2>/dev/null; sleep 5
for pid in ${PIDS[@]}; do pkill -P $pid 2>/dev/null; done; sleep 3
echo "=== JOB $JOB DONE rc=$RC ==="
