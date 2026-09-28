#!/bin/bash
# NOTE: published as a worked example, not as a runnable entry point. This script ran on our
# SLURM cluster and sources a private driver tree ($SCRATCH/scripts/lib/) that is not part of this
# repository; paths and partition names are ours. Read it for the methodology behind the numbers in
# the docs, and adapt rather than run. See the README in this directory.
# Two-node Slurm job body (srun one task per node): 8 stock vLLM AR replicas of step_18000 per node + gen_selfdistill.py
# on this node's half. env: OUT_DIR COUNT_PER_NODE(20000) STRIDE(17) MAXTOK(16384)
set -u
L=$SCRATCH; ROOT=$L/env/vllm-uv27; CKPT=$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000
SRC=$SCRATCH/dataset/diffusion-agent/v6; N=${SLURM_PROCID:-0}; OUT=${OUT_DIR}; mkdir -p $OUT/logs
# RESUME_OFFSETS="3000 3200": per-node number of source rows already processed (resume a paused run; output goes to
# shard-n$N$SHARD_SUFFIX.jsonl so the existing shard is never rewritten).
RESUME=0; if [ -n "${RESUME_OFFSETS:-}" ]; then set -- $RESUME_OFFSETS; RESUME=$(eval echo \${$((N+1))}); fi
echo "node $N: resume=$RESUME suffix=${SHARD_SUFFIX:-}"
COMPAT=$L/env/cudacompat/extracted/usr/local/cuda-13.0/compat
NVLIBS=$(find $ROOT/.venv/lib/python3.12/site-packages/nvidia -maxdepth 2 -name lib -type d 2>/dev/null | tr "\n" ":")
VENV="LD_LIBRARY_PATH=$COMPAT:$NVLIBS PATH=$ROOT/.venv/bin:$PATH CPATH=$HOME/.local/share/uv/python/cpython-3.12.13-linux-x86_64-gnu/include/python3.12 HF_HUB_OFFLINE=1"
PB=$((34000 + (${SLURM_JOB_ID:-$$} % 400) * 10)); PORTS=""; PIDS=()
for g in 0 1 2 3 4 5 6 7; do p=$((PB+g)); PORTS="$PORTS $p"
  env $VENV CUDA_VISIBLE_DEVICES=$g $ROOT/.venv/bin/vllm serve "$CKPT" --port $p --served-model-name trida-bd --tensor-parallel-size 1 --max-model-len 32768 --max-num-seqs 16 --gpu-memory-utilization 0.85 --trust-remote-code > $OUT/logs/serve_n${N}_g$g.log 2>&1 & PIDS+=($!); done
OK=""; for p in $PORTS; do for t in $(seq 1 300); do curl -s http://localhost:$p/v1/models 2>/dev/null | grep -q trida-bd && break; sleep 3; done; curl -s http://localhost:$p/v1/models 2>/dev/null | grep -q trida-bd && OK="$OK $p"; done
echo "node $N healthy ports:$OK"
env $VENV TRIDA_REPO=$L/code/trida-stack-main $ROOT/.venv/bin/python $L/scripts/lib/draftalign/gen_selfdistill.py --src $SRC --out $OUT/shard-n$N${SHARD_SUFFIX:-}.jsonl --ckpt $CKPT \
  --ports $OK --stride ${STRIDE:-17} --offset $((N * ${COUNT_PER_NODE:-20000} + RESUME)) --count $(( ${COUNT_PER_NODE:-20000} - RESUME )) --max_tokens ${MAXTOK:-16384} > $OUT/logs/gen_n$N${SHARD_SUFFIX:-}.log 2>&1
echo "node $N gen rc=$? : $(tail -1 $OUT/logs/gen_n$N${SHARD_SUFFIX:-}.log | cut -c1-300)"
for pid in ${PIDS[@]}; do kill $pid 2>/dev/null; done; sleep 5; for pid in ${PIDS[@]}; do pkill -P $pid 2>/dev/null; done
echo "=== GEN node $N DONE ==="
