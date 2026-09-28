#!/bin/bash
# Full eval matrix: {AR, self-spec N=4, 8, 32} x {gsm8k (2-node, 16 replicas), functionchat, ko_agentbench (1-node DP-8)}.
# Greedy, thinking on. Jobs chained mode by mode; agentic pairs of one mode run side by side (one node each).
# usage: bash submit_fulleval.sh [RUN_DIR]   env: MODES="0 4 8 32" TASKS="gsm8k fc koab"
set -u
L=$SCRATCH; LIB=$L/runs/lib; A=${1:-$L/runs/fulleval_$(date +%Y%m%d_%H%M%S)}; mkdir -p $A
CODE=$L/code/vllm-native-dev9; CC=$L/runs/cc/full_and_piecewise.json
MODES=${MODES:-0 4 8 32}; TASKS=${TASKS:-gsm8k fc koab}
echo "{\"run_dir\":\"$A\",\"created\":\"$(date -Is)\",\"code_md5\":\"$(md5sum $CODE/vllm_native_diffusion/qwen3_5_diffusion.py | cut -d' ' -f1)\",\"protocol\":\"greedy, enable_thinking=true, max_tokens 8192 (gsm8k), FULL cuda graphs, one sequence per replica\"}" > $A/manifest.json
DEP=""; JOBS=""
for N in $MODES; do
  tag=$([ $N = 0 ] && echo causal || echo selfspec-n$N)
  for T in $TASKS; do
    case $T in
      gsm8k)
        if [ $N = 0 ]; then EXP="ENGINE=vllm,MODE=causal,MAXLEN=16384"; else EXP="ENGINE=vllm,MODE=bd4,CL=$((2*N-2)),THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=$N,COMPILATION_CONFIG_FILE=$CC,MAXLEN=16384"; fi
        J=$(sbatch -p p0 -N2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -c 64 --mem=320G -t 06:00:00 -J gsm8k-$tag -o $A/slurm_gsm8k-${tag}_%j.out ${DEP:+--dependency=afterany:$DEP} \
            --export=ALL,RUN_DIR=$A,JOB=gsm8k-$tag,NREP=8,NPROB=1319,TRACE=jsonl,MAXTOK=8192,THINK=1,CODE_DIR=$CODE,$EXP $LIB/run_job_2n.sh | awk '{print $4}')
        DEP=$J ;;
      fc|koab)
        J=$(sbatch -p p0 -N1 --gres=gpu:8 --exclusive -c 64 --mem=320G -t 08:00:00 -J $T-$tag -o $A/slurm_$T-${tag}_%j.out ${DEP:+--dependency=afterany:$DEP} \
            --export=ALL,RUN_DIR=$A,JOB=$T-$tag,TASK=$T,N=$N,CODE_DIR=$CODE $LIB/agentic_job.sh | awk '{print $4}') ;;
    esac
    echo "$J $T $tag"; JOBS="$JOBS $J"
  done
  DEP=$(echo $JOBS | tr ' ' ':' | sed 's/^://')   # next mode waits for everything so far
done
echo "$JOBS" > $A/jobs.txt; echo "RUN_DIR=$A"; squeue -u $USER -o "%.6i %.18j %.3t %.2D %R" | head -20
