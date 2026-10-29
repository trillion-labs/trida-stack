#!/bin/bash
# Submit the comprehensive run as Slurm jobs. Usage: submit_all.sh [smoke]
set -u
L=$SCRATCH; LIB=$L/runs/lib; TS=$(date +%Y%m%d_%H%M%S)
SMOKE=${1:-}; PFX=""; [ -n "$SMOKE" ] && PFX="${SMOKE}_"; RUN_DIR=$L/runs/${PFX}$TS; mkdir -p $RUN_DIR
cd $L/code/vllm-native-dev9 && GIT_SHA=$(git rev-parse --short HEAD 2>/dev/null || echo unknown); cd $L
cat > $RUN_DIR/manifest.json <<M
{"run_dir":"$RUN_DIR","created":"$(date -Is)","git_sha":"$GIT_SHA",
 "code_md5":"$(md5sum $L/code/vllm-native-dev9/vllm_native_diffusion/qwen3_5_diffusion.py | cut -c1-32)",
 "vllm_config":{"canvas_length":4,"threshold":0.90,"max_denoising_steps":8,"cudagraph":"PIECEWISE","max_num_seqs":1},
 "sglang_config":"$L/diag/trida_b4_greedy.yaml (block_size 3, thr 0.9, greedy)",
 "ckpt":"qwen35-4b-flare-v6-2n/step_18000","eval":"HybridDiffusion eval_gsm8k.py, temp 0, top_k 1, no-think, boxed prompt, max_tokens 1024",
 "dataset":"gsm8k main test (HF cache $L/HybridDiffusion/cache/huggingface)"}
M
sub(){ # name engine mode nrep nprob trace ngpu
  sbatch -p p0 --gres=gpu:$7 -c 16 --mem=$((40*$7))G -t 08:00:00 -J $1 -o $RUN_DIR/slurm_$1_%j.out \
    --export=ALL,RUN_DIR=$RUN_DIR,JOB=$1,ENGINE=$2,MODE=$3,NREP=$4,NPROB=$5,TRACE=$6 $LIB/run_job.sh | awk "{print \$4}"; }
subg(){ # name engine mode nrep nprob trace ngpu CL THRESH MAXSTEPS SGL_YAML
  sbatch -p p0 --gres=gpu:$7 -c $((8*$7)) --mem=$((40*$7))G -t 10:00:00 -J $1 -o $RUN_DIR/slurm_$1_%j.out \
    --export=ALL,RUN_DIR=$RUN_DIR,JOB=$1,ENGINE=$2,MODE=$3,NREP=$4,NPROB=$5,TRACE=$6,CL=$8,THRESH=$9,MAXSTEPS=${10},SGL_YAML=${11} $LIB/run_job.sh | awk "{print \$4}"; }
if [ "$SMOKE" = grid ]; then
  # threshold/block grid: b4+b32 at thr 0.80, plus b32 at 0.90 as control (b4/0.90 = today's baseline run). Full GSM8K, 3 replicas each.
  N=${N:-0}; JOBS=""
  for cfg in "b4-t080 4 0.80 8 trida_b4_t08.yaml" "b32-t080 32 0.80 64 trida_b32_t08.yaml" "b32-t090 32 0.90 64 trida_b32_greedy.yaml"; do set -- $cfg
    JOBS="$JOBS $(subg vllm-$1 vllm bd4 3 $N jsonl 3 $2 $3 $4 $L/diag/$5)"
    JOBS="$JOBS $(subg sglang-$1 sglang bd4 3 $N none 3 $2 $3 $4 $L/diag/$5)"
  done
  echo "$JOBS" > $RUN_DIR/jobs.txt
elif [ -n "$SMOKE" ]; then
  J1=$(sub vllm-bd4-trace vllm bd4 1 3 jsonl 1); J2=$(sub sglang-bd4-clean sglang bd4 1 3 none 1); J3=$(sub vllm-bd4-timers vllm bd4 1 2 timers 1)
  echo "$J1 $J2 $J3" > $RUN_DIR/jobs.txt
else
  J1=$(sub vllm-bd4-clean vllm bd4 5 0 none 5)
  J2=$(sub sglang-bd4-clean sglang bd4 4 0 none 4)
  J3=$(sub vllm-causal-clean vllm causal 2 0 none 2)
  J4=$(sub sglang-causal-clean sglang causal 2 0 none 2)
  J5=$(sub vllm-bd4-trace vllm bd4 2 200 jsonl 2)
  J6=$(sub vllm-bd4-timers vllm bd4 1 50 timers 1)
  J7=$(sbatch -p p0 --gres=gpu:4 -c 16 --mem=160G -t 04:00:00 -J sweep -o $RUN_DIR/slurm_sweep_%j.out --dependency=afterany:$J1:$J2:$J3:$J4 --export=ALL,RUN_DIR=$RUN_DIR $LIB/sweep_job.sh | awk "{print \$4}")
  echo "$J1 $J2 $J3 $J4 $J5 $J6 $J7" > $RUN_DIR/jobs.txt
fi
echo "RUN_DIR=$RUN_DIR jobs: $(cat $RUN_DIR/jobs.txt)"; squeue -o "%i %j %T %M %N %b" -u $USER
