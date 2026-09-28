#!/bin/bash
# Chain the draft-align experiment in Slurm: [gen job] -> main train (300 steps, ckpt 150/300) -> evals -> control train -> evals.
# usage: autopilot_submit.sh <gen_job_id>
set -u
GEN=$1; L=$SCRATCH; LIB=$L/scripts/lib/draftalign; S=$L/models/scratch; mkdir -p $S/slurm
DATA=$S/data-draftalign-v1; V6=$SCRATCH/dataset/diffusion-agent/v6
STEPS=${STEPS:-300}; SAVE_EVERY=${SAVE_EVERY:-150}
T=$(sbatch --dependency=afterok:$GEN --export=ALL,DATASET=$DATA,SAVE_NAME=trida-4b-draftalign-bd8,MASK_PATTERN=canvas,BD=8,AR_W=1.0,LR=5e-6,LR_MIN_RATIO=1.0,WARMUP=100,MAX_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY $LIB/train_2node.sbatch | awk '{print $4}')
echo "main train: $T (after gen $GEN)"
# evals for the main run: each checkpoint -> 3 one-node jobs, after the training job ends (any state)
EV=""; for st in $(seq $SAVE_EVERY $SAVE_EVERY $STEPS); do
  CK=$S/trida-4b-draftalign-bd8/step_$st; TAG=da-bd8-s$st
  for cfg in "ss4 bd4 CL=6,THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=4,COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json" "ss8 bd4 CL=14,THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=8,COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json" "ar causal MAXLEN=8192"; do set -- $cfg
    J=$(sbatch --dependency=afterany:$T -p p0 -N1 --gres=gpu:8 --exclusive -c 64 --mem=320G -t 01:00:00 -J ev-$TAG-$1 -o $L/runs/draftalign_eval/slurm_${TAG}-$1_%j.out --export=ALL,RUN_DIR=$L/runs/draftalign_eval,JOB=$TAG-$1,ENGINE=vllm,MODE=$2,NREP=1,NPROB=30,TRACE=jsonl,CODE_DIR=$L/code/vllm-native-dev9,CKPT=$CK,$3 $L/scripts/lib/run_job.sh | awk '{print $4}'); EV="$EV:$J"; done; done
echo "main evals: ${EV#:}"
# control: same recipe on the ORIGINAL v6 text (mask pattern only), after the main training job
C=$(sbatch --dependency=afterany:$T --export=ALL,DATASET=$V6,SAVE_NAME=trida-4b-canvasctrl-bd8,MASK_PATTERN=canvas,BD=8,AR_W=1.0,LR=5e-6,LR_MIN_RATIO=1.0,WARMUP=100,MAX_STEPS=$STEPS,SAVE_EVERY=$SAVE_EVERY $LIB/train_2node.sbatch | awk '{print $4}')
echo "control train: $C (after $T)"
CE=""; for st in $(seq $SAVE_EVERY $SAVE_EVERY $STEPS); do
  CK=$S/trida-4b-canvasctrl-bd8/step_$st; TAG=ctrl-bd8-s$st
  for cfg in "ss4 bd4 CL=6,THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=4,COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json" "ss8 bd4 CL=14,THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=8,COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json" "ar causal MAXLEN=8192"; do set -- $cfg
    J=$(sbatch --dependency=afterany:$C -p p0 -N1 --gres=gpu:8 --exclusive -c 64 --mem=320G -t 01:00:00 -J ev-$TAG-$1 -o $L/runs/draftalign_eval/slurm_${TAG}-$1_%j.out --export=ALL,RUN_DIR=$L/runs/draftalign_eval,JOB=$TAG-$1,ENGINE=vllm,MODE=$2,NREP=1,NPROB=30,TRACE=jsonl,CODE_DIR=$L/code/vllm-native-dev9,CKPT=$CK,$3 $L/scripts/lib/run_job.sh | awk '{print $4}'); CE="$CE:$J"; done; done
echo "control evals: ${CE#:}"
echo "{\"gen\":$GEN,\"train\":$T,\"train_evals\":\"${EV#:}\",\"control\":$C,\"control_evals\":\"${CE#:}\",\"submitted\":\"$(date -Is)\"}" > $S/autopilot_chain.json
