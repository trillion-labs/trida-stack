#!/bin/bash
# Per-checkpoint eval (one-node Slurm job each, queued): vLLM self-spec N=4 and N=8 (30 GSM8K items, traces -> tok/fwd +
# accept histogram) and vLLM AR greedy (30 items) for the lossless guard vs step_18000. usage: eval_ckpt.sh <ckpt_dir> <tag>
set -u
CK=$1; TAG=$2; L=$SCRATCH; R=$L/runs/draftalign_eval; mkdir -p $R
# DEP=afterok:<jid> chains the three evals behind a training job (SBATCH_DEPENDENCY was not honored over ssh).
DEPARG=${DEP:+--dependency=$DEP}
NODEARG=${NODE:+-w $NODE}          # NODE=a GPU node pins the evals to one node
sub(){ sbatch $DEPARG $NODEARG -p p0 -N1 --gres=gpu:8 --exclusive -c 64 --mem=320G -t 01:00:00 -J ev-$TAG-$1 -o $R/slurm_${TAG}-$1_%j.out \
  --export=ALL,RUN_DIR=$R,JOB=$TAG-$1,ENGINE=vllm,MODE=$2,NREP=1,NPROB=30,TRACE=jsonl,CODE_DIR=$L/code/vllm-native-dev9,CKPT=$CK,$3 $L/scripts/lib/run_job.sh | awk "{print \$4}"; }
J1=$(sub ss4 bd4 "CL=6,THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=4,COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json")
J2=$(sub ss8 bd4 "CL=14,THRESH=0.90,MAXSTEPS=8,TRIDA_SELFSPEC_N=8,COMPILATION_CONFIG_FILE=$L/runs/cc/full_and_piecewise.json")
J3=$(sub ar causal "MAXLEN=8192")
echo "$TAG: ss4 $J1 ss8 $J2 ar $J3"; echo "$J1 $J2 $J3 $CK" >> $R/jobs.txt
