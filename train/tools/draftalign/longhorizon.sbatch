#!/bin/bash
#SBATCH --job-name=longhorizon
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=32 --mem=400G --exclusive --time=01:00:00
#SBATCH --output=$SCRATCH/models/scratch/slurm/%x-%j.out
set -uo pipefail
L=$SCRATCH; VENV=$SCRATCH/trida-stack/.venv
OUT=${OUT:-$L/runs/longhorizon_$(date +%Y%m%d_%H%M)_$SLURM_JOB_ID}; mkdir -p $OUT
DET=${DET:-$L/runs/vspec_20260909_172532/s3-n4-30/gsm8k_details_s3-n4-30.json}
CK=${CK:-$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000}
cd $L/code/trida-stack-main; export PYTHONPATH=. HF_HOME=$SCRATCH/trida-stack/.hf_cache TOKENIZERS_PARALLELISM=false
g=0; for bd in ${BDS:-8 32}; do
  CUDA_VISIBLE_DEVICES=$g $VENV/bin/python -m train.tools.draftalign.longhorizon --ckpt $CK --details $DET --bd $bd --tag bd$bd --out $OUT/bd$bd.json > $OUT/bd$bd.log 2>&1 &
  g=$((g+1)); done
wait
grep -h -E "^\[|Traceback|Error" $OUT/*.log | grep -v "fast path" | cut -c1-200
echo "===== LONGHORIZON DONE $(date) out=$OUT ====="
