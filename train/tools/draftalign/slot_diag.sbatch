#!/bin/bash
#SBATCH --job-name=slotdiag
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=32 --mem=400G --exclusive --time=02:00:00
#SBATCH --output=$SCRATCH/models/scratch/slurm/%x-%j.out
# Offline per-slot draft diagnostic for step_18000 vs pilot step_100 (see slot_diag.py). Two GPUs in parallel.
set -uo pipefail
L=$SCRATCH; CODE=$L/code/trida-stack-main; VENV=$SCRATCH/trida-stack/.venv
OUT=${OUT:-$L/runs/slotdiag_$(date +%Y%m%d_%H%M)_$SLURM_JOB_ID}; mkdir -p $OUT
DET=${DET:-$L/runs/vspec_20260909_172532/s3-n4-30/gsm8k_details_s3-n4-30.json}
cd $CODE; export PYTHONPATH=. HF_HOME=$SCRATCH/trida-stack/.hf_cache TOKENIZERS_PARALLELISM=false
# CKPTS="path:tag path:tag ..." (default: step_18000 vs pilot step_100); one GPU per checkpoint, in parallel
CKPTS=${CKPTS:-"$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000:step18000 $L/models/scratch/trida-4b-draftalign-pilot/step_100:pilot_s100"}
g=0; for spec in $CKPTS; do ck=${spec%%:*}; tag=${spec##*:}
  CUDA_VISIBLE_DEVICES=$g $VENV/bin/python -m train.tools.draftalign.slot_diag --ckpt $ck --details $DET --tag $tag --out $OUT/$tag.json > $OUT/$tag.log 2>&1 &
  g=$((g+1)); done
wait
grep -h -E "^\[|Error|Traceback" $OUT/*.log | grep -v "fast path" | cut -c1-220
echo "===== SLOTDIAG DONE $(date) out=$OUT ====="
