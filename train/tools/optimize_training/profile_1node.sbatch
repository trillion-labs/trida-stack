#!/bin/bash
#SBATCH --job-name=trainprof
#SBATCH --nodes=1 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=64 --mem=900G --exclusive --time=01:00:00
#SBATCH --output=$SCRATCH/models/scratch/slurm/%x-%j.out
# One-node profile of the FLARE two-stream trainer (train/docs/TASK_OPTIMIZE_TRAINING.md step 1).
# Runs the pilot recipe (canvas masks, bd 8) and the production recipe (random masks, bd 32) for a few steps each,
# profiling 2 optimizer steps after 2 warm-up steps. Outputs: <PROF_ROOT>/<variant>/{phase_summary.json,key_averages_*.txt,trace_rank0.json.gz}
set -euo pipefail
ulimit -l unlimited || true
L=$SCRATCH; CODE=$L/code/trida-stack-main; VENV=$SCRATCH/trida-stack/.venv
MODEL_ID=${MODEL_ID:-$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000}
DATASET=${DATASET:-$L/models/scratch/data-draftalign-pilot}
PROF_ROOT=${PROF_ROOT:-$L/runs/trainprof_$(date +%Y%m%d_%H%M)_$SLURM_JOB_ID}; mkdir -p $PROF_ROOT $L/models/scratch/slurm
VARIANTS=${VARIANTS:-"canvas_bd8 random_bd32"}
BUCKETS=${BUCKETS:-2048,4096,8192,16384,20480,24576,28672,32768}; PACK_EXAMPLES=${PACK_EXAMPLES:-11}
PROFILE_START=${PROFILE_START:-2}; PROFILE_STEPS=${PROFILE_STEPS:-2}   # BUCKETS=32768 -> every rank pads to 32k (uniform shapes)
cd $CODE
export HF_HOME=$SCRATCH/trida-stack/.hf_cache PYTHONPATH=. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# VARIANTS entries: <recipe>[:<extra flags with + for spaces>], e.g. "canvas_bd8:--fsdp_keep_params+--compile_mlp"
for spec in $VARIANTS; do
  v=${spec%%:*}; VEXTRA=$( [ "$spec" = "$v" ] && echo "${EXTRA:-}" || echo "${spec#*:}" | tr "+" " " )
  case $v in
    canvas_bd8)  MP=canvas; BD=8;  ARW=1.0;;
    random_bd32) MP=random; BD=32; ARW=0.1;;
    *) echo "unknown variant $v"; exit 2;;
  esac
  vname=$v$( [ -n "$VEXTRA" ] && echo "_$(echo "$VEXTRA" | tr -d " -" )" || true ); mkdir -p $PROF_ROOT/$vname   # `|| true`: under set -e an empty VEXTRA killed the job silently (3739/3744/3745)
  echo "[trainprof] variant=$vname mask=$MP bd=$BD ar_w=$ARW extra='$VEXTRA' buckets=$BUCKETS pack_examples=$PACK_EXAMPLES check=${TRIDA_BLOCK_CONV_CHECK:-1} -> $PROF_ROOT/$vname"
  $VENV/bin/torchrun --standalone --nproc_per_node=8 -m train.train \
      --model_id "$MODEL_ID" --trust_remote_code --dataset "$DATASET" \
      --pack --multiturn --keep_all_reasoning --linear_block_mode bidirectional \
      --bd_size $BD --ar_loss_weight $ARW --mask_pattern $MP --fused_ce $VEXTRA \
      --max_length 32768 --max_response_length 32768 \
      --length_buckets $BUCKETS --response_buckets $BUCKETS \
      --pack_examples $PACK_EXAMPLES --max_packed_rows 1 --micro_batch_size 1 --grad_accum 2 \
      --max_steps 20 --warmup 2 --lr 1e-6 --num_workers 8 --log_every 1 \
      --save_dir $PROF_ROOT/$vname/ckpt_unused --save_every 1000000 \
      --profile_steps $PROFILE_STEPS --profile_start $PROFILE_START --profile_dir $PROF_ROOT/$vname 2>&1 | grep -v -E "Loading weights|fast path is not available" || true
  echo "[trainprof] $vname done"; cat $PROF_ROOT/$vname/phase_summary.json 2>/dev/null || echo "[trainprof] $vname: NO phase_summary.json"
done
echo "===== TRAINPROF DONE $(date) ====="
