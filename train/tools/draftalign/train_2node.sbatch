#!/bin/bash
#SBATCH --job-name=draftalign
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1800G
#SBATCH --exclusive
#SBATCH --time=48:00:00
#SBATCH --output=$SCRATCH/models/scratch/slurm/%x-%j.out
# Draft-aligned fine-tune (see train/docs/DRAFT_ALIGN_PLAN.md). Env knobs (sbatch --export=ALL,...):
#   DATASET   jsonl dir (self-distilled shards, or the original v6 for the control)
#   SAVE_NAME checkpoint dir name under $SCRATCH/models/scratch/
#   MASK_PATTERN canvas|random  BD 8  AR_W 1.0  LR 5e-6  LR_MIN_RATIO 1.0  WARMUP 100  MAX_STEPS 1500  SAVE_EVERY 250
# Source weights: step_18000 via --model_id (fresh optimizer, no --resume). hyungguk's tree is never written.
set -euo pipefail
# Slurm propagates the submitter's soft limits; a Tailscale-SSH login has memlock 8 MB (PAM limits skipped), which
# makes NCCL's RDMA buffer registration fail (ibv_reg_mr ENOMEM). Hard limit is unlimited, so raise it here.
ulimit -l unlimited || true
L=$SCRATCH; CODE=$L/code/trida-stack-main; VENV=$SCRATCH/trida-stack/.venv
MODEL_ID=${MODEL_ID:-$SCRATCH/trida-stack-run/checkpoints/qwen35-4b-flare-v6-2n/step_18000}
DATASET=${DATASET:?}; SAVE_NAME=${SAVE_NAME:?}; SAVE_DIR=$L/models/scratch/$SAVE_NAME
MASK_PATTERN=${MASK_PATTERN:-canvas}; BD=${BD:-8}; AR_W=${AR_W:-1.0}; LR=${LR:-5e-6}; LR_MIN_RATIO=${LR_MIN_RATIO:-1.0}
WARMUP=${WARMUP:-100}; MAX_STEPS=${MAX_STEPS:-1500}; SAVE_EVERY=${SAVE_EVERY:-250}
ACC_METRIC_N=${ACC_METRIC_N:-1024}   # live draft-acceptance metric (0=off)
# Deploy-matched canvas masks (_make_canvas_views): N=4 -> BD=7 PCOLD=0.34 MWARM=3 ; N=8 -> BD=15 PCOLD=0.48 MWARM=7
export TRIDA_CANVAS_PCOLD=${PCOLD:-0.5} TRIDA_CANVAS_MWARM=${MWARM:-0}
AUF_FLOOR=${AUF_FLOOR:--1.0}   # Accept-Until-Fail loss weighting; -1 = off, 0.1 = typical
# Round-1 speed defaults (train/docs/TASK_OPTIMIZE_TRAINING.md, 2026-09-10): uniform 32k rows + 20 convs per pack
# (+10-15% real tokens/step, balanced ranks), compiled MLP (static shapes; -4.7% step, -4 GB), conv check off.
MAX_LEN=${MAX_LEN:-32768}; BUCKETS=${BUCKETS:-32768}
export TRIDA_BLOCK_CONV_CHECK=${TRIDA_BLOCK_CONV_CHECK:-0}
export COMPILE_FLAG=$([ "${COMPILE_MLP:-1}" = 1 ] && echo --compile_mlp || true)
# OFFLOAD=1 adds --activation_offload; hyungguk's final v6-2n runs ("nooff", 30 s/step, 64-68 GB) ran WITHOUT it —
# with it, job 3719 did <5 steps in 25 min (CPU offload of 32k activations per layer).
export OFFLOAD_FLAG=$([ "${OFFLOAD:-0}" = 1 ] && echo --activation_offload || true)   # exported: expanded inside the srun bash -c
MICRO=${MICRO:-1}; ACCUM=${ACCUM:-2}; PACK_EXAMPLES=${PACK_EXAMPLES:-20}; MAX_PACKED_ROWS=${MAX_PACKED_ROWS:-1}
mkdir -p $SAVE_DIR $L/models/scratch/slurm
[ -e $SAVE_DIR/RUN.json ] && { echo "refusing: $SAVE_DIR already has a run (RUN.json). Pick a new SAVE_NAME."; exit 2; }
cat > $SAVE_DIR/RUN.json <<J
{"source_ckpt":"$MODEL_ID","dataset":"$DATASET","dataset_files":"$(ls $DATASET/*.jsonl | xargs -n1 basename | tr "\n" ",")","mask_pattern":"$MASK_PATTERN","bd_size":$BD,"ar_loss_weight":$AR_W,"lr":"$LR","lr_min_ratio":$LR_MIN_RATIO,"warmup":$WARMUP,"max_steps":$MAX_STEPS,"save_every":$SAVE_EVERY,"max_len":$MAX_LEN,"micro":$MICRO,"accum":$ACCUM,"nodes":"$SLURM_JOB_NODELIST","slurm_job":"$SLURM_JOB_ID","code_commit":"$(git -C $CODE rev-parse --short HEAD 2>/dev/null || echo synced-from-local)","start":"$(date -Is)"}
J
cd $CODE
set -a; [ -f $SCRATCH/trida-stack/.env ] && . $SCRATCH/trida-stack/.env; set +a   # WANDB key
export HF_HOME=$SCRATCH/trida-stack/.hf_cache PYTHONPATH=. OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Cross-node transport (probe job 3734, 2026-09-10, 256 MB all-reduce over 16 ranks, memlock unlimited):
#   IB mlx5_2/3/4 on pkey index 1 (0x800c full membership) + RoCE mlx5_0/1: 332 GB/s bus | IB pkey1 alone: 176 |
#   RoCE alone: 30 | TCP over bond.2570 (NCCL_IB_DISABLE=1): 13.5. NCCL's default pkey index 0 is the LIMITED
#   partition 0x7fff -> IBV_WC_RETRY_EXC_ERR (jobs 3697/3704). Override any of these via the environment.
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0} NCCL_IB_PKEY=${NCCL_IB_PKEY:-1} NCCL_IB_GID_INDEX=${NCCL_IB_GID_INDEX:-3} NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond.2570} GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond.2570} NCCL_DEBUG=${NCCL_DEBUG:-WARN}
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MASTER_ADDR=$(srun --nodes=1 --ntasks=1 -w "$MASTER_NODE" --overlap hostname -I | tr " " "\n" | grep "^10\.121\." | head -1)
MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "[draftalign] nodes=$SLURM_JOB_NODELIST master=$MASTER_ADDR:$MASTER_PORT data=$DATASET save=$SAVE_DIR mask=$MASK_PATTERN bd=$BD ar_w=$AR_W lr=$LR steps=$MAX_STEPS"
srun --ntasks-per-node=1 --export=ALL bash -c '
  '"$VENV"'/bin/torchrun --nnodes=$SLURM_NNODES --nproc_per_node=8 --node_rank=$SLURM_NODEID \
    --master_addr='"$MASTER_ADDR"' --master_port='"$MASTER_PORT"' \
    -m train.train \
      --model_id "'"$MODEL_ID"'" --trust_remote_code \
      --dataset "'"$DATASET"'" \
      --pack --multiturn --keep_all_reasoning --linear_block_mode bidirectional \
      --bd_size '"$BD"' --ar_loss_weight '"$AR_W"' --mask_pattern '"$MASK_PATTERN"' --fused_ce ${OFFLOAD_FLAG} ${COMPILE_FLAG} \
      --max_length '"$MAX_LEN"' --max_response_length '"$MAX_LEN"' \
      --length_buckets '"$BUCKETS"' --response_buckets '"$BUCKETS"' \
      --pack_examples '"$PACK_EXAMPLES"' --max_packed_rows '"$MAX_PACKED_ROWS"' \
      --micro_batch_size '"$MICRO"' --grad_accum '"$ACCUM"' \
      --max_steps '"$MAX_STEPS"' --warmup '"$WARMUP"' --lr '"$LR"' --lr_min_ratio '"$LR_MIN_RATIO"' \
      --acc_metric_n '"$ACC_METRIC_N"' --auf_floor '"$AUF_FLOOR"' \
      --num_workers 8 \
      --save_dir "'"$SAVE_DIR"'" --save_every '"$SAVE_EVERY"' --log_every 5 \
      --wandb --wandb_project trida-draftalign --run_name "'"$SAVE_NAME"'"
'
echo "{\"end\":\"$(date -Is)\"}" > $SAVE_DIR/RUN_END.json
echo "===== DRAFTALIGN DONE $(date) ====="
