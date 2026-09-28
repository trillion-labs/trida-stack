#!/bin/bash
# usage (inside srun -N2 --ntasks-per-node=1): nccl_probe.sh <label> [ENV=VAL ...]
label=$1; shift
MASTER=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -1)
MADDR=$(getent hosts $MASTER | awk "{print \$1}" | head -1)
env "$@" NCCL_SOCKET_IFNAME=bond.2570 NCCL_DEBUG=WARN timeout 240 $SCRATCH/trida-stack/.venv/bin/torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$SLURM_NODEID --master_addr=$MADDR --master_port=2${SLURM_JOB_ID: -4} $SCRATCH/scripts/lib/draftalign/nccl_probe.py 2>&1 | grep -E "PROBE_OK|NCCL WARN|Error|error" | sort -u | head -4 | sed "s|^|[$label] |"
