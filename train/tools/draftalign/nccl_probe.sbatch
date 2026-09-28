#!/bin/bash
#SBATCH --job-name=nccl-probe
#SBATCH --nodes=2 --ntasks-per-node=1 --gres=gpu:8 --cpus-per-task=32 --mem=200G --exclusive --time=00:15:00
#SBATCH --output=$SCRATCH/models/scratch/slurm/%x-%j.out
# NOTE: published as a worked example, not as a runnable entry point. This script ran on our
# SLURM cluster and sources a private driver tree ($SCRATCH/scripts/lib/) that is not part of this
# repository; paths and partition names are ours. Read it for the methodology behind the numbers in
# the docs, and adapt rather than run. See the README in this directory.
# Which cross-node transport does NCCL get with memlock unlimited? (RoCE mlx5_0/1 works in ib_write_bw at 11 GB/s/card)
ulimit -l unlimited; echo "memlock=$(ulimit -l)"
P=$SCRATCH/scripts/lib/draftalign/nccl_probe.sh
srun --ntasks-per-node=1 --export=ALL bash -c "ulimit -l unlimited; $P roce_mlx5_0_1 NCCL_IB_HCA=mlx5_0,mlx5_1 NCCL_IB_GID_INDEX=3 || true; $P ib_pkey1 NCCL_IB_HCA=mlx5_2,mlx5_3,mlx5_4 NCCL_IB_PKEY=1 || true; $P ib_pkey1_plus_roce NCCL_IB_PKEY=1 NCCL_IB_GID_INDEX=3 || true; $P default_all_hcas || true; $P roce_gid_auto NCCL_IB_HCA=mlx5_0,mlx5_1 || true; $P tcp_baseline NCCL_IB_DISABLE=1 || true"
