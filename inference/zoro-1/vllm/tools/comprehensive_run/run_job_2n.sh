#!/bin/bash
# NOTE: published as a worked example, not as a runnable entry point. This script ran on our
# SLURM cluster and sources a private driver tree ($SCRATCH/scripts/lib/) that is not part of this
# repository; paths and partition names are ours. Read it for the methodology behind the numbers in
# the docs, and adapt rather than run. See the README in this directory.
# Two-node job body: one task per node, each running run_job.sh with NREP replicas on that node's 8 GPUs
# and half of the items. Submit with: sbatch -N2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -c 64 ...
srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 --gres=gpu:8 bash $SCRATCH/scripts/lib/run_job.sh
echo "=== 2N JOB $JOB DONE ==="
