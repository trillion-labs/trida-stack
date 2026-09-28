#!/bin/bash
# NOTE: published as a worked example, not as a runnable entry point. This script ran on our
# SLURM cluster and sources a private driver tree ($SCRATCH/scripts/lib/) that is not part of this
# repository; paths and partition names are ours. Read it for the methodology behind the numbers in
# the docs, and adapt rather than run. See the README in this directory.
# sbatch -N2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -c 64 --mem=320G --export=ALL,OUT_DIR=... gen_job_2n.sh
srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 --gres=gpu:8 bash $SCRATCH/scripts/lib/draftalign/gen_job.sh
echo "=== GEN 2N DONE ==="
