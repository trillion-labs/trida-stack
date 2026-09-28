#!/bin/bash
# sbatch -N2 --ntasks-per-node=1 --gres=gpu:8 --exclusive -c 64 --mem=320G --export=ALL,OUT_DIR=... gen_job_2n.sh
srun --ntasks=${SLURM_NTASKS} --ntasks-per-node=1 --gres=gpu:8 bash $SCRATCH/scripts/lib/draftalign/gen_job.sh
echo "=== GEN 2N DONE ==="
