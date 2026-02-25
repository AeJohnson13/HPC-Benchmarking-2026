#!/bin/bash
#SBATCH --partition=l40s

export OMP_NUM_THREADS=8 

source /home/${USER}/.bashrc
source activate pytorch_env

echo "testing from in slurm script $SLURM_JOB_ID"

query_gpu.sh $SLURM_JOB_ID

torchrun --nproc_per_node=$GPUS src/main.py


