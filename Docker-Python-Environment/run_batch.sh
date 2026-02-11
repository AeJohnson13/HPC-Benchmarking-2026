#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --gpus=1



export TMPDIR=/home/aejohnson13/.tmp
export SLURM_TMPDIR=/home/aejohnson13/.tmp

df -h
apptainer run --nv /import/unsupported/FIREAID/containers/my-pytorch-app_1.0.sif
