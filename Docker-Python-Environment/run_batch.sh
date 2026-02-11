#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --gpus=1




echo TMPDIR=$TMPDIR
echo SLURM_TMPDIR=$SLURM_TMPDIR
df -h
apptainer run --nv /import/unsupported/FIREAID/containers/my-pytorch-app_1.0.sif
