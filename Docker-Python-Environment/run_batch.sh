#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --gpus=1



IMAGE=/import/unsupported/FIREAID/containers/my-pytorch-app_1.0.sif
SCRIPT="./main.py"

apptainer run --nv IMAGE \
    torchrun $SCRIPT
