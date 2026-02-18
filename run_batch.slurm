#!/bin/bash
#SBATCH --partition=l40s

export OMP_NUM_THREADS=8 

source /home/${USER}/.bashrc
source activate pytorch_env

torchrun --nproc_per_node=$GPUS src/main.py
