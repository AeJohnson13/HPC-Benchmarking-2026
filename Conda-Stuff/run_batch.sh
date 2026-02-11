#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --gpus=2
#SBATCH --time=01:00:00

torchrun main_DDP.py
