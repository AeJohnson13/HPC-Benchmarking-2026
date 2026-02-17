#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --gpus=8
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=8

source /home/${USER}/.bashrc
source activate pytorch_env


#python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', train=True, download=True)"



torchrun --nproc_per_node=8 src/main.py
