#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --ntasks=24
#SBATCH --tasks-per-node=24
#SBATCH --gpus=8
#SBATCH --time=01:00:00

export OMP_NUM_THREADS=8

python -c "from torchvision.datasets import CIFAR10; CIFAR10(root='./data', train=True, download=True)"

#python main.py
torchrun --nproc_per_node=8 main_DDP.py
