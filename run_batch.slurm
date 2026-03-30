#!/bin/bash
#SBATCH --partition=l40s

export NODE_RANK=$SLURM_NODEID


source /home/${USER}/.bashrc
conda activate pytorch_env


module load slurm

nodes=( $( scontrol show hostnames $SLURM_NODELIST ) )
node_array=($nodes)
head_node=${node_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" /bin/hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

GPUS=$SLURM_GPUS_ON_NODE


srun /home/$USER/miniforge3/envs/pytorch_env/bin/torchrun \
  --nnodes=$NUM_NODES \
  --nproc_per_node=$GPUS \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  /home/aejohnson13/HPC-Benchmarking-2026/src/main.py \
  --job_id $SLURM_JOB_ID \
  --num_nodes $NUM_NODES



