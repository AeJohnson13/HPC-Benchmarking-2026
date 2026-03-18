#!/bin/bash
#SBATCH --partition=l40s

export OMP_NUM_THREADS=2 
export NODE_RANK=$SLURM_NODEID


source /home/${USER}/.bashrc
conda activate pytorch_env

./query_gpu.sh $SLURM_JOB_ID &
query_pid=$!
echo $query_pid

cleanup() {
	kill -TERM "$query_pid" 2>/dev/null
	wait "$query_pid" 2>/dev/null
}

trap cleanup SIGINT EXIT

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=$(node_array[0])
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip
export LOGLEVEL=INFO

srun /home/aejohnson13/miniforge3/envs/pytorch_env/bin/torchrun \
  --nnodes=$nodes \
  --nproc_per_node=$gpus \
  --rdzv_id $RANDOM \
  --rdzv_backend c10d \
  --rdzv_endpoint $head_node_ip:29500 \
  /home/aejohnson13/HPC-Benchmarking-2026/src/main.py \
  --job_id $SLURM_JOB_ID




#srun --ntasks=$SLURM_NNODES --ntasks-per-node=1 \
#/home/${USER}/miniforge3/envs/pytorch_env/bin/python \
#  -m torch.distributed.run \
#  --nnodes=$SLURM_NNODES \
#  --nproc_per_node=8 \
#  --node_rank=$NODE_RANK \
#  --master_addr=$MASTER_ADDR \
#  --master_port=$MASTER_PORT \
#  src/main.py --job_id $SLURM_JOB_ID

