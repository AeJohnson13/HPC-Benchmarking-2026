#!/bin/bash
#SBATCH --partition=l40s

export OMP_NUM_THREADS=8 

export MASTER_ADDR=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
export MASTER_PORT=12345
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

torchrun \
	--nnodes=2 \
	--nproc_per_node=8 \
	--node_rank=$NODE_RANK \
	src/main.py --job_id $SLURM_JOB_ID

