#!/bin/bash
#SBATCH --partition=l40s
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$GPUS
#SBATCH --gpus-per-node=$GPUS

export OMP_NUM_THREADS=8 

export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=12345

export NODE_RANK=$SLURM_NODEID


source /home/${USER}/.bashrc
conda activate pytorch_env

#RESULT_DIR="./results/job_$SLURM_JOB_ID"

#mkdir -p $RESULT_DIR

#cd $RESULT_DIR

./query_gpu.sh $SLURM_JOB_ID &
query_pid=$!
echo $query_pid

cleanup() {
	kill -TERM "$query_pid" 2>/dev/null
	wait "$query_pid" 2>/dev/null
}

trap cleanup SIGINT EXIT

torchrun \
	--nnodes=$NODES \
	--nproc_per_node=$GPUS \
	--node_rank=$NODE_RANK \
	src/main.py --job_id $SLURM_JOB_ID

