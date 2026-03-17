#!/bin/bash
#SBATCH --partition=l40s

export OMP_NUM_THREADS=8 

export MASTER_ADDR=$(hostname)
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

srun /home/aejohnson13/miniforge3/envs/pytorch_env/bin/python \
	-m torch.distributed.run \
	--nnodes=2 \
	--nproc_per_node=8 \
	--node_rank=$NODE_RANK \
	--master_addr=$MASTER_ADDR \
	--master_port=$MASTER_PORT \
	src/main.py --job_id $SLURM_JOB_ID

