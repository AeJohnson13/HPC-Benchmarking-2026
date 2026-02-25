#!/bin/bash
#SBATCH --partition=l40s

export OMP_NUM_THREADS=8 

source /home/${USER}/.bashrc
source activate pytorch_env

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

torchrun --nproc_per_node=$GPUS src/main.py

