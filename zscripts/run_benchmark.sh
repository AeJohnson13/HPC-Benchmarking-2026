#!/bin/bash

# Benchmarking default values
NODES = 1
GPUS = 1
PARTITION = l40s
MODEL = "resnet"

# Argument Parser
while ["$#" -gt > 0]; do 
    case $1 in
        --nodes=*) NODES="${1#*=}"
        --gpus=*) GPUS="${1#*=}"
        --partition=*) PARTITION="${1#*=}"
        --model=*) MODEL="${1#*=}"
    esac
    shift
done

JOB_NAME="bench_${MODEL}_N${NODES}_G${GPUS}"

# Make temporary sbatch script to run
cat <<EOT > temp_bench.sh
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --nodes=$NODES
#SBATCH --ntasks-per-node=$GPUS
#SBATCH --gres=gpu:$GPUS
#SBATCH --partition=$PARTITION
#SBATCH --output=logs/%j_$JOB_NAME.out

# Use srun to launch the PyTorch distributed processes
srun python engine.py --model=$MODEL --nodes=$NODES --gpus_per_node=$GPUS
EOT

sbatch temp_bench.sh
rm temp_bench.sh