# main.py
# Alex Johnson
# Started 2025-11-09
# Updated 2026-02-11

"""
Based on code from:
https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html?utm_source=chatgpt.com
"""
# *******************************
# Imports
# *******************************

import os
import time 
import pandas as pd
from datetime import datetime
import torch
import argparse
import torch.distributed as dist

from config import NUM_EPOCHS
from ddp_utils import setup_ddp, cleanup_ddp
from data import get_dataloader
from model import build_model
from train import train_epoch, get_optimizer, get_loss_fn

parser = argparse.ArgumentParser()
parser.add_argument("--job_id", type=str)
args = parser.parse_args()\

# *******************************
# Main 
# *******************************

def main():
    use_ddp = False
    local_rank = 0
    global_rank = 0
    
    if "RANK" in os.environ:
        print("setting up ddp")
        device, local_rank, global_rank = setup_ddp()
        use_ddp = True 
    else:
        device = torch.device("cuda")


    loader = get_dataloader(use_ddp)
    model = build_model(device, local_rank, use_ddp)
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()

    if global_rank == 0:
        print(f'is cuda available {torch.cuda.is_available()}')
        print(f'cuda version: {torch.version.cuda}')
        print(f'number of gpus: {torch.cuda.device_count()}')

        output = []


    for epoch in range(NUM_EPOCHS):

        ## start both timers
        if use_ddp:
            dist.barrier()

        torch.cuda.synchronize()

        if global_rank == 0:
            start_time = time.perf_counter()

        ## run training loop
        train_epoch(model, optimizer, loss_fn, loader, device)

        ## end timers and compute elapsed
        torch.cuda.synchronize()

        if use_ddp:
            dist.barrier()

        if global_rank == 0:
            epoch_time = time.perf_counter() - start_time
            output.append({"Epoch":[epoch], "epoch_time":[epoch_time]})
        
    if use_ddp == True : 
        cleanup_ddp()

    if global_rank == 0:
        gpu_count = dist.get_world_size() if use_ddp else 1
        filename = f"gpu_{gpu_count}_{args.job_id}.csv"

        df = pd.DataFrame(output)
        df.to_csv(filename, index=False)


## runs main function when script is called directly 
if __name__ == "__main__":
    main()
