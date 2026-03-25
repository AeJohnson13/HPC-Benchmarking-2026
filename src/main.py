# main.py
# Alex Johnson
# Started 2025-11-09
# Updated 2026-03-20

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
parser.add_argument("--num_nodes", type=str)
parser.add_argument("--job_id", type=str)
args = parser.parse_args()\

os.environ["OMP_NUM_THREADS"] = "8"
torch.set_num_threads(8)

# *******************************
# Main 
# *******************************

def main():
    print("made it to main")
    use_ddp = False
    local_rank = 0
    global_rank = 0
    
    if "RANK" in os.environ:
        device, local_rank, global_rank = setup_ddp()
        use_ddp = True 
    else:
        device = torch.device("cuda")

    gpu_count = torch.cuda.device_count() if use_ddp else 1
    print(gpu_count)
    node_count = args.num_nodes
    print(node_count)
    world_size = gpu_count * node_count
    print(world_size)



    print("after ddp setup", flush=True)
    loader = get_dataloader(use_ddp, world_size)
    print("after dataloader", flush=True)
    model = build_model(device, local_rank, use_ddp)
    print("after model", flush=True)
    optimizer = get_optimizer(model, world_size)
    print("after optimizer", flush=True)
    loss_fn = get_loss_fn()
    print("after loss", flush=True)

    if global_rank == 0:
        print(f'is cuda available {torch.cuda.is_available()}')
        print(f'cuda version: {torch.version.cuda}')
        print(f'number of gpus: {torch.cuda.device_count()}')

        output = []


    #for epoch in range(NUM_EPOCHS):

    epoch_loss = 999_999_999
    epoch = 0
    while epoch_loss > 0.25:
        ## start both timers
        if use_ddp:
            dist.barrier()

        torch.cuda.synchronize()

        if global_rank == 0:
            start_time = time.perf_counter()

        if use_ddp:
            sampler = loader.sampler
            sampler.set_epoch(epoch)

        ## run training loop
        epoch_loss, num_samples = train_epoch(model, optimizer, loss_fn, loader, device)

        ## end timers and compute elapsed
        torch.cuda.synchronize()

        if use_ddp:
            dist.barrier()

        if global_rank == 0:
            epoch_time = time.perf_counter() - start_time

        if use_ddp:
            tensor = torch.tensor([epoch_loss, num_samples], device=device)
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)

            epoch_loss = tensor[0].item()
            num_samples = tensor[1].item()
        
        epoch_loss /= num_samples
        if global_rank == 0: 
            print(f"Loss: {epoch_loss}")
            output.append({"Epoch": epoch, "epoch_time": epoch_time, " loss": epoch_loss})
        
        epoch+=1

    if use_ddp == True : 
        dist.barrier()

    if global_rank == 0:
       
        filename = f"run_{node_count}_{gpu_count}_{args.job_id}.csv"

        df = pd.DataFrame(output)
        df.to_csv(filename, index=False)

        print(f"done training, output saved to {filename}")
    if use_ddp == True : 
        dist.barrier()
        cleanup_ddp

## runs main function when script is called directly 
if __name__ == "__main__":
    main()
