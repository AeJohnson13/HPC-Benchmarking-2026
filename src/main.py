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

import time 
import pandas as pd
from datetime import datetime
import torch

from config import NUM_EPOCHS
from ddp_utils import setup_ddp, cleanup_ddp
from data import get_dataloader
from models import build_model
from train import train_epoch, get_optimizer, get_loss_fn


# *******************************
# Hardware check
# *******************************
print(f'is cuda available {torch.cuda.is_available()}')
print(f'cuda version: {torch.version.cuda}')
print(f'number of gpus: {torch.cuda.device_count()}')

# *******************************
# Main 
# *******************************


def main():
    device, local_rank = setup_ddp()
    loader = get_dataloader(device)
    model = build_model(device, local_rank)
    optimizer = get_optimizer(model)
    loss_fn = get_loss_fn()

    if local_rank == 0:
        print(f'is cuda available {torch.cuda.is_available()}')
        print(f'cuda version: {torch.version.cuda}')
        print(f'number of gpus: {torch.cuda.device_count()}')


    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    df = pd.DataFrame()

    for epoch in range(NUM_EPOCHS):

        ## start both timers
        start_time = time.perf_counter()
        start.record()

        ## run training loop
        train_epoch(model, optimizer, loss_fn, loader, device)

        ## end timers and compute elapsed
        end.record()
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_time
        gpu_time = start.elapsed_time(end)


        # recod times
        new_data = pd.DataFrame({"Epoch":[epoch], "epoch_time":[epoch_time], "gpu_time":[gpu_time/1000]})
        df = pd.concat([df, new_data], ignore_index=True)
        
    cleanup_ddp()
    curr_time = datetime.now().strftime("%m%d_%H%M")
    gpu_count = torch.cuda.device_count()

    filename = f"gpu_0_{curr_time}_{gpu_count}.csv"
    df.to_csv(filename, index=False)


## runs main function when script is called directly 
if __name__ == "__main__":
    main()
