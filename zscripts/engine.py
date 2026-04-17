# engine.py
# Kohlby Vierthaler
# 4/15/26

import torch
import torch.distributed as dist
import time
import os

def run_benchmark(model_name, batch_size):
    dist.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    # Model-specific logic
    model = torch.hub.load().cuda()

    # Benchmarking sequence
    start, end = torch.cuda.Event(enable_timing=true), torch.cuda.Event(enable_timing=true)
    
    start.record()
    # Misc paramaters
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end)