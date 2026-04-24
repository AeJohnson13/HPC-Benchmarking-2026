# engine.py
# Kohlby Vierthaler (based on code from Alex Johnson)
# Last modified: 4/23/26

import pandas as pd
import torch
import torch.distributed as dist
import time
import os
import argparse

def run_benchmark(model_name, batch_size):
    dist.distributed.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # Model-specific logic
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=False).to(device)
    criterion = torch.nn.CrossEntropyLoss()

    # GPU Warmup
    img = torch.randn(32, 3, 224, 224).to(device)
    target = torch.randint(0, 1000, (32,)).to(device)

    for _ in range(10):
        output = model(img)
        loss = criterion(output, target)
        loss.backward()

    # Benchmarking sequence
    dist.barrier()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    
    for _ in range(50): # Measure 50 iterations
        output = model(img)
        loss = criterion(output, target)
        loss.backward()
    
    end_event.record()
    torch.cuda.synchronize()
    
    if global_rank == 0:
        elapsed_time = start_event.elapsed_time(end_event) / 50 # Avg per batch
        print(f"BENCHMARK_RESULT: {elapsed_time} ms per batch")


if __name__ == "__main__":
    run_benchmark()