#!/usr/bin/env python3
# ddp_utils.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-02-17

import os
import torch
import torch.distributed as dist


def setup_ddp():
    print(f"[DEBUG] RANK={os.environ.get('RANK')} LOCAL_RANK={os.environ.get('LOCAL_RANK')}", flush=True)
    dist.init_process_group(backend='nccl')
    print(f"[DEBUG] init_process_group DONE for rank {os.environ.get('RANK')}", flush=True)
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank, global_rank


def cleanup_ddp():
    dist.destroy_process_group()
