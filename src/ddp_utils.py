#!/usr/bin/env python3
# ddp_utils.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-03-18

import os
import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    print(f"[DEBUG] RANK={global_rank} LOCAL_RANK={local_rank}", flush=True)
    print(f"[DEBUG] init_process_group DONE for rank {global_rank}", flush=True)
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank, global_rank


def cleanup_ddp():
    dist.destroy_process_group()
