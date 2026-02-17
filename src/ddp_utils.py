#!/usr/bin/env python3
# ddp_utils.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-02-17

import os
import torch
import torch.distributed as dist


def setup_ddp():
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return device, local_rank


def cleanup():
    dist.destroy_process_group()
