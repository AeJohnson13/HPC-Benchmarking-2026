import torch

print("CUDA available:", torch.cuda.is_available())
print("NCCL available:", torch.distributed.is_nccl_available())