#!/usr/bin/env python3
# data.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-02-17

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from config import BATCH_SIZE, DATA_DIR, NUM_WORKERS, SAMPLE_SIZE



def get_dataloader():
    transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
    ])

    full_training_data = torchvision.datasets.CIFAR10(
    root=DATA_DIR, 
    train=True, 
    download=False, 
    transform=transform
    )
    small_indices = torch.randperm(len(full_training_data))[:SAMPLE_SIZE]
    training_data = Subset(full_training_data, small_indices)

    training_sampler = DistributedSampler(training_data)
    training_loader = torch.utils.data.DataLoader(
        training_data,
        batch_size=BATCH_SIZE, 
        sampler=training_sampler,
        num_workers=NUM_WORKERS
    )
    return training_loader
    
