#!/usr/bin/env python3
# data.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-03-20

import torch
import random
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from config import BATCH_SIZE, DATA_DIR, NUM_WORKERS, SAMPLE_SIZE



def get_dataloader(use_ddp, world_size):
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

#    full_training_data = torchvision.datasets.CIFAR10(
#    root=DATA_DIR, 
#    train=True, 
#    download=False, 
#    transform=transform
#    )
    
   
    full_training_data = torchvision.datasets.ImageNet(
        root="/import/beegfs/FIREAID/aejohnson13/img-net-2012/", 
        split="train", 
        transform=transform
    )

    indices = random.sample(range(len(full_training_data)), SAMPLE_SIZE)
    training_data = Subset(full_training_data, indices)



    if use_ddp == True:
        training_sampler = DistributedSampler(training_data)
        training_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=BATCH_SIZE*world_size, 
            sampler=training_sampler,
            num_workers=NUM_WORKERS,
            pin_memory=True
        )
        return training_loader
    else:
        training_loader = torch.utils.data.DataLoader(
            training_data,
            batch_size=BATCH_SIZE*world_size, 
            num_workers=NUM_WORKERS
        ) 
        return training_loader

    
