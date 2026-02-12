# main_DDP.py
# Alex Johnson
# Started 2025-11-09
# Updated 2026-02-11

"""
Based on code from:
https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html
https://docs.pytorch.org/tutorials/intermediate/ddp_tutorial.html
"""

# *******************************
# Imports
# *******************************
import os
    # for os.environ
import sys
import tempfile
import time 
    # for time.perf_counter()

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

import torchvision
import torchvision.transforms as transforms

from torchvision.models import ResNet50_Weights
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# *******************************
# Doing a bad thing
# *******************************
sys.stderr = open(os.devnull, "w") # discard errors

# *******************************
# Configuration
# *******************************
NUM_EPOCHS = 10 
BATCH_SIZE = 32
LEARNING_RATE = 0.001
SAMPLE_SIZE = 50000
NUM_WORKERS = 1   # check on this
DATA_DIR = './data'


# *******************************
# Data Transform
# *******************************
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


# *******************************
# Per Epoch Training
# *******************************
def train_epoch():
    
    model.train(True)
    
    for i, data in enumerate(training_loader):
        # Every data instance is an input + label pair
        inputs, labels = data
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute the loss and its gradients
        loss = criterion(outputs, labels)
        loss.backward()

        #Adjust learning weights
        optimizer.step()



# *******************************
# Main 
# *******************************


def main():
    # *******************************
    # Setup DDP
    # *******************************
    dist.init_process_group(backend='nccl')

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    device = torch.device(f"cuda:{local_rank}")


    # *******************************
    # Hardware check
    # *******************************
    if dist.get_rank() == 0:
        print(f'cuda version: {torch.version.cuda}')
        print(f'number of gpus: {torch.cuda.device_count()}')

    # *******************************
    # Dataset Setup and Dataloader
    # *******************************

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

    # *******************************
    # Model, loss, optimizer
    # *******************************
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    



    # *******************************
    # Start Training/Timing runs
    # *******************************
    print(f"starting epochs on gpu {dist.get_rank()}")
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    for epoch in range(NUM_EPOCHS):

        ## start both timers
        start_time = time.perf_counter()
        start.record()

        ## run training loop
        train_epoch()

        ## end timers and compute elapsed
        end.record()
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - start_time
        gpu_time = start.elapsed_time(end)

        ## print times //TODO record and average
        print(f"Epoch {epoch} time (perf_counter): {epoch_time:.3f}s")
        print(f"Epoch {epoch} time (event): {gpu_time / 1000:.3f}s")



## runs main function when script is called directly 
if __name__ == "__main__":
    main()
