#!/usr/bin/env python3
# train.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-03-06

import torch
import torch.nn as nn
import torch.optim as optim
from config import LEARNING_RATE

def get_optimizer(model, world_size):
    return optim.Adam(model.parameters(), lr=LEARNING_RATE * world_size)

def get_loss_fn():
    return torch.nn.CrossEntropyLoss()


def train_epoch(model, optimizer, loss_fn, loader, device):
    
    model.train(True)
    
    total_loss = 0 
    total_samples = 0 
    for i, data in enumerate(loader):
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
        loss = loss_fn(outputs, labels)
        loss.backward()

        #Adjust learning weights
        optimizer.step()

        batch_size = inputs.size(0)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss, total_samples
        

