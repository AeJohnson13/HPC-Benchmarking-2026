#!/usr/bin/env python3
# model.py
# Alex Johnson
# Started 2026-02-17
# Updated 2026-02-17

import torch
import torch.nn as nn
import torchvision
from torchvision.models import ResNet50_Weights
from torch.nn.parallel import DistributedDataParallel as DDP


def build_model(device, local_rank):
    model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    model = DDP(model, device_ids=[local_rank])
    return model
