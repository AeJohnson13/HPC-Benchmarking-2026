# based heavily on code from https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/

import torch
import torchvision
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import ResNet50_Weights
from torch.utils.data import Subset
import time

# set params
num_epochs = 100
batch_size = 64
learning_rate = 0.001


#checks if cuda is available and uses it if it is
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device} for inference')


# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),   # small random shifts
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465],
                         [0.2470, 0.2435, 0.2616]),
])

# pull training data
training_data = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
small_indices = torch.randperm(len(training_data))[:10000]
small_cifar10 = Subset(training_data, small_indices)

train_loader = torch.utils.data.DataLoader(small_cifar10,
                                         batch_size=batch_size, 
                                         shuffle=True, 
                                         num_workers=2)

model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.to(device)

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(num_epochs):

    start_time = time.time()  # Start timer
    for inputs, labels in train_loader:
        # Move input and label tensors to the device
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    end_time = time.time()  # End timer
    epoch_duration = end_time - start_time
    print(f'Epoch {epoch+1}/{num_epochs}, Time: {epoch_duration}')   # , Loss: {loss.item():.4f}, Time: {epoch_duration}'