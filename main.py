# from efficient_kan.kan import KAN
# Train on MNIST
import sys
import os
# import certifi
# os.environ['SSL_CERT_FILE'] = certifi.where() #this is for the MNIST data set loader
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow_datasets as tfds
import tensorflow as tf

#from tfds_data_set_wrapper import TfdsDataset
import tensorflow_datasets as tfds

#from efficient_kan.kan import KAN # importing KAN class
from KANLinear import KAN
from TfdsDataset import TfdsDataset

# for normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a numpy array (H x W x C) to a tensor (C x H x W) in [0,1]
    transforms.Normalize((0.5,), (0.5,))
])

# current_dir = os.path.dirname(__file__)        # .../your_project/examples
# src_dir = os.path.join(current_dir, "..", "src")  # .../your_project/src
# sys.path.append(os.path.abspath(src_dir))

# Load MNIST-C Data Set
ds_corrupted_train = tfds.load('mnist_corrupted/glass_blur', split='train', as_supervised=True)
ds_corrupted_test  = tfds.load('mnist_corrupted/glass_blur', split='test', as_supervised=True)

# Wrap the tfds datasets in our custom PyTorch dataset
train_dataset = TfdsDataset(ds_corrupted_train, transform=transform)
test_dataset  = TfdsDataset(ds_corrupted_test, transform=transform)

# Create DataLoaders for training and testing
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader   = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Try loading a small portion of the dataset to see if it crashes.
ds = tfds.load('mnist_corrupted/glass_blur', split='train', as_supervised=True)
for image, label in ds.take(1):
    # Convert to numpy (this line might be triggering the issue)
    print(image.numpy().shape, label)


# Define model
model = KAN([28 * 28, 64, 10])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# Define optimizer
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Define learning rate scheduler
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)


# number of epochs
epochs = 10

# Define loss
criterion = nn.CrossEntropyLoss()
for epoch in range(epochs):
    # Train
    model.train()
    with tqdm(trainloader) as pbar:
        for i, (images, labels) in enumerate(pbar):
            images = images.view(-1, 28 * 28).to(device)
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels.to(device))
            loss.backward()
            optimizer.step()
            accuracy = (output.argmax(dim=1) == labels.to(device)).float().mean()
            pbar.set_postfix(loss=loss.item(), accuracy=accuracy.item(), lr=optimizer.param_groups[0]['lr'])

    # Validation
    model.eval()
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for images, labels in valloader:
            images = images.view(-1, 28 * 28).to(device)
            output = model(images)
            val_loss += criterion(output, labels.to(device)).item()
            val_accuracy += (
                (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            )
    val_loss /= len(valloader)
    val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}, Val Accuracy: {val_accuracy}"
    )