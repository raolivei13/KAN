# from efficient_kan.kan import KAN
# Train on MNIST
import sys
import os
import matplotlib.pyplot as plt
import torch
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from new_loss import CombinedMSESSIMLoss
import tensorflow as tf
import pickle

#from tfds_data_set_wrapper import TfdsDataset
import tensorflow_datasets as tfds

#from efficient_kan.kan import KAN # importing KAN class
from KANLinear import KAN
from make_data_set import PairedMNISTTFDSDataset
from TfdsDataset import TfdsDataset

# for normalization
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts a numpy array (H x W x C) to a tensor (C x H x W) in [0,1]
    transforms.Normalize((0.5,), (0.5,))
])


# make PyTorch dataset
full_dataset = PairedMNISTTFDSDataset(root='./data', train = True, original_transform=transform, corrupted_transform=transform)

full_len = len(full_dataset)
train_len = int(0.8 * full_len)
val_len = full_len - train_len

train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len]) # split data set

# val_dataset = PairedMNISTTFDSDataset(root='./data', subset="val", original_transform=transform)
test_dataset  = PairedMNISTTFDSDataset(root='./data', train = False, original_transform=transform, corrupted_transform=transform) # for testing


# save the test data set as pt
test_data_pairs = [test_dataset[i] for i in tqdm(range(len(test_dataset)), desc="Precomputing test dataset")]
torch.save(test_data_pairs, "test_dataset.pt")

# with open("test_dataset.pkl", "wb") as f:
#     pickle.dump(test_dataset, f)


# Create DataLoaders for training and validation
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader   = DataLoader(val_dataset, batch_size=64, shuffle=False)


# define KAN model
model = KAN([28 * 28, 128, 64, 128, 28 * 28])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)


optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4) # define optimizer
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) # change learning


# number of epochs
epochs = 15

# Define loss
#criterion = nn.MSELoss() # regression
criterion = CombinedMSESSIMLoss(mse_weight=1.0, ssim_weight=0.0)


for epoch in range(epochs):


    # training loop
    model.train()

    # enter batch loop

    for img, img_corr, _ in trainloader:

        img = img.view(-1, 28 * 28).to(device)
        img_corr = img_corr.view(-1, 28 * 28).to(device)


        optimizer.zero_grad()
        output = model(img_corr) # model output
        loss = criterion(output, img)
        loss.backward()
        optimizer.step()



    model.eval() # switch model to eval
    val_loss = 0
    val_accuracy = 0
    with torch.no_grad():
        for img, img_corr, _ in valloader:


            img = img.view(-1, 28 * 28).to(device)
            img_corr = img_corr.view(-1, 28 * 28).to(device)
            output = model(img_corr)

            val_loss += criterion(output, img).item()


            # val_accuracy += (
            #     (output.argmax(dim=1) == labels.to(device)).float().mean().item()
            # )

    val_loss /= len(valloader)
    # val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Val Loss: {val_loss}"
    )


# save model for testing
model_path = 'kan_model_trained.pth'
torch.save(model.state_dict(), model_path)

