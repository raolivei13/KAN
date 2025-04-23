# import tensorflow as tf
# print("TensorFlow version:", tf.__version__)


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
# import tensorflow as tf
# import pickle

#from tfds_data_set_wrapper import TfdsDataset
# import tensorflow_datasets as tfds

#from efficient_kan.kan import KAN # importing KAN class
from KANLinear import KAN
from data_loader import get_dataloaders
from make_data_set import PairedMNISTTFDSDataset
# from TfdsDataset import TfdsDataset

# get data

train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = get_dataloaders(batch_size=5, shuffle=False, train_ratio=0.7, val_ratio=0.2)

# save the test data set
test_dataset_path = os.path.join(os.path.dirname(__file__), "test_dataset.pth")
torch.save(test_dataset, test_dataset_path)

# for normalization
# transform = transforms.Compose([
#     transforms.ToTensor(),  # Converts a numpy array (H x W x C) to a tensor (C x H x W) in [0,1]
#     transforms.Normalize((0.5,), (0.5,))
# ])


# make PyTorch dataset
# full_dataset = PairedMNISTTFDSDataset(root='./data', train = True, original_transform=transform, corrupted_transform=transform)
#
# full_len = len(full_dataset)
# train_len = int(0.8 * full_len)
# val_len = full_len - train_len
#
# train_dataset, val_dataset = random_split(full_dataset, [train_len, val_len]) # split data set

# val_dataset = PairedMNISTTFDSDataset(root='./data', subset="val", original_transform=transform)
# test_dataset  = PairedMNISTTFDSDataset(root='./data', train = False, original_transform=transform, corrupted_transform=transform) # for testing
#
#
# save the test data set as pt
# test_data_pairs = [test_dataset[i] for i in tqdm(range(len(test_dataset)), desc="Precomputing test dataset")]
# torch.save(test_data_pairs, "test_dataset.pt")

# with open("test_dataset.pkl", "wb") as f:
#     pickle.dump(test_dataset, f)


# Create DataLoaders for training and validation
# trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# valloader   = DataLoader(val_dataset, batch_size=64, shuffle=False)


# Hyperparameters
learning_rate = 1e-3
num_scalar_fields = 6 # including velocity in the inputs

# save number of hidden layers, number of scalar fields and save number of hidden channels
with open("num_scalar_fields.txt", "w") as f:
    f.write(str(num_scalar_fields))


# define KAN model
print("Instantiating the model")
model = KAN([num_scalar_fields * 15532, 10000, 5000, 10000, 15532])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)


optimizer = optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4) # define optimizer
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) # change learning


# number of epochs
epochs = 5

# Define loss
criterion = nn.MSELoss() # regression
#criterion = CombinedMSESSIMLoss(mse_weight=0.9, ssim_weight=0.0, epi_weight=0.1, psnr_weight=0.0)


for epoch in range(epochs):


    # training loop
    epoch_loss = 0
    model.train()

    # enter batch loop

    for batch_idx, (inputs, targets) in enumerate(train_loader):

        inputs = inputs.unsqueeze(2)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.squeeze(1).squeeze(1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        # ---------------- VALIDATION pass --------------- #
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for val_batch_idx, (val_inputs, val_targets) in enumerate(val_loader):
                val_inputs = val_inputs.unsqueeze(2)
                preds = model(val_inputs)  # make prediction
                preds = preds.squeeze(1).squeeze(1)
                val_loss += criterion(preds, val_targets)
                # Accumulate the loss, weighted by the batch size
                # total_val_loss += loss_val.item() * val_inputs.size(0)

        # # ---------------- VALIDATION pass --------------- #
        #
        avg_loss = epoch_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
    # val_accuracy /= len(valloader)

    # Update learning rate
    scheduler.step()

    print(
        f"Epoch {epoch + 1}, Epoch Loss: {avg_loss}, Val Loss: {avg_val_loss}"
    )


# save model for testing
model_path = 'kan_model_trained.pth'
torch.save(model.state_dict(), model_path)

