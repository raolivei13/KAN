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
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf

#from tfds_data_set_wrapper import TfdsDataset
import tensorflow_datasets as tfds

#from efficient_kan.kan import KAN # importing KAN class
from KANLinear import KAN
from make_data_set import PairedMNISTTFDSDataset
from TfdsDataset import TfdsDataset
import pickle
import random

inv_normalize = transforms.Compose([
    transforms.Normalize(mean=[0.], std=[1/0.5]),  # undo std normalization
    transforms.Normalize(mean=[-0.5], std=[1.])
])


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate the KAN model with same params (can change this later)
# input dimension: 28 * 28
# compressed_dimension: 64
# output_dimension: 28 * 28
model = KAN([28 * 28, 128, 64, 128, 28 * 28])
model.to(device)

# load test data set
test_dataset = torch.load("test_dataset.pt")

# with open("test_dataset.pkl", "rb") as f:
#     test_dataset = pickle.load(f)

#test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False) # wrap the data loader for testing


# load the trained model
model_path = 'kan_model_trained.pth'
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)

# switch model to evaluation mode
model.eval()


# pick random image for testing
idx = random.randint(0, len(test_dataset) - 1)
original_img, corrupted_img, _ = test_dataset[idx]


input_tensor = corrupted_img.unsqueeze(0).view(-1, 28 * 28).to(device)

# Make a forward pass through the model.
with torch.no_grad():
    output = model(input_tensor)

predicted_img = output.view(28, 28).cpu()

# Undo normalization for display. (Assumes normalization: Normalize((0.5,), (0.5,)) )
# original_img_disp = inv_normalize(original_img.squeeze()).cpu().numpy()
# predicted_img_disp = inv_normalize(predicted_img).cpu().numpy()


plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.imshow(original_img.squeeze(0), cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(corrupted_img.squeeze(0), cmap='gray')
plt.title("Corrupted Image")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(predicted_img.squeeze(0), cmap='gray')
plt.title("Model Prediction")
plt.axis("off")

plt.show()



