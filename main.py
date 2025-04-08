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
from torch.utils.data import DataLoader
from tqdm import tqdm
import tensorflow as tf

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

# Load MNIST-C Data Set for both training and testing
ds_corrupted_train = tfds.load('mnist_corrupted/glass_blur', split='train', as_supervised=True)
ds_corrupted_test  = tfds.load('mnist_corrupted/glass_blur', split='test', as_supervised=True)

# but also load the original not corrupted images from MNIST


# make PyTorch dataset
train_dataset = PairedMNISTTFDSDataset(root='./data', train=True, original_transform=transform)
test_dataset  = PairedMNISTTFDSDataset(root='./data', train=False, original_transform=transform)

# Create DataLoaders for training and validation
trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valloader   = DataLoader(test_dataset, batch_size=64, shuffle=False)


# for img, img_c in trainloader:
#
#     print(img.shape)
#     print(img_c.shape)
#
#     single_image = img[0]  # shape [1, 28, 28]
#
#     # Remove the channel dimension (squeeze if it's a single channel)
#     single_image = single_image.squeeze(0)  # Now shape is [28, 28]
#
#     # Convert tensor to numpy array for plotting
#     single_image = single_image.cpu().numpy()
#
#     # Optionally, reverse the normalization so pixel values are in the [0, 1] range
#     # Original transform: Normalize((0.5,), (0.5,)) --> x_norm = (x - 0.5) / 0.5
#     # To reverse: x = x_norm * 0.5 + 0.5
#     single_image = single_image * 0.5 + 0.5
#
#     # Plotting using matplotlib
#     plt.imshow(single_image, cmap='gray')
#     plt.title("Clean Image")
#     plt.axis('off')  # Hide the axis
#     plt.show()
#
#     break



# define KAN model
model = KAN([28 * 28, 64, 28 * 28])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model.to(device)


optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) # define optimizer
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8) # change learning


# number of epochs
epochs = 10

# Define loss
criterion = nn.MSELoss() # regression


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
model_info = {
    'input_size': input_dim,
    'hidden_size': hidden_dim,
    'output_size': n_classes,
    'num_hidden_layers': num_hidden_layers,
    'conv_channels': 64,
    'kernel_size': 3,
    'pool_size': 2,
    'dropout_rate': 0.5,
    'state_dict': model.state_dict()
}
# saving
file_path_for_saving = Path(__file__).resolve().parent.parent.parent / "files" / f"mlp_trained_{lamb}_{eta}.pth"
torch.save(model_info, file_path_for_saving)

