import torch
import pandas as pd
# import pyvista as pv
import os
import torch.optim as optim
import torch.nn as nn
import numpy as np
from data_loader import get_dataloaders
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from KANLinear import KAN
from torch.utils.data import DataLoader
import random


# ----------------- Load the spatial coordinate values ---------------------- #
centroids_path = os.path.join(os.path.dirname(__file__), "cell_centroids_Texas_a&m.csv")
if not os.path.exists(centroids_path):
    raise FileNotFoundError(f"Centroids CSV file not found: {centroids_path}")

centroids_df = pd.read_csv(centroids_path)
if not {'Centroid_X', 'Centroid_Z'}.issubset(centroids_df.columns):
    raise KeyError("The centroids CSV file must contain columns 'Centroid_X' and 'Centroid_Y'.")

x_centroids = centroids_df['Centroid_X'].values
y_centroids = centroids_df['Centroid_Z'].values
# ----------------- Load the spatial coordinate values ---------------------- #


# load the saved test dataset
test_dataset = torch.load(os.path.join(os.path.dirname(__file__), "test_dataset.pth"))
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False) # get a batch of size batch_size

# load model parameters
with open("num_scalar_fields.txt", "r") as f:
    num_scalar_fields = f.read() # load number of scalar fields

# convert to int
num_scalar_fields = int(num_scalar_fields)

model = KAN([num_scalar_fields * 15532, 10000, 5000, 10000, 15532])
model_path = os.path.join(os.path.dirname(__file__), "kan_model_trained.pth")
state_dict = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()


# forward pass no_grad()
with torch.no_grad():

    idx = random.randint(0, len(test_loader.dataset) - 1) # pick random data point from the test data set
    inputs, targets = test_loader.dataset[idx]
    inputs = inputs.unsqueeze(2)
    outputs = model(inputs)
    predictions = outputs.squeeze(1).squeeze(1)

    true_nut = targets.cpu().numpy()
    pred_nut = predictions[0].cpu().numpy()


true_nut_min, true_nut_max = true_nut.min(), true_nut.max()
pred_nut_min, pred_nut_max = pred_nut.min(), pred_nut.max()

# if (pred_nut_max - pred_nut_min) < 1e-12:
#     # If the prediction is (nearly) constant, map it to mid-range of GT.
#     pred_nut_rescaled = np.full_like(pred_nut, 0.5*(true_nut_min + true_nut_max))
# else:
# Standard min–max rescaling
pred_nut_rescaled = ((pred_nut - pred_nut_min) / (pred_nut_max - pred_nut_min)) \
                          * (true_nut_max - true_nut_min) + true_nut_min

# ------------------------------------------------
# OPTIONAL: If you notice that after rescaling, the color “lobes” appear inverted
# (i.e., high values where the ground truth is low and vice versa), you can
# flip (invert) them by doing:
#   pred_nut_rescaled = (true_nut_min + true_nut_max) - pred_nut_rescaled
#
# That way, wherever the model gave a "high" value becomes "low" and vice versa.
# Uncomment the line below if you suspect you need this “color inversion.”
# ------------------------------------------------

# UNCOMMENT IN CASE OF COLOR INVERSION
pred_nut_rescaled = (true_nut_min + true_nut_max) - pred_nut_rescaled

# ------------------------------------------------
# Check matching lengths


if len(true_nut) != len(x_centroids):
    raise ValueError("Mismatch between centroid count and nut vector length.")

# ------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(16, 8))

scatter_gt = axs[0].scatter(
    x_centroids, y_centroids, c=true_nut, cmap='coolwarm', s=30
)
axs[0].set_title("Ground Truth")
axs[0].set_xlabel("x")
axs[0].set_ylabel("y")
plt.colorbar(scatter_gt, ax=axs[0], label='nut value')

scatter_pred = axs[1].scatter(
    x_centroids, y_centroids, c=pred_nut_rescaled, cmap='coolwarm', s=30
)
axs[1].set_title("Network Prediction")
axs[1].set_xlabel("x")
axs[1].set_ylabel("y")
plt.colorbar(scatter_pred, ax=axs[1], label='nut value')

plt.suptitle("Ground Truth vs. Network Prediction")
plt.tight_layout()
plt.show()
