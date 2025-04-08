import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tensorflow_datasets as tfds
import random


def tf_tensor_to_pil(image_tensor):
    image_np = image_tensor.numpy()
    if image_np.dtype != np.uint8:
        image_np = (255 * image_np).astype(np.uint8)
    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_np = image_np.squeeze(-1)
    pil_image = Image.fromarray(image_np)
    return pil_image


class PairedMNISTTFDSDataset(Dataset):
    def __init__(self, root, train=True, original_transform=None, corrupted_transform=None):
        # Load clean MNIST from torchvision.
        self.original_dataset = datasets.MNIST(root=root, train=train, download=True, transform=original_transform)

        # Load TFDS corrupted MNIST.
        tfds_split = 'train' if train else 'test'
        ds_corrupted = tfds.load('mnist_corrupted/glass_blur', split=tfds_split, as_supervised=True)
        self.corrupted_data = list(ds_corrupted)

        if corrupted_transform is None:
            self.corrupted_transform = original_transform
        else:
            self.corrupted_transform = corrupted_transform

        # Build a mapping from label to all indices for the corrupted dataset.
        self.corrupted_indices_by_label = {}
        for idx, (_, label) in enumerate(self.corrupted_data):
            label = int(label)
            self.corrupted_indices_by_label.setdefault(label, []).append(idx)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Get the original MNIST image and its label.
        original_img, original_label = self.original_dataset[idx]
        # Get list of indices in the corrupted dataset that have the same label.
        corrupted_indices = self.corrupted_indices_by_label.get(original_label)
        if not corrupted_indices:
            raise ValueError(f"No corrupted sample found for label {original_label}")
        # Select one corrupted image index (randomly or by any preferred strategy).
        corrupted_idx = random.choice(corrupted_indices)
        corrupted_img_tf, corrupted_label = self.corrupted_data[corrupted_idx]

        # Optional: you can check that int(corrupted_label) equals original_label.
        assert int(corrupted_label) == original_label, (
            f"Label mismatch after grouping: {original_label} vs {corrupted_label}"
        )

        # Convert the TFDS image (EagerTensor) to a PIL image.
        corrupted_img_pil = tf_tensor_to_pil(corrupted_img_tf)

        # Apply the transformation.
        if self.corrupted_transform:
            corrupted_img = self.corrupted_transform(corrupted_img_pil)
        else:
            corrupted_img = transforms.ToTensor()(corrupted_img_pil)

        return original_img, corrupted_img, original_label
