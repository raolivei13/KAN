import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import tensorflow_datasets as tfds
import random
import tensorflow as tf


def tf_tensor_to_pil(image_tensor):
    """
    Convert a TensorFlow EagerTensor to a PIL Image.
    """
    image_np = image_tensor.numpy()
    if image_np.dtype != np.uint8:
        image_np = (255 * image_np).astype(np.uint8)
    if image_np.ndim == 3 and image_np.shape[-1] == 1:
        image_np = image_np.squeeze(-1)
    pil_image = Image.fromarray(image_np)
    return pil_image


class PairedMNISTTFDSDataset(Dataset):
    def __init__(self, root, train=True, original_transform=None, corrupted_transform=None, indices=None):
        """
        Returns pairs of images: (original_image, corrupted_image, label)
        where the original_image is loaded from the TFDS 'mnist_corrupted/identity' dataset
        and the corrupted_image is loaded from the TFDS 'mnist_corrupted/dotted_line' dataset.
        The pairing is done by index.
        Args:
            root (str): Not directly used (kept for interface compatibility).
            train (bool): If True, use the 'train' split; else use 'test'.
            original_transform: Transform to apply to the original image.
            corrupted_transform: Transform to apply to the corrupted image.
                                 If None, original_transform is used.
            indices (list or None): Optional list of indices for subsetting.
        """
        # Choose the TFDS split.
        tfds_split = 'train' if train else 'test'

        # Load and convert the original (identity) dataset to a list.
        self.original_dataset = list(
            tfds.load('mnist_corrupted/identity', split=tfds_split, as_supervised=True)
        )

        # Load and convert the corrupted (dotted_line) dataset to a list.
        self.corrupted_data = list(
            tfds.load('mnist_corrupted/fog', split=tfds_split, as_supervised=True)
        )

        self.original_transform = original_transform
        self.corrupted_transform = corrupted_transform if corrupted_transform is not None else original_transform

        # Optional: allow subsetting using a list of indices.
        self.indices = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.original_dataset)

    def __getitem__(self, idx):
        # Map local index to an actual index if a subset was provided.
        actual_idx = self.indices[idx] if self.indices is not None else idx

        # Retrieve the original image and label.
        original_img, original_label = self.original_dataset[actual_idx]
        if not isinstance(original_img, Image.Image):
            original_img = tf_tensor_to_pil(original_img)
        # Convert original_label from Tensor to int.
        if not isinstance(original_label, int):
            original_label = int(original_label.numpy())
        if self.original_transform:
            original_img = self.original_transform(original_img)

        # Retrieve the corrupted image and label by the same index.
        corrupted_img_tf, corrupted_label = self.corrupted_data[actual_idx]
        if not isinstance(corrupted_label, int):
            corrupted_label = int(corrupted_label.numpy())
        if corrupted_label != original_label:
            raise ValueError(
                f"Label mismatch at index {actual_idx}: original label {original_label} vs corrupted label {corrupted_label}"
            )
        corrupted_img_pil = tf_tensor_to_pil(corrupted_img_tf)
        if self.corrupted_transform:
            corrupted_img = self.corrupted_transform(corrupted_img_pil)
        else:
            corrupted_img = transforms.ToTensor()(corrupted_img_pil)

        return original_img, corrupted_img, original_label
