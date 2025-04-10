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


# class for forming (original_written_digit - corrupted_written_digit) pair
class PairedMNISTTFDSDataset(Dataset):
    def __init__(self, root, train=True, original_transform=None, corrupted_transform=None, indices=None):
        
        tfds_split = 'train' if train else 'test'

        
        self.original_dataset = list(
            tfds.load('mnist_corrupted/identity', split=tfds_split, as_supervised=True)
        )

        
        self.corrupted_data = list(
            tfds.load('mnist_corrupted/fog', split=tfds_split, as_supervised=True)
        )

        self.original_transform = original_transform
        self.corrupted_transform = corrupted_transform if corrupted_transform is not None else original_transform

        
        self.indices = indices

    def __len__(self):
        if self.indices is not None:
            return len(self.indices)
        return len(self.original_dataset)

    def __getitem__(self, idx):
        
        actual_idx = self.indices[idx] if self.indices is not None else idx

        
        original_img, original_label = self.original_dataset[actual_idx]
        if not isinstance(original_img, Image.Image):
            original_img = tf_tensor_to_pil(original_img)
      
        if not isinstance(original_label, int):
            original_label = int(original_label.numpy())
        if self.original_transform:
            original_img = self.original_transform(original_img)

        
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
