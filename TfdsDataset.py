import tensorflow_datasets as tfds
import torch
from torch.utils.data import Dataset, DataLoader


# Custom Dataset that wraps a tfds dataset
class TfdsDataset(Dataset):
    def __init__(self, tfds_dataset, transform=None):
        # Convert the tfds dataset to a list so we can index it
        self.data = list(tfds_dataset)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image, label = self.data[idx]
        # If image is a tf.Tensor, convert it to a numpy array
        if hasattr(image, 'numpy'):
            image = image.numpy()
        if hasattr(label, 'numpy'):
            label = label.numpy()  # convert tf.Tensor to numpy
        # Ensure the image shape is what transforms.ToTensor expects (H x W x C).
        # For grayscale images, you might have shape (28,28,1); that's fine.
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

