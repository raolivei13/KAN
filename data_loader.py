import os
import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class FlowDataset(Dataset):
    """
    A PyTorch Dataset that loads fluid flow data from all CSV files
    in a specified folder.

    For each CSV file:
        - Input (X): a stacked array from columns (p, T, U_mag, k, epsilon)
                     of shape (5, ~15000)
        - Target (y): the column 'nut' of shape (~15000,)
    """
    def __init__(self, folder_path, expected_len=15000):
        """
        Args:
            folder_path (str): Path to the folder containing the CSV files.
            expected_len (int): Expected length for each column.
        """
        # get all of the csv files
        all_csv_files = glob.glob(os.path.join(folder_path, "SAM-timeStep-*.csv"))
        self.samples = []

        for csv_file in all_csv_files:
            # read each csv file onto a data frame
            df = pd.read_csv(csv_file)

            #print(f"Processing file: {csv_file}, initial shape: {df.shape}")

            # detect if the CSV appears to be transposed.
            # ex: if the number of rows is less than expected (and columns are many), transpose.
            if df.shape[0] < expected_len and df.shape[1] >= expected_len:
                df = df.transpose()
                print(f"Transposed DataFrame shape: {df.shape}")

            # modify for extracting the params of interest
            required_columns = ['p', 'T', 'U_mag', 'k', 'epsilon', 'q', 'nut']
            for col in required_columns:
                if col not in df.columns:
                    raise KeyError(f"CSV file {csv_file} is missing required column '{col}'.")

            # convert each required column to a numpy array of type float32.
            p       = df['p'].values.astype(np.float32)
            T       = df['T'].values.astype(np.float32)
            U_mag   = df['U_mag'].values.astype(np.float32)
            k       = df['k'].values.astype(np.float32)
            epsilon = df['epsilon'].values.astype(np.float32)
            q = df['q'].values.astype(np.float32)
            nut     = df['nut'].values.astype(np.float32)

            # checks
            for name, col in zip(required_columns, [p, T, U_mag, k, epsilon, q, nut]):
                if col.size < expected_len:
                    raise ValueError(f"Column '{name}' in file {csv_file} has {col.size} values, expected at least {expected_len}.")


            X = np.stack([p, T, U_mag, k, epsilon, q], axis=0) # these are d scalar fields
            y = nut  # prediction to be made

            # uncomment for debugging purpose and to check shapes
            #print(f"File {csv_file} processed: X shape = {X.shape}, y shape = {y.shape}")
            self.samples.append((X, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        X, y = self.samples[idx]

        # torch object conversion
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return X_tensor, y_tensor


def get_dataloaders(batch_size, shuffle, train_ratio, val_ratio):
    """
    Create and return a training DataLoader and a testing DataLoader for the FlowDataset.

    Args:
        batch_size (int): How many samples per batch to load.
        shuffle (bool): Whether to shuffle data each epoch.
        train_ratio (float): Fraction of the data to be used for training.

    Returns:
        Tuple[DataLoader, DataLoader, Dataset, Dataset]:
            (train_loader, test_loader, train_dataset, test_dataset)
    """

    folder_path = os.path.join(os.path.dirname(__file__), "SAM_csv_files")
    dataset = FlowDataset(folder_path, expected_len=15000)

    # compute train-test splits
    total_samples = len(dataset)
    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # wrap to data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset
