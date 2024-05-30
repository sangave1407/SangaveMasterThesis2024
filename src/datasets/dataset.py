import os
import h5py
import random
import torch
import random
from torch.utils.data import Dataset


class SleepDataset(Dataset):
    def __init__(self, data_path, split_file, channels, labels, window_size, transform=None):
        self.hdf5_files = []
        with open(split_file, "r") as f:
            for row in f.readlines():
                self.hdf5_files.append(os.path.join(
                    data_path, row.strip() + ".hdf5"))
        self.transform = transform
        self.channels = channels
        self.labels = labels
        self.window_size = window_size

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, idx):
        patient_path = self.hdf5_files[idx]
        with h5py.File(patient_path, 'r') as file:
            # Collect the individual channels
            data = []
            for c in self.channels:
                data.append(torch.tensor(file[c][:]))
            # Combine the channels into a single tensor
            data = torch.stack(data, dim=0)

            # Collect the labels
            patient_labels = []
            for l in self.labels:
                patient_labels.append(torch.tensor(file[l][:]))
            # Combine the labels into a single tensor
            patient_labels = torch.stack(patient_labels, dim=0)

            # Randomly select a window of size self.window_size
            start = random.randint(0, data.shape[1] - self.window_size)
            data = data[:, start: start + self.window_size]
            patient_labels = patient_labels[:, start: start + self.window_size]

            # Remove unnecessary dimensions
            data = torch.squeeze(data, dim=2)
            patient_labels = torch.squeeze(patient_labels, dim=2)
            patient_labels = patient_labels.to(torch.float32)

            # Apply the transformation function
            if self.transform:
                try:
                    data = self.transform(data)
                except Exception as e:
                    print(
                        f"Error applying transform to sample at index {idx}: {e}")

            return data, patient_labels


if __name__ == "__main__":
    data_path = "data/hdf5"
    split_file = "data/splits/split1/train.txt"
    channels = ["x2", "x3"]
    labels = ["y", "sleep_label"]
    window_size = 10
    ds = SleepDataset(data_path, split_file, channels, labels,
                      window_size, transform=None)
    data, labels = ds.__getitem__(0)
    print(data.dtype, labels.dtype)
