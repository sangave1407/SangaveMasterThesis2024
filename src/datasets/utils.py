import os
from datasets.dataset import SleepDataset


def get_sleep_dataset(data_path: str,
                      split_folder: str,
                      split_name: str,
                      channels: list,
                      labels: list,
                      window_size: int,
                      transform):
    return SleepDataset(
        data_path=data_path,
        split_file=os.path.join(split_folder, f"{split_name}.txt"),
        channels=channels,
        labels=labels,
        window_size=window_size,
        transform=transform
    )
