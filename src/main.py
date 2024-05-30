import torch
from datasets.dataset import SleepDataset
from datasets.utils import get_sleep_dataset
from monai.networks.nets import UNet

from train import Trainer


def main():
    """
    This function is the entry point for the application.
    """
    data_path = "/work/projects/heart_project/OSA_MW/all_files_ahi_sleep_complete_apneas/DATA"
    result_path = "results"
    split_folder = "/work/projects/heart_project/OSA_MW/splits/split1"
    channels = ["x2", "x3", "x4", "x5"]
    labels = ["y"]
    window_size = 2*60*60*64  # 2 hours of data
    transform = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 10

    print(data_path)
    print(split_folder)
    print(channels)
    print(labels)
    print(window_size)
    print(transform)
    print(device)
    print(num_epochs)

    train_dataset = get_sleep_dataset(
        data_path=data_path,
        split_folder=split_folder,
        split_name="train",
        channels=channels,
        labels=labels,
        window_size=window_size,
        transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True
    )

    val_dataset = get_sleep_dataset(
        data_path=data_path,
        split_folder=split_folder,
        split_name="val",
        channels=channels,
        labels=labels,
        window_size=window_size,
        transform=transform
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False
    )

    test_dataset = get_sleep_dataset(
        data_path=data_path,
        split_folder=split_folder,
        split_name="test",
        channels=channels,
        labels=labels,
        window_size=window_size,
        transform=transform
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False
    )

    print("Data loaders created successfully.")

    #  Create model
    model = UNet(
        spatial_dims=1,  # Assuming 1D data, change if you have 2D or 3D data
        in_channels=len(channels),  # Number of input channels
        out_channels=len(labels),  # Number of output channels
        channels=(
            16,
            32,
            64,
            128,
            256,
            256,
            256,
            256,
            256,
            256,
            256,
            256,
        ),
        strides=(2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2),
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    trainer = Trainer(model, optimizer, criterion, device, result_path)
    trainer.fit(train_loader, val_loader, num_epochs)


if __name__ == "__main__":
    main()
