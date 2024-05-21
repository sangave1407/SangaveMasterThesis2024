#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 20 16:54:01 2024

@author: sangavethanigaivelan

# 20 may 2024
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from monai.data import Dataset, DataLoader
from monai.networks.nets import UNet
from PIL import Image
from datasets import load_dataset
from torchvision.transforms import v2
import h5py
import random
from torch.utils.data import Sampler
from monai.networks.nets import UNet
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

#%% EDA

os.getcwd()
os.chdir('/Users/sangavethanigaivelan/Desktop/PythonScripts/data')
f_mesa_sleep_21 = h5py.File('mesa-sleep-0021.hdf5', 'r')

list(f_mesa_sleep_21)

#['sleep_label', 'time', 'x2', 'x3', 'x4', 'x5', 'y']

data_sleep_label = f_mesa_sleep_21['sleep_label'][:]

plt.plot(data_sleep_label)
plt.show()
#%%

def extract_data(filename):
    data_h5py=h5py.File(filename, 'r')
    data_sleep_label = data_h5py['sleep_label'][:].reshape(-1)
    data_time= data_h5py['time'][:].reshape(-1)
    data_x2= data_h5py['x2'][:]
    data_x3= data_h5py['x3'][:]
    data_x4= data_h5py['x4'][:]
    data_x5= data_h5py['x5'][:]
    data_y= data_h5py['y'][:]
    
    title_lab= str("Time vs x2 sleep_label")+str(filename)
    plt.plot(data_time,data_sleep_label,'red')
    #plt.plot(data_time,data_x2)
    #plt.plot(data_time,data_x3)
    plt.title(title_lab)
    plt.show()
    
    plt.plot(data_time,data_x4)
    title_lab= str("Time vs x2 analysis")+str(filename)
    plt.title(title_lab)
    plt.show()
    
    plt.plot(data_time,data_y)
    title_lab= str("Time vs y")+str(filename)
    plt.title(title_lab)
    plt.show()
    
    dataFrame_d= pd.DataFrame([data_sleep_label,data_time])
    
    return data_sleep_label,data_time,data_x2,data_x3,data_x4,data_x5,data_y,dataFrame_d

mydata=extract_data('mesa-sleep-0021.hdf5')

mydata=extract_data('shhs1-201306.hdf5')

mydata=extract_data('mros-visit1-aa1263.hdf5')

mydata=extract_data('shhs2-204473.hdf5')


mydata_df=extract_data('shhs2-204473.hdf5')[7]




#mros-visit1-aa1263.hdf5


#%% Pre data transform

import os
import h5py
import torch
import random
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet


#DATA_PATH = os.path.join(".", "data")

class SleepDataset(Dataset):
    def __init__(self, split_file, channels, window_size, transform=None):
        self.hdf5_files = []
        with open(split_file, "r") as f:
            for row in f.readlines():
                #self.hdf5_files.append(os.path.join(DATA_PATH, row.strip()))
                self.hdf5_files.append(row.strip())
        self.transform = transform
        self.channels = channels
        self.window_size = window_size 

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, idx):
        patient_path = self.hdf5_files[idx]
        with h5py.File(patient_path, 'r') as file:
            print("##**** ->",patient_path)
            channels = torch.tensor(file[self.channels][:])
            # C is the number of channels and T is the number of time steps
            start = random.randint(0, channels.shape[1] - self.window_size)
            #channels = channels[:, start: start + self.window_size]
            channels = channels[:]

            if self.transform:
                try:
                    channels = self.transform(channels)
                except Exception as e:
                    print(f"Error applying transform to sample at index {idx}: {e}")

            return channels

# Define a transformation function
def custom_transform(data):
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    transformed_data = normalize(data)
    return transformed_data

# Initialize the dataset with the transformation function
transformed_dataset = SleepDataset(
    split_file="/Users/sangavethanigaivelan/Desktop/PythonScripts/text-split.txt",
    channels=["x2"],
    window_size=200,  # 2 * 60 * 60 * 64,
    transform=custom_transform
)

# Define a DataLoader to load data batches from the dataset
data_loader = DataLoader(transformed_dataset, batch_size=8, shuffle=True)



#%%   U net implementation - Monai

# Define the UNet architecture
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet = UNet(
    spatial_dims=1,  # Assuming 1D data, change if you have 2D or 3D data
    in_channels=1,  # Number of input channels
    out_channels=1,  # Number of output channels
    channels=[16, 32, 64, 128, 256],  # Number of channels in each layer
    strides=[2, 2, 2, 2]  # Strides for each layer
).to(device)

# Define loss function and optimizer
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(unet.parameters(), lr=1e-3)

# Training loop
num_epochs = 10  # Define the number of epochs
for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch in data_loader:
        inputs = batch.to(device)
        
        # Forward pass
        outputs = unet(inputs)
        
        # Assuming to have targets (ground truth) for your regression task
        targets = torch.randn_like(outputs)  # Example random targets
        
        # Compute loss
        loss = criterion(outputs, targets)
        epoch_loss += loss.item() * inputs.size(0)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Print epoch loss
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(transformed_dataset)}")



#%%

#%% blocktest

patient_path="/Users/sangavethanigaivelan/Desktop/PythonScripts/data/mesa-sleep-0021.hdf5"

with h5py.File(patient_path, 'r') as file:
    print("##**** ->",patient_path)
    channels = torch.tensor(file["x2"][:])
    print("returning channels",channels)

