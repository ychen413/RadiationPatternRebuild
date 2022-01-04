#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 03:26:05 2020

@author: ychen413
"""

import numpy as np
import os
import glob
#import time
#import random
#import argparse
import torch
import torch.nn as nn
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
#import torchvision.utils as vutils

### load project files
import models
from models import weights_init
from config import DefaultConfig

# Define Custom TensorDataset (able to use transforms())

#class CustomTensorDataset(torch.utils.data.Dataset):
#    """TensorDataset with support of transforms.
#    """
#    def __init__(self, tensors, transform=None):
#        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
#        self.tensors = tensors
#        self.transform = transform
#
#    def __getitem__(self, index):
#        x = self.tensors[0][index]
#
#        if self.transform:
#            x = self.transform(x)
#
#        y = self.tensors[1][index]
#
#        return x, y
#
#    def __len__(self):
#        return self.tensors[0].size(0)
    
class CustomTensorDataset(torch.utils.data.Dataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, root):
        self.dat_files = []
        folders = glob.glob(root + '*')
        for folder in folders:
            dat_files = glob.glob(folder + '/*.dat')
            self.dat_files.extend(dat_files)

    def __getitem__(self, index):
        dat_path = self.dat_files[index]
        label = 1
        
        data_ = np.loadtxt(dat_path)
        data_ = (data_ - data_.mean()) / data_.std()    # Normalize with mean and std
#        data_ = (data_ - data_.min()) / (data_.max() - data_.min())    # Normalize to [0 1]
        data = torch.from_numpy(data_[None, :, :])

        return data, label

    def __len__(self):
        return len(self.dat_files)

# Import numpy array (64, 64)
opt = DefaultConfig()
dataset = CustomTensorDataset(opt.data_root)


