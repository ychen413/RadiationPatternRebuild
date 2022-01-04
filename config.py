#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 03:20:18 2020

@author: ychen413
"""

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms


class DefaultConfig(object):
    data_root = '/home/ychen413/Research_USA/dataset/Spherical_scattering/Es_64array/'
#    data_root = '/home/ychen413/Research_USA/dataset/Spherical_scattering/'
    workers = 12
    batch_size = 128
    image_size = 64
    
    nc = 1
    nz = 100
    ngf = 64
    ndf = 64
    niter = 1001
    
    lr = 0.0005
    beta1 = 0.5
    
    cuda = True
    ngpu = 1
    
    net_g = ''
    net_d = ''
    
    out_dir = './results'
    out_num = 4
    
    model = 1
    d_label_smooth = 0.1
    n_extra_layers_d = 0
    n_extra_layers_g = 1
    
    binary = True
    
    
    
