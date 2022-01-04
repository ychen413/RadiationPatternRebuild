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
    # '--dataRoot', required=True, help='path to dataset'
    # '--workers', type=int, default=2, help='number of data loading workers'
    # '--batchSize', type=int, default=64, help='input batch size'
    # '--imageSize', type=int, default=64, help='the height / width of the input image to network'
    # '--nz', type=int, default=100, help='size of the latent z vector'
    # '--ngf', type=int, default=64
    # '--ndf', type=int, default=64
    # '--niter', type=int, default=25, help='number of epochs to train for'
    # '--lr', type=float, default=0.0002, help='learning rate, default=0.0002'
    # '--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5'
    # '--cuda'  , action='store_true', help='enables cuda'
    # '--ngpu'  , type=int, default=1, help='number of GPUs to use'
    # '--netG', default='', help="path to netG (to continue training)"
    # '--netD', default='', help="path to netD (to continue training)"
    # '--outDir', default='.', help='folder to output images and model checkpoints'
    # '--model', type=int, default=1, help='1 for dcgan, 2 for illustrationGAN-like-GAN'
    # '--d_labelSmooth', type=float, default=0, help='for D, use soft label "1-labelSmooth" for real samples'
    # '--n_extra_layers_d', type=int, default=0, help='number of extra conv layers in D'
    # '--n_extra_layers_g', type=int, default=1, help='number of extra conv layers in G'
    # '--binary', action='store_true', help='z from bernoulli distribution, with prob=0.5'

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
    
    
    