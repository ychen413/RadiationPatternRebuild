#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 04:45:00 2020

@author: ychen413
"""

from __future__ import print_function
import os
import time
import random
import numpy as np
#import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
#from torch.autograd import Variable

### load project files
import models
from models import weights_init
from config import DefaultConfig
from numpyData_import import dataset

opt = DefaultConfig()

try:
    os.makedirs(opt.out_dir)
except OSError:
    pass

opt.manualSeed = random.randint(1, 10000) # fix seed, a scalar
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

# Replace if torch.cuda.is_available() then tensor.cuda()
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
nc = opt.nc
ngpu = opt.ngpu
nz = opt.nz
ngf = opt.ngf
ndf = opt.ndf
n_extra_d = opt.n_extra_layers_d
n_extra_g = opt.n_extra_layers_g

# For result plot
#xx_ = np.linspace(1e-4, np.pi - 1e-4, opt.image_size)
#yy_ = np.linspace(   0,    2 * np.pi, opt.image_size)


#dataset = dset.ImageFolder(
#    root=opt.data_root,
#    transform=transforms.Compose([
#            transforms.Resize(opt.image_size),
#            # transforms.CenterCrop(opt.imageSize),
#            transforms.ToTensor(),
#            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
#        ])
#)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batch_size,
                                         shuffle=True, num_workers=opt.workers)

# load models 
if opt.model == 1:
    netG = models._netG_1(ngpu, nz, nc, ngf, n_extra_g)
    netD = models._netD_1(ngpu, nz, nc, ndf, n_extra_d)
elif opt.model == 2:
    netG = models._netG_2(ngpu, nz, nc, ngf)
    netD = models._netD_2(ngpu, nz, nc, ndf)

netG.apply(weights_init)
if opt.net_g != '':
    netG.load_state_dict(torch.load(opt.net_g))
print(netG)

netD.apply(weights_init)
if opt.net_d != '':
    netD.load_state_dict(torch.load(opt.net_d))
print(netD)

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

input = torch.FloatTensor(opt.batch_size, nc, opt.image_size, opt.image_size)
noise = torch.FloatTensor(opt.batch_size, nz, 1, 1)
if opt.binary:
    bernoulli_prob = torch.FloatTensor(opt.batch_size, nz, 1, 1).fill_(0.5)
    fixed_noise = torch.bernoulli(bernoulli_prob)
else:
    fixed_noise = torch.FloatTensor(opt.batch_size, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batch_size)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.to(device)
    netG.to(device)
    criterion.to(device)
    criterion_MSE.to(device)
    input, label = input.to(device), label.to(device)
    noise, fixed_noise = noise.to(device), fixed_noise.to(device)
    
#input = Variable(input)
#label = Variable(label)
#noise = Variable(noise)
#fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        start_iter = time.time()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
#        with torch.no_grad():
        input.resize_(real_cpu.size()).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label - opt.d_label_smooth) # use smooth label for discriminator

        output = netD(input)
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()
        # train with fake
        noise.resize_(batch_size, nz, 1, 1)
        if opt.binary:
            bernoulli_prob.resize_(noise.data.size())
            noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
        else:
            noise.data.normal_(0, 1)
        fake, z_prediction = netG(noise)
        label.data.fill_(fake_label)
        output = netD(fake.detach()) # add ".detach()" to avoid backprop through G
        errD_fake = criterion(output, label)
        errD_fake.backward() # gradients for fake/real will be accumulated
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step() # .step() can be called once the gradients are computed

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward(retain_graph=True) # True if backward through the graph for the second time
        if opt.model == 2: # with z predictor
            errG_z = criterion_MSE(z_prediction, noise)
            errG_z.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        
        end_iter = time.time()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data, errG.data, D_x, D_G_z1, D_G_z2, end_iter-start_iter))

#        if i % 100 == 0:
#            # the first 64 samples from the mini-batch are saved.
#            vutils.save_image(real_cpu[0:opt.out_num, :, :, :],
#                    '%s/real_samples.png' % opt.out_dir, nrow=4)
#            fake, _ = netG(fixed_noise)
#            score = netD(fake).detach()
#            index = score.topk(k=opt.out_num, dim=0)[1]
#            result = []
#            for ii in index:    
#                result.append(fake.detach()[ii])
#            vutils.save_image(torch.stack(result).squeeze(dim=1),
#                    '%s/fake_samples_epoch_%03d.png' % (opt.out_dir, epoch), nrow=4)
#
#            vutils.save_image(fake.data[0:opt.out_num, :, :, :],
#                    '%s/fake_samples_epoch_%03d.png' % (opt.out_dir, epoch), nrow=4)

    if epoch % 100 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.out_dir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.out_dir, epoch))
        
        fake, _ = netG(fixed_noise)
        score = netD(fake).detach()
        index = score.topk(k=opt.out_num, dim=0)[1]
        for i_, ii in enumerate(index):
            result = fake.detach().to(torch.device("cpu"))[ii[0], 0, :, :].numpy()
            np.savetxt('%s/epoch%d_best%d.dat' % (opt.out_dir, epoch, i_+1), result, fmt='%6.4e', delimiter='\t')
        
        
        
        