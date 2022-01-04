#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 22:52:11 2020

@author: ychen413
"""

import numpy as np
import matplotlib.pyplot as plt

from config import DefaultConfig

def data_normalize(data):
    return (data - data.mean()) / data.std()

opt = DefaultConfig()

root = opt.out_dir
#es = np.loadtxt(root + '/epoch300_best1.dat')
es = np.loadtxt(root + '/epoch800_best1.dat')

#root = opt.data_root
#es = np.loadtxt(root + '/04/286.dat')
#es = data_normalize(es)

x = np.linspace(1e-4, np.pi - 1e-4, 64)
y = np.linspace(   0,      2*np.pi, 64)

plt.pcolormesh(x, y, es)
#plt.contour(x, y, es)


