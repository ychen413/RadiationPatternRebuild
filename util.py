# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 12:17:06 2019

@author: deep362
"""

import os
import glob
import shutil
import re
from operator import itemgetter
#import img_process as imp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sl
#import pandas as pd
#from matplotlib import cm
#from PIL import Image, ImageOps
#import cv2

# ===============================================
# Compile, match 用法
#datepat = re.compile(r'(\d+).(\d+).(\d+)_(\d+)')
#m = datepat.match('10.16.2013_1412.PNG')
#m_  = datepat.match(file[0])
# ===============================================

def tuple_delete(tu_sample, ind_delete):
    tu_sample = tu_sample[:ind_delete] + tu_sample[ind_delete+1:]
    return tu_sample

def tuple_insert(tu_sample, ind_insert, tu_insert):
    # tu_insert must be tuple, too
    # For only one element, tu_insert has to be formed as (d1, )
    tu_sample = tu_sample[:ind_insert] + tu_insert + tu_sample[ind_insert:]
    return tu_sample

def tuple_modify(tu_sample, ind_modify, tu_modify):
    tu_sample = tuple_delete(tu_sample, ind_modify)
    tu_sample = tuple_insert(tu_sample, ind_modify, tu_modify)
    return tu_sample

def date_fillzero(tu_sample, ind_modify):
    # Fill '0' before any one digit number (eg, '8' -> '08')
    new_date = ('0' + tu_sample[ind_modify], )
    tu_sample = tuple_delete(tu_sample, ind_modify)
    tu_sample = tuple_insert(tu_sample, ind_modify, new_date)
    return tu_sample

def date_modify(date_list):
    # For this data pre-process only
    # if date value is only one digit, fill zero before it
    # eg, ('8', '7', '2009', '1314') -> ('08', '07', '2009', '1314')
    for line, a in enumerate(date_list):
        for ind in [0, 1]:
            if len(date_list[line][ind]) == 1:
                date_list[line] = date_fillzero(date_list[line], ind)
#    print(date_list)
    return date_list

def sort_by_date(doc_path):
    file = os.listdir(doc_path)
#    print(file)
    m = re.findall(r'(\d+).(\d+).(\d+)_(\d+)', str(file))    # Need format consistant
    m = date_modify(m)
    for ind, name in enumerate(m):
        m[ind] = name + (ind, )
    
    m.sort(key=itemgetter(2, 0, 1, 3))
    i = [d[-1] for d in m]
    file_ = []
    for line in i:
        file_.append(file[line])
#    print(file_)
    
    return file_

def rename(doc_path, new_name, start_num=1, img_type='.PNG'):
    # Rename the files due to numbering header
    # File name cannot initial as numbers when doing img processing
    file = sort_by_date(doc_path)
    i = start_num
    for line in file:
        if i<10:
            dst_ = new_name + '_' + str(0) + str(i) + img_type
        else:
            dst_ = new_name + '_' + str(i) + img_type
        src = doc_path + '/' + line
        dst = doc_path + '/' + dst_
        os.rename(src, dst)
        i+=1  
    pass

def file_move(source, distination, file_type='.PNG'):
    # Move specific type files from the source document to the distination
    files = os.listdir(source)
    for f in files:
        if f.endswith(file_type):
            shutil.move(os.path.join(source, f), os.path.join(distination, f))
    pass

def file_remove(file_dir, file_type='.PNG'):
    # Remove specific type files from the file_dir document
    files = os.listdir(file_dir)
    for f in files:
        if f.endswith(file_type):
            os.remove(os.path.join(file_dir, f))
    pass

def filter_gaussian1d(num_taps, sigma):
    x = np.arange(num_taps) - np.floor(num_taps / 2)
    g = np.exp(-x**2 / 2 / sigma**2) / np.sqrt(2 * np.pi) / sigma
    fg = g / g.sum()
    return fg

#loc = 'D:/E/Research_USA/dataset/Pwave_project_sample_EKGs_after_timeSeries/kk/'
#for doc in os.listdir(loc):
#    path = loc + doc
#    rename(path, doc)

#loc = 'D:/E/Research_USA/dataset/Pwave_project_sample_EKGs_after_timeSeries/kk/58/'
#imp.EKG_to_value_single(loc, cut=(570,700), window=(200,1000))

# =============
#loc = ['/home/ychen413/Research_USA/dataset/ECG_TimeSeries_Data/TS_II_normal/',
#       '/home/ychen413/Research_USA/dataset/ECG_TimeSeries_Data/TS_II_abnormal/']
#
#loc = '/home/ychen413/Research_USA/dataset/ECG_TimeSeries_Data/TS_II_abnormal/'
#for doc in glob.iglob(loc+'*'):
#    file_remove(doc+'/', file_type='.PNG')



