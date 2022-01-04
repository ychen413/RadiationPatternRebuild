#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 02:28:09 2020

@author: ychen413
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
#import pandas as pd
from PIL import Image, ImageOps
import cv2
import glob
import os

#%% Function
def img_crop(img, x, y, w, h):
    return img[y:y+h, x:x+w]

def img_center_crop(img, width_new=None, height_new=None):

    w, h = img.size
    left = (w - width_new) / 2
    right = (w + width_new) / 2
    top = (h - height_new) / 2
    bottom = (h + height_new) / 2
    
    return img.crop((left, top, right, bottom))

def img_rename(document_loc, new_name, start_num=1, img_type='.PNG'):
    # Rename the img files due to numbering header
    # File name cannot initial as numbers when doing img processing
    i = start_num
    for line in os.listdir(document_loc):
        if i<10:
            dst_ = new_name + str(0) + str(i) + img_type
        else:
            dst_ = new_name + str(i) + img_type
        src = document_loc + '/' + line
        dst = document_loc + '/' + dst_
        os.rename(src, dst)
        i+=1  
    pass

def file_addname(document_loc, add_string, file_type='.dat'):
    # Add words in the end of the files' name
    
    for line in os.listdir(document_loc):
        if line[-4:] == file_type:
            dst_ = line[:-4] + add_string + line[-4:]
            src = document_loc + '/' + line
            dst = document_loc + '/' + dst_
            os.rename(src, dst)
    pass


def img_resized(imag, scale_x = 0.2, scale_y = 0.2):
    return cv2.resize(imag, (0,0), fx = scale_x, fy = scale_y)


def img_show(imag, method='Console', colormap=None):
    
    if method == 'Console':
        plt.figure(figsize=(10,6))
        plt.imshow(imag, cmap=colormap)
        plt.show()
    elif method == 'New':
        cv2.imshow('imag', imag)
        cv2.waitKey(0)
    pass

#%%
loc = ['/home/ychen413/Research_USA/dataset/Spherical_scattering/']

#for fold in glob.iglob(loc[0] + '*'):
#    for image in glob.iglob(fold + "/*.jpg"):
#        img_ = cv2.imread(image, cv2.IMREAD_UNCHANGED)
#        img_ = img_[50:850, 100:1100]
#        img = cv2.resize(img_, (960, 960))
#        cv2.imwrite(image, img)
        
for image in glob.iglob(loc[0] + '03/*.jpg'):
    img_ = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    img_ = img_[50:850, 100:1100]
    img = cv2.resize(img_, (960, 960))
    cv2.imwrite(image, img)
        
#img_dir = loc[0] + '02/1.jpg'

#for doc in glob.iglob('D:/E/Research_USA/dataset/ECG_original/ECG_normal/'):
#    for images in glob.iglob(doc + "*.png"):
#        img = EKGs_extract(images)
#        path = images.replace('Pwave_project_sample_EKGs', 'Pwave_project_sample_EKGs_after')
#        cv2.imwrite(path, img)
# =================================


