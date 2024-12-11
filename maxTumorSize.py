#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 11:07:46 2024

@author: dagi
"""

import cv2
import os 
import numpy as np 
from utils import getTumorSize

# In[] set src path for the files
root_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/"
src_path_train = root_path + "Train/masks/"
src_path_val = root_path + "Val/masks/"
src_path_test = root_path + "Test/masks/"

src_lst_path = [src_path_train, src_path_val, src_path_test]

max_width = 0
max_height = 0

# In[] create lists from the file
for index, path in enumerate(src_lst_path):
    # create list from directory
    src_lst = os.listdir(path)
    # iterate through the list
    for mask_file in src_lst:
        # get the full path of the file
        full_path = src_lst_path[index] + mask_file 
        # load the image
        mask = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        # get the width and height of the tumor bounding rectangle
        width, height = getTumorSize(mask)
        # comapre the width and height with the maximum size recorded
        max_width = width if width > max_width else max_width 
        max_height = height if height > max_height else max_height
  
# max_height = 56 + 10% + 20 pixel for padding = 81.6 ~ 84 pixel
# max_width = 57 + 10% + 20 pixel for padding = 72.7 ~ 84 pixel
