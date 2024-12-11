#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 10:48:33 2024
Objective:
    - Read the original mri which is 256x256
    - Cropout the tumor section which will not be larger than 84x84
    - Save the cropped image @ UNet_Refined dataset
    - Save the corresponding mask file @ UNet_Refined dataset
@author: dagi
"""
import os 
import cv2 
from utils import regionOfInterest
from tqdm import tqdm

# In[] Set path of the source dataset
src_train_path_img = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Train/Images/"
src_val_path_img = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Val/Images/"
src_test_path_img = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Images/"

src_train_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Train/Masks/"
src_val_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Val/Masks/"
src_test_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Masks/"

src_path_img = [src_train_path_img, src_val_path_img, src_test_path_img]
src_path_mask = [src_train_path_mask, src_val_path_mask, src_test_path_mask]
# In[] Create list objects from directory path
src_train_lst_img = os.listdir(src_train_path_img)
src_val_lst_img = os.listdir(src_val_path_img)
src_test_lst_img = os.listdir(src_test_path_img)

src_train_lst_mask = os.listdir(src_train_path_mask)
src_val_lst_mask = os.listdir(src_val_path_mask)
src_test_lst_mask = os.listdir(src_test_path_mask)

src_lst_img = [src_train_lst_img, src_val_lst_img, src_test_lst_img]
src_lst_mask = [src_train_lst_mask, src_val_lst_mask, src_test_lst_mask]

# In[] Set Destination pathes
dest_train_path_img = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Train/images/"
dest_val_path_img = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Val/images/"
dest_test_path_img = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Test/images/"

dest_train_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Train/masks/"
dest_val_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Val/masks/"
dest_test_path_mask = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Test/masks/"

dest_path_img = [dest_train_path_img, dest_val_path_img, dest_test_path_img]
dest_path_mask = [dest_train_path_mask, dest_val_path_mask, dest_test_path_mask]
 
# In[] Iterate through the lists
for index, lst in enumerate(src_lst_img) :
    if index == 0:
        stage = "Training"
    elif index == 1:
        stage = "Validation"
    else:
        stage = "Test"
    for image in tqdm(lst, desc=stage, leave=False):
        # get the full path
        full_path_img = src_path_img[index] + image
        full_path_mask = src_path_mask[index] + image.replace(".jpg", ".png")
        # Load both mri scan and its corresponding mask
        img = cv2.imread(full_path_img, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(full_path_mask, cv2.IMREAD_GRAYSCALE)
        # Get the region of interest
        top_corner, bottom_corner = regionOfInterest(mask)
        y1 = top_corner[1]
        y2 = bottom_corner[1] 
        x1 = top_corner[0]
        x2 = bottom_corner[0] 
        # crop the selected region from mri and mask
        crop_img = img[x1:x2, y1:y2]
        crop_mask = mask[x1:x2, y1:y2]
        # Resize the cropped region to be 84x84
        resize = (64, 64)
        img_resized = cv2.resize(crop_img, dsize=resize, interpolation=cv2.INTER_CUBIC)
        mask_resized = cv2.resize(crop_mask, dsize=resize, interpolation=cv2.INTER_CUBIC)
        # set the destination path 
        save_path_img = dest_path_img[index] + image
        save_path_mask = dest_path_mask[index] + image.replace(".jpg", ".png")
        
        # save the resized images in their respective folder
        cv2.imwrite(save_path_img, img_resized)
        cv2.imwrite(save_path_mask, mask_resized)
        
    
