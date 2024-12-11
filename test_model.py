#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  1 07:29:48 2024
Objective:
    - Using the test set the model segement out the tumor and save the mask 
    file in the Segmented directory
@author: dagi
"""
import os
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 
from torch.utils.data import DataLoader
from UNET_architecture import UNet
from unet_plus_plus import UNetPlusPlus
import cv2
from custom_dataset import UNetDataset
from tqdm import tqdm
import numpy as np 
from utils import calculate_mean_std, regionOfInterest

# In[] Routing path to the Source directory
src_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Images/"
src_path_2 = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/Test/images/"
mask_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Masks/"
src_list = os.listdir(src_path)

# In[] Destination folder
dest_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/Test/unet_2/"

# In[] Setting Hyperparameters
WIDTH = 256 
HEIGHT = 256 
WIDTH_2 = 64
HEIGHT_2 = 64
OUTPUT_SHAPE = 1
BATCH  = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# In[] Calculate teh mean and standard deviation of the dataset
mean, std = calculate_mean_std(src_path, HEIGHT, WIDTH)
mean_2, std_2 = calculate_mean_std(src_path_2, HEIGHT_2, WIDTH_2)

# In[] Set Transform Functions
transform_fn = A.Compose([
                        A.Resize(height=HEIGHT, width=WIDTH),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [mean],
                            std = [std],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])

transform_fn_2 = A.Compose([
                        A.Resize(height=HEIGHT_2, width=WIDTH_2),
                        # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                        A.Normalize(
                            mean = [mean_2],
                            std = [std_2],
                            max_pixel_value = 1.0),
                        ToTensorV2(),
                        ])
                 
# In[] Setting the dataset and dataloader
dataset = UNetDataset(src_path, mask_path, transform_fn)
data_loader = DataLoader(dataset = dataset,
                              batch_size = BATCH,
                              shuffle = True,
                              num_workers = 4,
                              pin_memory = True)

# In[] Load the model 
unet_1_path = "/home/dagi/Documents/PyTorch/MIP/Final/Task_4/best_model.pth"
unet_2_path = "best_model.pth"

model_1 = UNet(in_channels= 1, out_channels=OUTPUT_SHAPE)
model_2 = UNetPlusPlus(in_channels = 1,  num_classes = OUTPUT_SHAPE)
# load the saved dict
saved_state_dict_1 = torch.load(unet_1_path, weights_only=True)
saved_state_dict_2 = torch.load(unet_2_path, weights_only=True)

# load the state_dict into the model
model_1.load_state_dict(saved_state_dict_1)
model_2.load_state_dict(saved_state_dict_2)

model_1 = model_1.to(DEVICE)
model_2 = model_2.to(DEVICE)

# In[] Evaluation Loop
model_1.eval() # set the model_1 to evaluation mode
model_2.eval() # set the model_2 to evaluaation mode

for img_name in tqdm(src_list, desc="Testing", leave=False):
    # set the full path of the image
    full_path = src_path + img_name
    # set the destination path
    full_dest_path = dest_path + img_name.replace(".jpg", ".png")
    ##########################################################
    # load the image
    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    # convet img to numpy array and standardize the values
    img_standardized = np.array(img) / 255.0
    # Transform the image
    img_transformed = transform_fn(image = img_standardized)["image"].unsqueeze(0).to(DEVICE) # add batch dimension
    
    ##############################################################
    # Perform the evaluation task
    with torch.no_grad():
        # Forward pass
        predictions = torch.sigmoid(model_1(img_transformed))
        # convert probabilites to 0 or 1
        binary_predictions = (predictions > 0.5).float()
    ##############################################################
    # move the binary predictions to cpu and numpy
    mask = binary_predictions.squeeze(0).squeeze(0).cpu().detach().numpy()
    # convert mask to uint8 and values between 0 and 255
    mask = (mask * 255).astype(np.uint8)
    
    # If tumor is not detected save file mask and move to the next image
    non_zero = np.argwhere(mask)
    if len(non_zero) == 0:
        cv2.imwrite(full_dest_path, mask)
        continue
    ##############################################################
    # Get the region of interest
    top_corner, bottom_corner = regionOfInterest(mask)
    y1 = top_corner[1]
    y2 = bottom_corner[1] 
    x1 = top_corner[0]
    x2 = bottom_corner[0] 
    # crop the selected region mask
    crop_mask = mask[x1:x2, y1:y2]
    crop_img = img[x1:x2, y1:y2]

    # Resize the cropped region to be 84x84
    # mask_cropped = cv2.resize(crop_img, dsize=(HEIGHT_2, WIDTH_2), interpolation=cv2.INTER_CUBIC)
    # Standardize the cropped image
    cropped_standardized = crop_img/255.0
    # Transform the image
    cropped_transformed = transform_fn_2(image = cropped_standardized)["image"].unsqueeze(0).to(DEVICE)
    
    ##############################################################
    # Perform the evaluation task
    with torch.no_grad():
        # Forward pass
        cropped_predictions = torch.sigmoid(model_2(cropped_transformed))
        # convert probabilites to 0 or 1
        binary_cropped = (cropped_predictions > 0.5).float()
    
    ##############################################################
    # move the binary predictions to cpu and numpy
    cropped_mask = binary_cropped.squeeze(0).squeeze(0).cpu().detach().numpy()
    # convert mask to uint8 and values between 0 and 255
    cropped_mask = (cropped_mask * 255).astype(np.uint8)
    # Save the output in a 256x256 mask file
    mask_output = np.zeros((256, 256), dtype=np.uint8)
    mask_output[x1:x1+64, y1:y1+64] = cropped_mask
    # save the mask file
    cv2.imwrite(full_dest_path, mask_output)
        
