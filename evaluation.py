#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 06:43:50 2024
Objective:
    - Evaluate the performance output of the UNet model using evaluating metrices 
    like centeroid distance, area ratio, average BCEDice loss, averagae Dice loss and 
    average value of Intersection Over Union
@author: dagi
"""
import numpy as np
import cv2
import os 
from utils import (bounding_rectangle, get_area, get_dist_centroid, pixelwise_comparison, 
                    dice_loss, intersection_over_union, bce_dice_loss)
from loss_functions import DiceLoss, IoULoss, BCEDiceLoss

# In[] Set Route path to data directories
segmented_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/Test/unet_2/"
mask_path = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_256/Test/Masks/"

# In[] Create lists from the directories
segmented_list = os.listdir(segmented_path)
mask_list = os.listdir(mask_path)

segmented_list.sort()
mask_list.sort()

# In[] Set path to the output .txt files
unet_output_3 = "unet_2.txt"

# In[] Instantiation of list object to store the values of mask comparison
dice_loss_list = []
iou_list = []
area_ratio_list = []
centroid_dist_list = []
pixelwise_similarity_list = []
bceDiceLoss_list = []


# In[] Calculate the evaluation matrices
for mask in segmented_list:
    # abs_path_original = mask_path + mask.replace(".png", "_mask.tif")
    abs_path_original = mask_path + mask
    abs_path_segmented = segmented_path + mask
    # load the images
    img_segmented = cv2.imread(abs_path_segmented, cv2.IMREAD_GRAYSCALE)
    img_original = cv2.imread(abs_path_original, cv2.IMREAD_GRAYSCALE)
    # Calculate the dice loss
    dice_loss_list.append(dice_loss( img_segmented, img_original ))
    # Calculat e the loss on intersection over union
    iou_list.append(intersection_over_union(img_segmented, img_original))
    # calculate the loss using BCEDice Loss function
    bceDiceLoss_list.append(bce_dice_loss(img_segmented, img_original))
    # To calculate the centroid and corners of both original and masked rectangles
    orig_coord_n, orig_center_n, seg_coord_n, seg_center_n = bounding_rectangle(img_original, img_segmented)

    # Calculate the ratio in size of the original box against the segmented box
    area_ratio_list.append(get_area(orig_coord_n, seg_coord_n))
    
    # to calculate the centroid distance
    centroid_dist_list.append( get_dist_centroid(orig_center_n, seg_center_n))
    
    # Pixel wise comparison between the original mask and segmented mask
    pixelwise_similarity_list.append( pixelwise_comparison(img_original, img_segmented) )

# In[] Save the out put of yolo8n
with open(unet_output_3, "w") as f:
    # summary of the model output
    f.write(f'Average Dice Loss: {np.mean(dice_loss_list)}\n')
    f.write(f'Average IoU Loss: {np.mean(iou_list)} \n')
    f.write(f'Average Area Ratio: {np.mean(area_ratio_list)}\n')
    f.write(f'Average Distance Between the Centroids: {np.mean(centroid_dist_list)} \n')
    f.write(f'Average Similarity in Pixel Wise Comparison: {np.mean(pixelwise_similarity_list)} \n')
    # Output of Each mask file comparison
    f.write('###########################################\n')

f.close()     

