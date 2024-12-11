#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 30 11:24:20 2024
Objective:
    - Compute the accuracy of the segmentation model:
        - by calculating the dice_loss
        - by calculating IntersectionOverunion 
        - by caculaatign the average difference between the centroids interms of pixels
        - by calculating the average difference in the area of the bounding rectangles
        - Crop out the region of interest 
@author: dagi
"""
import os
import torch
import torchvision
from torch.utils.data import DataLoader 
import numpy as np 
import cv2 

# In[3] 
"""
Objectives
- Creates training_dataset
- Creates training_dataloader
- Creates validation_dataset
- Creates validation_dataloader
"""
def get_loaders(train_ds, 
                val_ds,
                batch_size
                ):
    # Create Train & validation dataset
    # train_ds = CarvanaDataset(train_dir, train_maskdir, train_transform )
    # val_ds = CarvanaDataset(val_dir, val_maskdir, val_transform)
    
    # Create Training and validation DataLoader
    train_dataloader = DataLoader(dataset = train_ds,
                                  batch_size = batch_size,
                                  shuffle = True,
                                  num_workers = 4,
                                  pin_memory = True)
    val_dataloader = DataLoader(dataset = val_ds,
                                batch_size = batch_size,
                                shuffle = False,
                                num_workers = 4,
                                pin_memory = True
                                )
    return train_dataloader, val_dataloader 

# In[5]
def save_predictions_as_imgs(loader, model, folder ="saved_images/", device="cuda"):
    model.eval()
    save_path = "/media/dagi/Linux/Mallie_Dagmawi/PyTorch/data/Brain MRI Segmentation/" + folder
    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
        torchvision.utils.save_image(preds, f"{save_path}/pred_{idx}.png")
        torchvision.utils.save_image(y, f"{save_path}{idx}.png")
    model.train()

# In[] Get the coordinates & centroid of the bounding rectangle
def bounding_rectangle(orig_mask, seg_mask):
    # Convert to torch tensor datatype
    orig_mask = torch.tensor(orig_mask)
    seg_mask = torch.tensor(seg_mask)
    # extract only the nonzero values 
    orig_nonzero = torch.nonzero(orig_mask)
    seg_nonzero = torch.nonzero(seg_mask)
    # if the nonzero is empty then we assign (0,0) (0, 0) to the rectangle
    if len(orig_nonzero) == 0:
        orig_top_left = (0, 0)
        orig_bottom_right = (0, 0)
    # Else get the coordinates for the top_left and bottom_right corners
    else:
        orig_top_left  = torch.min(orig_nonzero, dim=0)[0]
        orig_bottom_right = torch.max(orig_nonzero, dim=0)[0]
    
    if len(seg_nonzero) == 0:
        seg_top_left = (0, 0)
        seg_bottom_right = (0, 0)
    else:
        seg_top_left = torch.min(seg_nonzero, dim=0)[0]
        seg_bottom_right = torch.max(seg_nonzero, dim=0)[0]
    # Calculate the center points of the two rectangles
    # top_left = (top_most, left_most)
    # bottom_right = (bottom_most, right_most)
    # center_x = left_most + (right_most - left_most)/2
    # center_y = top_most + (bottom_most - top_most)/2
    # center = (center_y, center_x)
    # center_orig = (y1, x1), center_seg = (y2, x2)
    center_x1 = int(orig_top_left[1] + (orig_bottom_right[1] - orig_top_left[1])/2)
    center_y1 = int(orig_top_left[0] + (orig_bottom_right[0] - orig_top_left[0])/2)
    # center for the segmented box
    center_x2 = int(seg_top_left[1] + (seg_bottom_right[1] - seg_top_left[1])/2)
    center_y2 = int(seg_top_left[0] + (seg_bottom_right[0] - seg_top_left[0])/2)
 #   print(f"orig_top_left: {orig_top_left}")
    orig_top_left = orig_top_left
    orig_bottom_right = orig_bottom_right
    # collect the bounding corder of the rectangle
    orig_coord = (orig_top_left, orig_bottom_right)
    seg_coord = (seg_top_left, seg_bottom_right)
    # calculate the centroid of the rectangle
    center_x_orig = (orig_bottom_right[1] - orig_top_left[1])/2
    center_y_orig = (orig_bottom_right[0] - orig_top_left[0])/2
    orig_center = (center_y1, center_x1)
    
    seg_center = (center_y2, center_x2)
    #print(f"type(orig_coord): {type(orig_coord)}")
    # return the values
    return orig_coord, orig_center, seg_coord, seg_center

# In[] Calculates the area of teh rectangle
def get_area(orig_coord, seg_coord):
    # unpack the orig_coord & seg_coord
    orig_top_left = orig_coord[0]
    orig_bottom_right = orig_coord[1]
    
    seg_top_left = seg_coord[0]
    seg_bottom_right = seg_coord[1]
    # calculate the width of both rectangles
    orig_width = orig_bottom_right[1] - orig_top_left[1]
    seg_width = seg_bottom_right[1] - seg_top_left[1]
    # claculate the height of both rectangles
    orig_height = orig_bottom_right[0] - orig_top_left[0]
    seg_height = seg_bottom_right[0] - seg_top_left[0]

    # calculate the area
    orig_area = orig_height * orig_width
    seg_area = seg_height * seg_width
    
    # calculate the ratio of the two areas
    if  seg_area == 0:
        return 0
    if orig_area > seg_area:
        return np.round((seg_area/orig_area)*100, 3)
    else:
        return np.round((orig_area/seg_area)*100, 3)


# In[] Calculates distance between the centroids
def get_dist_centroid(orig_center, seg_center):
    # calcualte the a in the pythagores theorm a² + b² = c²
    a = abs(orig_center[0] - seg_center[0])
    # calcualte the b in the pythagores theorm a² + b² = c²
    b = abs(orig_center[1] - seg_center[1])
    # calculate the c which is the distance b/n the centroids
    c = np.sqrt(a**2 + b**2)
    # return the distance
    return np.round(c, 3)

# In[] Pixel wise comparison 
def pixelwise_comparison(orig_mask, seg_mask):
    # Convert to torch tensor
    orig_mask = torch.tensor(orig_mask)
    seg_mask = torch.tensor(seg_mask)
    # select only nonzero values
    orig_nonzero = torch.nonzero(orig_mask)
    seg_nonzero = torch.nonzero(seg_mask)
    # calculate the percentile difference between the two binary file
    if len(seg_nonzero) == 0:
        return 0
    if len(seg_nonzero) > len(orig_nonzero):
        return np.round((len(orig_nonzero)/len(seg_nonzero))*100, 3)
    else:
        return np.round((len(seg_nonzero)/len(orig_nonzero))*100, 3)

# In[] Dice Loss: calculates the dice loss between two binary masks
def dice_loss(orig_mask, seg_mask, epsilon = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.float32)/255.
    seg_mask = seg_mask.astype(np.float32)/255.
    # Compute the intersection that is the sum of element wise multiplication
    intersection = np.sum(orig_mask *  seg_mask)
    # Compute the union between the two masks that is sum of element wise addition
    union = np.sum(orig_mask) + np.sum(seg_mask) + epsilon
    # Compute the dice coefficient
    dice_coef = (2 * intersection)/union
    # Calculate the dice loss
    dice_loss = 1 - dice_coef
    # return the dice loss
    return np.round(dice_loss, 3)

# In[] IntersectionOverUnion IoU
def intersection_over_union(orig_mask, seg_mask, eps = 1e-6):
    # convert to float32 numpy array
    orig_mask = orig_mask.astype(np.uint8)/255.
    seg_mask = seg_mask.astype(np.uint8)/255.
        
    # Calculate the intersection (logical AND)
    intersection = np.logical_and(orig_mask, seg_mask)
    
    # Calculate the union (logical OR)
    union = np.logical_or(orig_mask, seg_mask)
    
    # Sum the intersection and union
    intersection_sum = np.sum(intersection)
    union_sum = np.sum(union)
    
    # Compute the IoU and subtract from 1 to make it a loss
    iou = intersection_sum /(union_sum + eps)
    # return the loss
    return np.round(1 - iou, 3)    

# In[] BCE  adn Dice Loss
def bce_dice_loss(pred, target, bce_weight=0.5, smooth=1e-6):
    # Ensure predictions are in the range [0, 1]
    pred = np.clip(pred, smooth, 1 - smooth)

    # 1. Binary Cross-Entropy (BCE) Loss
    bce_loss = -np.mean(target * np.log(pred) + (1 - target) * np.log(1 - pred))

    # 2. Dice Loss
    intersection = np.sum(pred * target)
    dice_loss = 1 - (2. * intersection + smooth) / (np.sum(pred) + np.sum(target) + smooth)

    # 3. Combined BCE + Dice Loss
    bce_dice_loss = bce_weight * bce_loss + (1 - bce_weight) * dice_loss

    return bce_dice_loss

# In[] Returns Region of Interest the coordinates of top corner, width and height
def regionOfInterest_2(mask):
    # Convert to torch tensor datatype
    mask = torch.tensor(mask)
    # extract only the nonzero values 
    nonzero = torch.nonzero(mask)
    # if the nonzero is empty then we assign (0,0) (0, 0) to the rectangle
    if len(nonzero) == 0:
        top_left = (0, 0)
        bottom_right = (0, 0)
    # Else get the coordinates for the top_left and bottom_right corners
    else:
        top_left  = torch.min(nonzero, dim=0)[0]
        bottom_right = torch.max(nonzero, dim=0)[0]
    # widen the area of interest by 60 pixels along all directions
    scale = 20
    new_top_left = [0, 0]
    new_top_left[0] = top_left[0] - scale # move corner to the left by 30 pixels
    new_top_left[1] = top_left[1] - scale # move corner to the top by 30 pixels
    
    # widen the area of interest by 30 pixels by moving the bottom right corner move to the right and bottom
    new_bottom_right = [mask.shape[0], mask.shape[1]]
    new_bottom_right[0] = bottom_right[0] + scale
    new_bottom_right[1] = bottom_right[1] + scale

    # make sure the new coordinates do not jump out of the picture
    # top left: convert back to int    
    new_top_left[0] = new_top_left[0].item() if new_top_left[0] > 0 else 0
    new_top_left[1] = new_top_left[1].item() if new_top_left[0] > 0 else 0
    # bottom right: convert back to int
    new_bottom_right[0] = new_bottom_right[0].item() if new_bottom_right[0] < mask.shape[0] else mask.shape[0]
    new_bottom_right[1] = new_bottom_right[1].item() if new_bottom_right[1] < mask.shape[1] else mask.shape[1]

    # Calculate the width
    width = new_bottom_right[0] - new_top_left[0]
    height = new_bottom_right[1] - new_top_left[1]
    
    top_corner = [new_top_left[0], new_top_left[1]]
    bottom_corner = [new_bottom_right[0], new_bottom_right[1]]
    
    return top_corner, bottom_corner

# In[] returns the coordinates of top_left corner and bottom_right corner of the bounding rectangle with padding
def regionOfInterest(mask, padding=10):
    """
    Returns the coordinates of the top-left and bottom-right corners of 
    the bounding rectangle around the non-zero region of the mask, 
    with optional padding.
    """
    # Ensure input is a NumPy array
    mask = np.asarray(mask)

    # Find non-zero coordinates
    nonzero = np.argwhere(mask)

    # If no non-zero values are present
    if nonzero.size == 0:
        return (0, 0), (0, 0)

    # Find top-left and bottom-right coordinates
    top_left = nonzero.min(axis=0)
    bottom_right = nonzero.max(axis=0)
    # print(f"top_left: {top_left}\t bottom_right: {bottom_right}")
    # Calculate the width and height
    padding_width = int((84 - (bottom_right[1] - top_left[1]))/2)
    padding_height = int((84 - (bottom_right[0] - top_left[0]))/2)
    # print(f"padding_width: {padding_width} \t padding_height: {padding_height}")
    # Apply padding and clamp values to image boundaries
    top_left[1] = np.maximum(top_left[1] - padding_width, 0)
    top_left[0] = np.maximum(top_left[0] - padding_height, 0)
    bottom_right[0] = np.minimum(bottom_right[0] + padding_height, mask.shape[0])
    bottom_right[1] = np.minimum(bottom_right[1] + padding_width, mask.shape[1])

    # print(f"top_left: {top_left}\t bottom_right: {bottom_right}")

    # Convert to tuple for consistency
    return tuple(top_left), tuple(bottom_right)

# In[] Calculate the mean and standard deviation for test dataset
def calculate_mean_std(path, height=None, width=None):
    # create list of all images with their full_path
    dataset_mri = []
    for dirPath, dirNames, fileNames in os.walk(path):
        for file in fileNames:
            if file.endswith('.jpg' ) or file.endswith('.tif'):
                full_path = dirPath + "/" + file
                dataset_mri.append(full_path)

    # set parameters based on which to process mean
    sum_img = None # accummulate the sum of pixel values of the entire dataset
    # height = 256 
    # width = 256
    print(f"height: {height} \t width: {width}\t len(dataset_mri): {len(dataset_mri)}")
    # Calculate the mean
    for img_name in dataset_mri:
        # Read the image in grayscale
        img = cv2.imread(img_name, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 256x256 for consistency
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)
        # accumulate teh sum of pixel values of each individual pixels
        if sum_img is None:
            sum_img = img / 255
        else:
            sum_img += img/255
    print(f"sum_img: {sum_img.shape}")
    #  calculating the mean
    mean_img = sum_img / len(dataset_mri)

    # Calculate  the mean value of pixels for each channel
    mean_pixel_value = np.mean(mean_img, axis=(0, 1))

    # set parameters for standard deviation
    sum_squared_img = None
    squared_diff = 0

    # calculate the standard deviation
    for img_path in dataset_mri:
        # Read the image as Grayscale
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        # Resize the image to 256x256
        img = cv2.resize(img, (height, width), interpolation=cv2.INTER_LINEAR)

        # Accumulate the squared differences from the mean image
        squared_diff = (img/255 - mean_img) ** 2
        if sum_squared_img is None:
            sum_squared_img = squared_diff
        else:
            sum_squared_img += squared_diff
    # Calculating the variance
    variance = sum_squared_img / len(dataset_mri)

    # Standard Deviation
    std = np.sqrt(np.mean(variance, axis = (0, 1)))

    # return the mean and standard deviation of the dataset
    return mean_pixel_value, std

# In[] returns the coordinates of top_left corner and bottom_right corner of the bounding rectangle with padding
def getTumorSize(mask):
    """
    Returns the coordinates of the top-left and bottom-right corners of 
    the bounding rectangle around the non-zero region of the mask, 
    with optional padding.
    """
    # Ensure input is a NumPy array
    mask = np.asarray(mask)

    # Find non-zero coordinates
    nonzero = np.argwhere(mask)

    # If no non-zero values are present
    if nonzero.size == 0:
        return (0, 0), (0, 0)

    # Find top-left and bottom-right coordinates
    top_left = nonzero.min(axis=0)
    bottom_right = nonzero.max(axis=0)

    # Calculate the width and height of the rectangle
    width = bottom_right[1] - top_left[1]
    height = bottom_right[0] - top_left[0]
    # Convert to tuple for consistency
    return width, height
