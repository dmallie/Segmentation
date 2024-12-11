#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 07:08:42 2024

@author: dagi
"""
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2 
import os
from loss_functions import  BCEDiceLoss
import torch.optim as optim
from unet_plus_plus import UNetPlusPlus
from utils import get_loaders, calculate_mean_std
from custom_dataset import UNetDataset
from model_loops import train_and_validate
from timeit import default_timer as timer

# In[0] Settin the Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2**4
EPOCHS = 200
# NUM_WORKERS = 2 
IMAGE_HEIGHT = 64 
IMAGE_WIDTH = 64
# PIN_MEMORY = True 
LOAD_MODEL = False 
ROOT_PATH = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_Refined/"
TRAIN_IMG_DIR = ROOT_PATH + "Train/image_enhanced/"
VAL_IMG_DIR = ROOT_PATH + "Val/image_enhanced/"
TRAIN_MASK_DIR = ROOT_PATH + "Train/masks/"
VAL_MASK_DIR = ROOT_PATH + "Val/masks/"

TRAIN_PATH = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Train/images/"
VAL_PATH = "/media/Linux/Mallie_Dagmawi/PyTorch/data/Dataset/UNet_64/Val/images/"
       
# In[2]
def main():
    # Get list of all image and mask paths
    # train_image_paths = [os.path.join(TRAIN_IMG_DIR, img) for img in os.listdir(TRAIN_IMG_DIR)]    
    # train_mask_paths = [os.path.join(TRAIN_MASK_DIR, mask) for mask in os.listdir(TRAIN_MASK_DIR)]
    
    # val_image_paths = [os.path.join(VAL_IMG_DIR, img) for img in os.listdir(VAL_IMG_DIR)]
    # val_mask_paths = [os.path.join(VAL_MASK_DIR, img) for img in os.listdir(VAL_MASK_DIR)]
    
    mean_train, std_train = calculate_mean_std(TRAIN_PATH, IMAGE_HEIGHT, IMAGE_WIDTH)
    mean_val, std_val = calculate_mean_std(VAL_PATH, IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f"mean_train: {mean_train}\t std_train: {std_train}")
    print(f"mean_val: {mean_val}\t std_val: {std_val}")
    # Set transform functions 
    train_transform = A.Compose([
                            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                            # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                            A.Rotate(limit=35, p=1.0),
                            A.HorizontalFlip(p=0.5),
                            A.VerticalFlip(p=0.1),
                            A.RandomBrightnessContrast(p=0.2),
                            A.ElasticTransform(p=0.2),
                            A.Normalize(
                                mean = [mean_train],
                                std = [std_train],
                                max_pixel_value= 1.0
                                ),
                            ToTensorV2(),
                            ])
    
    val_transform = A.Compose([
                            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
                            # A.ToGray(p=1.0), # p=1.0 ensures that the grayscale transform is always applied
                            A.Normalize(
                                mean = [mean_val],
                                std = [std_val],
                                max_pixel_value = 1.0),
                            ToTensorV2(),
                            ])
                        
    # Create dataset instances
    train_dataset = UNetDataset(TRAIN_PATH, TRAIN_MASK_DIR, train_transform)
    val_dataset = UNetDataset(VAL_PATH, VAL_MASK_DIR, val_transform)
    
    # create dataloader
    
    train_loader, val_loader = get_loaders(
                                    train_dataset,
                                    val_dataset,
                                    BATCH_SIZE )
    
    # initialize the model
    model = UNetPlusPlus(in_channels = 1,  num_classes = 1).to(DEVICE)

    loss_fn = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    # StepLR scheduler: decrease LR by a factor of 0.1 every 10 epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                 mode = 'min',
                                                 factor = 0.5,
                                                 patience = 40)

    BCEDiceLoss_list, dice_list, iou_list = train_and_validate(
                                                    model=model,
                                                    train_loader=train_loader,
                                                    val_loader=val_loader,
                                                    optimizer=optimizer,
                                                    criterion=loss_fn,
                                                    num_epochs=EPOCHS,
                                                    scheduler= scheduler,
                                                    save_path="best_model.pth"
                                            )   

    
    
    return BCEDiceLoss_list, dice_list, iou_list

# In[ ]
if __name__ == "__main__":
    torch.cuda.empty_cache()
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'  # Adjust the size as needed
    start_time = timer()
    BCEDiceLoss_list, dice_list, iou_list = main()
    end_time = timer()
    print(f"[INFO] Total training time: {end_time-start_time:.3f} seconds or {(end_time - start_time )/60:.2f} minutes")
