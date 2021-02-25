# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
# Load Packages
# Basic packages/visualization
import os
import pandas as pd
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
import gc
import random

# Deep Learning Packages
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms


# Custom written .py files
os.chdir('C:\\Users\\zhengq\\Desktop\\ClubfootData\\Final_Seg_Maps_NIFTI2\\Code')
from helper import ground_truth_background, back_images

from Dataset_DL1 import BoneBackground
from Transforms_DL1 import toTensorMask, Resize_Imgs, Normalize
from Transforms_DL1 import RandomRotate, RandomFlip, ClaheFilter
from helper_DL1 import show_image_batch_model, show_single_image_model, save_image_batch_model, return_criterions
from models import FCNmodel_3pool, FCNmodel_2pool
from loss_functions import *
from training import *

#%%
# Helper Functions
##############################################################################
os.chdir("../Data")
base = os.getcwd()
print(base)
#%%
# from helper import ground_truth_background, back_images
data = ground_truth_background(base, None) # clubfoot_type = 'AP','LAT', None
print('Total # of radiographs inlcuded in this study: ', len(data))
# Example input image, background mask, and mask overlay on input image
back_images(data, 24)


# Splitting the data into the predetermined training and validation sets
split = pd.read_csv('Clubfoot_Update_012521.csv')

split['Folder ID'] = split['Folder ID'].astype(str)
data_2 = data.merge(split, left_on='folder', right_on='Folder ID', how='inner')
data_2 = data_2.drop(columns = ['Folder ID'])

training_data = data_2[data_2['Split'] == 'Training']
validation_data = data_2[data_2['Split'] == 'Validation']

# Clubfoot_Update_012521
print(' The training data size is: ', len(training_data),'\n',
      'The validation data size is: ', len(validation_data), '\n')

del data, data_2, split
gc.collect()
gc.collect();

#%%
# from helper import ground_truth_background, back_images

transformations = [toTensorMask(),
                   Resize_Imgs([272, 256]),
                   Normalize(), 
                   RandomRotate(0.5),
                   RandomFlip(0.5),
                   ClaheFilter()] 

training_dataset = BoneBackground(df = training_data,
                                    transform=transforms.Compose(transformations)
                                  )
training_dataloader = DataLoader(training_dataset, batch_size=8,
                        shuffle=True, num_workers=0)
#%%
# Helper function to show a batch
# Mover to Helper.py
from helper import show_single_image, show_image_batch
        
for i_batch, sample_batched in enumerate(training_dataloader):
    print(i_batch, sample_batched['image'].size(),sample_batched['mask'].size())    
    if i_batch == 0:
        show_single_image(sample_batched)
        break
    
#%%
# loaded from models.py
model = FCNmodel_3pool(n_class=1)
# move model to cuda/gpu device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)
#summary(model, input_size=(1, 272, 256), device = device.type) # device.type) 
# Trainable params: 6,041,188

for i_batch, sample_batched in enumerate(training_dataloader):
    print(i_batch, sample_batched['image'].size(),sample_batched['mask'].size())
    if i_batch == 0:
        mask = model(sample_batched['image'].to(device))
        plt.figure()
        show_single_image_model(sample_batched, torch.round(mask))
        break

#%%
# Simple function to calculate the dice coeffificent by an individual image in batch
def dice_by_image(sample_batched, output, data, i_batch):
    for j in range(sample_batched['mask'].shape[0]):
        gt = sample_batched['mask'][j,]
        model_mask = output[j,]
        
        for i in range(sample_batched['mask'].shape[1]):
            gt_bone = gt[i,]
            model_mask_bone = model_mask[i,]
            data.append([dice_coef(model_mask_bone, gt_bone).item(), i_batch, j])
         
    return data

#%%
# Random Grid Search
epochs = 100
base_dir = 'C:\\Users\\zhengq\\Desktop\\ClubfootData\\Final_Seg_Maps_NIFTI2\\Models'

i = 0
while i < 10:
    i += 1
    # Hyperparamters being tuned
    batch_group = [4, 8, 12, 16, 24]
    lr_group = [1e-2, 0.005, 1e-3]
    criteria_group = [DC_loss2(), BCELoss(), 'Combined']
    augmentations_group = ['Clahe_Aug', 'Aug', 'No_Aug']
    model_num_group = [2, 3]
    
    # Random Selection
    batch = random.choice(batch_group)
    lr = random.choice(lr_group)
    criterionn = random.choice(criteria_group)
    augments = random.choice(augmentations_group)
    model_num = random.choice(model_num_group)
    
    # Defining parameters based upon criterion
    criterion, criterion2, transformations, model = return_criterions(criterionn, augments, model_num) 
        
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)

    logpath = 'batch'+ str(batch) + '_lr' + str(lr) + '_augs' + str(augments) + '_model' + str(model_num) +'_Criterions' + str(criterionn) #str(criterion)[:-3] + '_' + str(criterion2)[:-2]        
    print(batch, lr, criterionn, augments, model_num, '\n')
    print(logpath)

    #################################################################################
    # Training Loop
    #################################################################################
    # Load the data
    validation_dataset = BoneBackground(df = validation_data,
                                         transform=transforms.Compose(transformations)
                                       )
    validation_dataloader = DataLoader(validation_dataset, batch_size = batch,
                            shuffle=False, num_workers=0)

    training_dataset = BoneBackground(df = training_data,
                                        transform=transforms.Compose(transformations)
                                      )
    training_dataloader = DataLoader(training_dataset, batch_size = batch,
                            shuffle=True, num_workers=0)

    # Creating a dictionary of paramters to simplify visualization
    params={
     "num_epochs": epochs, 
        "optimizer": optimizer,
     "criterion": criterion,
     "criterion2": criterion2,
     "train_dl": training_dataloader,
     "val_dl": validation_dataloader,
     "lr_scheduler": scheduler,
        "log_path": logpath,
    "base_dir": base_dir}

    # Training the model
    train_epoch_metrics, val_epoch_metrics = model_training(model, params)

    # Save loss function training and validation plots
    save_images(train_epoch_metrics, train_epoch_metrics, base_dir, logpath)

    # Reloading the Model with the Best Parameters
    if model_num == 2:
        model = FCNmodel_2pool(n_class=1)
    else:
        model = FCNmodel_3pool(n_class=1)

    optimizer_ = optim.Adam(model.parameters())
    checkpoint = torch.load(os.path.join(base_dir, logpath + '.best.pth.tar'))
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to('cpu')

    # Saving example images of the model output
    for i_batch, sample_batched in enumerate(validation_dataloader):
        print(i_batch, sample_batched['image'].size(),sample_batched['mask'].size())
        if i_batch == 0:
            mask = model(sample_batched['image'].to('cpu'))
            plt.figure()
            save_image_batch_model(sample_batched, torch.round(mask), base_dir, logpath)
            break      


    # Create the pandas DataFrame of Dice coefficients on a slice by slice basis for validation data
    mask_name = ['background', 'batch_num', 'j']
    data = []

    for i_batch, sample_batched in enumerate(validation_dataloader):
        mask = model(sample_batched['image'])
        data = dice_by_image(sample_batched, torch.round(mask), data, i_batch)

    dice_df = pd.DataFrame(data, columns = ['dice', 'batch', 'batch_num']) 
    dice_df.to_csv(logpath + '.csv')