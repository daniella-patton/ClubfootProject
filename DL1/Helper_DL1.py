# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 11:18:45 2020

@author: danie
"""
# Loding in key packages for the helper functions
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
import numpy as np
import torch
from torch.nn import BCELoss
#from clubfoot_transforms import ClaheFilter, toTensorMask2, Resize_Imgs
#from clubfoot_transforms import Normalize, RandomRotate, RandomFlip, toTensorMask
#from bone_transformations import *

from Transforms_DL1 import toTensorMask, Resize_Imgs, Normalize
from Transforms_DL1 import RandomRotate, RandomFlip, ClaheFilter

from loss_functions import DC_loss2
from models import FCNmodel_2pool, FCNmodel_3pool
#import segmentation_models_pytorch as smp

# Creating Key functions
def ground_truth_files(base_dir, clubfoot_type):
    
    #Loop through labeled folders (Normal, Clubfoot, Hindfoot Valgus) to find all ground truth images of a 
    #specified radiograph type ('AP' or 'Lat') with a ground truth segmentation map and background
    
    folder_list = ['ClubFoot', 'Normals', 'Hindfoot_Valgus']
    nifti_file_path_list = []
    vtk_file_path_list = []
    bmp_background_file_path_list = []
    folder_num_list = []
    
    if clubfoot_type  == None:
        clubfoot_type = ['AP', 'LAT']

    for group_type in clubfoot_type:
        for group in folder_list:
            os.chdir(os.path.join(base_dir, group))
            group_subfolders = [f.path for f in os.scandir(os.getcwd()) if f.is_dir() ]
            for patient_folder in group_subfolders:
                folder_num = os.path.basename(os.path.normpath(patient_folder))
                os.chdir(patient_folder)        
                # In a specificed file, we need to identify all of the image files that are nii.gz
                for image in [f for f in glob.glob("*.nii")]:            
                    if group_type in image:
                        x = 0
                        y = 0
                        file_path = os.path.join(patient_folder, image) 
                        #nifti_file_path_list.append(file_path)
                        for vtk in [f for f in glob.glob("*.vtk")]:
                            if vtk[:12] == image[:12]:
                                x = x + 1
                                vtk_file_path = os.path.join(patient_folder, vtk) 
                                # vtk_file_path_list.append(vtk_file_path)
                            
                        for bmp in [f for f in glob.glob("*.bmp")]:
                            if bmp[:12] == image[:12]:
                                y = y + 1
                                bmp_file_path = os.path.join(patient_folder, bmp) 
                                #bmp_background_file_path_list.append(bmp_file_path)
                                
                        if x == 1 and y == 1:
                            folder_num_list.append(folder_num)
                            bmp_background_file_path_list.append(bmp_file_path)
                            nifti_file_path_list.append(file_path)
                            vtk_file_path_list.append(vtk_file_path) 
                        else:    
                            print(patient_folder)
                            print(image)      

    os.chdir(base_dir)
    

    # Splitting into training and test sets
    data = pd.DataFrame({'folder': folder_num_list,
                         'nifti_path': nifti_file_path_list,
                         'bmp_background':bmp_background_file_path_list,
                         'vtk_path': vtk_file_path_list})

    #print(len(nifti_file_path_list))
    #print(len(vtk_file_path_list))
    #print(len(bmp_background_file_path_list))
    return data

def mask_images(data, num):
    nifti = sitk.ReadImage(data['nifti_path'].iloc[num])
    nifti_arr = sitk.GetArrayViewFromImage(nifti)
    
    bmp_back = sitk.ReadImage(data['bmp_background'].iloc[num])
    bmp_back_arr = sitk.GetArrayViewFromImage(bmp_back)
    bmp_back_arr = bmp_back_arr[:,:,0]
    
    vtk = sitk.ReadImage(data['vtk_path'].iloc[num])
    vtk_arr = sitk.GetArrayViewFromImage(vtk)
    
    bmp_back_arr = np.where(bmp_back_arr == 255, 1, bmp_back_arr)
    #print(bmp_back_arr.min(), bmp_back_arr.max())
    
    masked_image = nifti_arr[0,] * bmp_back_arr
    
    f, axarr = plt.subplots(1,4, figsize = (10,10)) 
    axarr[0].imshow(nifti_arr[0,], cmap='gray') # [0,]
    axarr[1].imshow(bmp_back_arr, cmap='gray')
    axarr[2].imshow(vtk_arr, cmap='gray')
    axarr[3].imshow(masked_image, cmap='gray')




# Creating Key functions

def ground_truth_background(base_dir, clubfoot_type = None):
    ''' 
    Loop through labeled folders (Normal, Clubfoot, Hindfoot Valgus) to find all ground truth images of a 
    specified radiograph type ('AP' or 'Lat') with a ground truth segmentation map and background.
    This funciton return a list of all of the files with ground truth masks
    '''
    folder_list = ['ClubFoot', 'Normals', 'Hindfoot_Valgus']
    nifti_file_path_list = []
    bmp_background_file_path_list = []
    vtk_file_path_list = []
    folder_num_list = []
    
    if clubfoot_type  == None:
        clubfoot_type = ['AP', 'LAT']
        
    for group_type in clubfoot_type:
        for group in folder_list:
            os.chdir(os.path.join(base_dir, group))
            group_subfolders = [f.path for f in os.scandir(os.getcwd()) if f.is_dir() ]
            
            for patient_folder in group_subfolders:
                folder_num = os.path.basename(os.path.normpath(patient_folder)) 
                os.chdir(patient_folder)        
                
                # In a specificed file, we need to identify all of the image files that are nii
                for image in [f for f in glob.glob("*.nii")]:
                    x, y, z = 0, 0, 0
                    
                    if group_type in image:
                        x = 1
                        file_path = os.path.join(patient_folder, image) 
                        
                            
                        for bmp in [f for f in glob.glob("*.bmp")]:
                            if bmp[:12] == image[:12]:
                                y = 1
                                bmp_file_path = os.path.join(patient_folder, bmp) 
                        
                        for vtk in [f for f in glob.glob("*.vtk")]:
                            if vtk[:12] == image[:12]:
                                z = 1
                                vtk_file_path = os.path.join(patient_folder, vtk) 


                        if x == 1 and y == 1 and z == 1:
                            folder_num_list.append(folder_num)
                            bmp_background_file_path_list.append(bmp_file_path)
                            nifti_file_path_list.append(file_path)
                            vtk_file_path_list.append(vtk_file_path) 
                        else:    
                            print(patient_folder)
                            print(image)             

    os.chdir(base_dir)
          
    data = pd.DataFrame({'folder': folder_num_list,
                         'nifti_path': nifti_file_path_list,
                         'bmp_background':bmp_background_file_path_list,
                         'vtk_path': vtk_file_path_list})

    return data



def back_images(data, num):
    nifti = sitk.ReadImage(data['nifti_path'].iloc[num])
    nifti_arr = sitk.GetArrayViewFromImage(nifti)
    
    bmp_back = sitk.ReadImage(data['bmp_background'].iloc[num])
    bmp_back_arr = sitk.GetArrayViewFromImage(bmp_back)
    bmp_back_arr = bmp_back_arr[:,:,0]
        
    bmp_back_arr = np.where(bmp_back_arr == 255, 1, bmp_back_arr)
    #print(bmp_back_arr.min(), bmp_back_arr.max())
    
    masked_image = nifti_arr[0,] * bmp_back_arr
    
    f, axarr = plt.subplots(1,3, figsize = (10,10)) 
    axarr[0].imshow(nifti_arr[0,], cmap='gray') # [0,]
    axarr[1].imshow(bmp_back_arr, cmap='gray')
    axarr[2].imshow(masked_image, cmap='gray')
    
    
def show_single_image(sample_batched):
    """Show image with landmarks for a batch of samples."""
    image = sample_batched['image'][0,:,:]
    mask = sample_batched['mask'][0,:,:]
    f, axarr = plt.subplots(1,2, figsize = (5,5))
    axarr[0].imshow(image[0,].cpu(), cmap='gray') # [0,]
    axarr[1].imshow(mask[0,].cpu(), cmap='gray')

def show_image_batch(sample_batched):
    batch_size = len( sample_batched['image'])
    for i in range(batch_size):
        image = sample_batched['image'][i,:,:]
        mask = sample_batched['mask'][i,:,:]
        f, axarr = plt.subplots(1,2, figsize = (10,10))
        axarr[0].imshow(image[0,].cpu(), cmap='gray') # [0,]
        axarr[1].imshow(mask[0,].cpu(), cmap='gray')
        
        
        
def show_image_batch_model(sample_batched, output):
    batch_size = len(sample_batched['image'])
    
    for i in range(batch_size):
        image = sample_batched['image'][i,:,:]
        mask = sample_batched['mask'][i,:,:]
        mask = torch.sum(mask, dim = 0)
        output_mask = output[i,:,:].cpu().detach()#.numpy()
        
        output_mask = torch.sum(output_mask, dim = 0)
        f, axarr = plt.subplots(1,3,figsize=(15,15)) 
        axarr[0].imshow(image[0,].cpu(), cmap='gray')
        axarr[1].imshow(mask.cpu(), cmap='gray')
        axarr[2].imshow(output_mask.cpu(), cmap='gray')
        
def save_image_batch_model(sample_batched, output, base_dir, logpath):
    batch_size = len(sample_batched['image'])
    
    for i in range(batch_size):
        image = sample_batched['image'][i,:,:]
        mask = sample_batched['mask'][i,:,:]
        mask = torch.sum(mask, dim = 0)
        output_mask = output[i,:,:].cpu().detach()#.numpy()
        
        output_mask = torch.sum(output_mask, dim = 0)
        f, axarr = plt.subplots(1,3,figsize=(15,15)) 
        axarr[0].imshow(image[0,].cpu(), cmap='gray')
        axarr[1].imshow(mask.cpu(), cmap='gray')
        axarr[2].imshow(output_mask.cpu(), cmap='gray')        
        plt.savefig(os.path.join(base_dir, logpath + '_ex_' + str(i) + '.png'))
        plt.clf()
        plt.close('all')
        
def show_single_image_model(sample_batched, output):    
    image = sample_batched['image'][0,:,:]
    mask = sample_batched['mask'][0,:,:]
    mask = torch.sum(mask, dim = 0)
    output_mask = output[0,:,:].cpu().detach()#.numpy()

    output_mask = torch.sum(output_mask, dim = 0)
    f, axarr = plt.subplots(1,3,figsize=(15,15)) 
    axarr[0].imshow(image[0,].cpu(), cmap='gray')
    axarr[1].imshow(mask.cpu(), cmap='gray')
    axarr[2].imshow(output_mask.cpu(), cmap='gray')
    

def return_criterions(criterionn, augments, model_num):
    if criterionn == 'Combined':
        criterion = DC_loss2()
        criterion2 = BCELoss()
    else:
        criterion = criterionn
        criterion2 = None
        
    if augments == 'Clahe_Aug':
        transformations = [toTensorMask(),
                   Resize_Imgs([272, 256]),
                   Normalize(), 
                   RandomRotate(0.5),
                   RandomFlip(0.5),
                   ClaheFilter()] 
    
    if augments == 'Aug':
        transformations = [toTensorMask(),
                   Resize_Imgs([272, 256]),
                   Normalize(), 
                   RandomRotate(0.5),
                   RandomFlip(0.5)]
    
    if augments == 'No_Aug':
        transformations = [toTensorMask(),
                   Resize_Imgs([272, 256]),
                   Normalize()]
    
    if model_num == 2:
        model = FCNmodel_2pool(n_class=1)
    else:
        model = FCNmodel_3pool(n_class=1)
    
    return criterion, criterion2, transformations, model    
    

