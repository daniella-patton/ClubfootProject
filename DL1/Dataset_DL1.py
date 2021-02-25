# -*- coding: utf-8 -*-
"""
Last Edited: Tuesday, Fabruay 9th 2021
@author: Daniella Patton
@email: pattondm@chop.edu
"""
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset

##############################################################################
# Helper Functions
##############################################################################

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

##############################################################################
# Bone Background Dataset
##############################################################################

class BoneBackground(Dataset):
    """Clubfoot Dataset."""

    def __init__(self, df, transform=None):
        """
        Args:
            df (pandas dataframe): 
                Directory with all the images with corresponding foot mask.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df
        self.transform = transform
        self.data = [(row.nifti_path,
                      row.bmp_background) for row in self.data.itertuples()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        # Read in  nifti, background mask, and bone mask
        path = self.data[idx]
        image, back_mask = path[0], path[1]

        nifti = sitk.ReadImage(image)
        nifti_arr = sitk.GetArrayViewFromImage(nifti)
        
        if nifti_arr.shape[2] == 3:
            nifti_arr = rgb2gray(nifti_arr)
            nifti_arr = torch.from_numpy(nifti_arr)
            nifti_arr = nifti_arr.unsqueeze(0)
            nifti_arr = nifti_arr.numpy()
    
    
        bmp_back = sitk.ReadImage(back_mask)
        bmp_back_arr = sitk.GetArrayViewFromImage(bmp_back)
        bmp_back_arr = np.where(bmp_back_arr[:,:,0] == 255,
                                  1, bmp_back_arr[:,:,0])
        
        sample = {'image': nifti_arr.astype(dtype = 'float32'), 
                  'mask': bmp_back_arr.astype(dtype = 'long')} #'long' # 


        if self.transform:
            sample = self.transform(sample)
            
        
        return sample  
