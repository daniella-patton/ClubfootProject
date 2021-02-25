# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 15:52:39 2021

@author: zhengq
"""

import numpy as np#import opencv
import random
from PIL import Image
from scipy import ndimage
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
import cv2
from skimage import exposure

##############################################################################
# Custom Augmentations for Bone Background Mask
##############################################################################

class ClaheFilter():
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        # Preprocessing - Apply the CLAHE algorithm 
        nifti_arr = exposure.equalize_adapthist(image[0,].numpy(), 
                                                kernel_size=(25,25),
                                                clip_limit=0.01,
                                                nbins=256)
        nifti_arr = torch.tensor(nifti_arr, dtype=torch.float32)
        return {'image': nifti_arr.unsqueeze(0), 'mask' : mask}
    
    
class toTensorMask2():
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.shape)
        preprocess = transforms.Compose([transforms.ToTensor()])
        image = preprocess(image)
        mask = preprocess(mask)
        #image = image.permute(1,2, 0)
        return {'image': image, 'mask' : mask}
    


class toTensorMask():
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.shape)
        preprocess = transforms.Compose([transforms.ToTensor()])
        image = preprocess(image)
        mask = preprocess(mask)
        image = image.permute(1,2, 0)
        return {'image': image, 'mask' : mask}

class Resize_Imgs(object):
    """ Greyscale input image"""
    def __init__(self, factor):
        #assert isinstance(factor, int)
        self.factor = factor
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.size())
        h, w = image.shape[:2]
        #print(h,w)
        down_sized = self.factor
        if len(down_sized)== 2:
            down_sized_x = down_sized[0]
            down_sized_y = down_sized[1]
        else:
            down_sized_x = down_sized
            down_sized_y = down_sized
            
        image = TF.resize(image.unsqueeze(0), size = [down_sized_x, down_sized_y])
        mask = TF.resize(mask.unsqueeze(0), size = [down_sized_x, down_sized_y],
                         interpolation  = Image.NEAREST)
        
        #print(image.size())
          
        return {'image': image.squeeze(0), 'mask' : mask.squeeze(0)}

class Normalize(object):
    """ Greyscale input image"""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        m, s = image.mean(), image.std()
        epsilon = 1e-7
        preprocess = transforms.Compose([transforms.Normalize(mean= m, std= (s + epsilon))])
        image = preprocess(image)
        # Adjust Range from 0 - 1
        slope = 1/(image.max() - image.min())
        image = (image - image.min()) * slope
        
        #print(image.min(), image.max())
        return {'image': image, 'mask' : mask}
    
class RandomRotate(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.dtype, mask.dtype)
        #print(image.shape, mask.shape)
        prob = self.probability
        num = int(1/prob)
        
        angle_counter = random.randint(0, num)
        if angle_counter == 0:
            angle = random.randint(0, 360)
            image = TF.rotate(image.unsqueeze(0), angle)
            mask = TF.rotate(mask.unsqueeze(0), angle)
            image, mask = image.squeeze(0), mask.squeeze(0)
            #print(image.dtype, mask.dtype)
            #print(image.shape, mask.shape)
        
        return {'image': image, 'mask' : mask}
    
    
class RandomFlip(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        prob = self.probability
        num = int(1/prob)
        
        flip_counter = random.randint(0, num)
        if flip_counter == 0:
            preprocess = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
            image = preprocess(image)
            mask = preprocess(mask)
            
        return {'image': image, 'mask' : mask}