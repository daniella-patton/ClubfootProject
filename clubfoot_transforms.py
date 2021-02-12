# -*- coding: utf-8 -*-
"""
Daniella Patton
Pytorch Custom Image Transformations
Date: 02/11/2021
"""

import numpy as np#import opencv
import random
from PIL import Image
from scipy import ndimage
import torchvision.transforms.functional as TF
from torchvision import transforms
import torch
import cv2

##############################################################################
# Custom Augmentations for Bone Background Mask
##############################################################################

class ClaheFilter():
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.min(), image.max())
        image_2 = image.copy()

        slope = 65535/(image_2.max() - image_2.min())
        image_2 = np.subtract(image_2, image_2.min()) * slope
        image = image_2
        
        #print(image[0,].astype(dtype = 'uint16').min(), image[0,].astype(dtype = 'uint16').max())
        image = image_2.astype('uint16')
        #else: 
        #image = image[0,].astype('uint16')
        #print(image.shape)    
        # create a CLAHE object (Arguments are optional).
        clahe = cv2.createCLAHE(clipLimit = 0.1, tileGridSize=(25,25))
        image = clahe.apply(image[0,])
        return {'image': image.astype('float32'), 'mask' : mask}
    
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
        
        
class RandomZoom(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        prob = self.probability
        num = int(1/prob)
        
        zoom_counter = random.randint(0, num)
        if zoom_counter == 0:
            scale_val = round(random.uniform(0.70, 1.5), 2)
            image = TF.affine(image, scale = scale_val, angle = 0,
                              translate = [0,0], shear = [0,0])
            mask = TF.affine(mask, scale = scale_val, angle = 0,
                              translate = [0,0], shear = [0,0])
            
        return {'image': image, 'mask' : mask}
    
    
        
class RandomNoise(object):
    def __init__(self, probability):
        assert isinstance(probability, float)
        self.probability = probability
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        prob = self.probability
        num = int(1/prob)       
        randomnoise_counter = random.randint(0, num)
        if randomnoise_counter == 0:
            #torch_fill = torch.zeros(image.size())
            min_val = torch.min(image).item()
            num_1 = random.randint(0, 256)
            num_2 = 0
            while num_1 > num_2:
                num_2 = random.randint(0, 256)
            sd = random.uniform(0, 1)
            mask_edit = torch.zeros(image.size())
            # Axis 0 = x axis
            #axis0 = np.sum(image, axis=0)
            val = []
            for j in range(0, 256):
                if image[0,0,j] != min_val:
                    val.append(j)
            if len(val) <2:
                min_val = 0
                max_val = 256
            else:
                min_val = val[0]
                max_val = val[-1]
            for i in range(num_1, num_2):
                for j in range(min_val, max_val):
                    mask_edit[0, i, j] = sd
            image = image + mask_edit
        return {'image': image, 'mask' : mask}
    
    

    
##############################################################################
# Custom Augmentations for Bone Regions Mask
##############################################################################

class toTensor():
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        orig = mask.shape[0]
        preprocess = transforms.Compose([transforms.ToTensor()])
        image = preprocess(image)
        mask = preprocess(mask)
        

        if orig > 2:
            mask = mask.permute(1, 2, 0)
            
        return {'image': image, 'mask' : mask}


class LargeSizeCrop(object):
    """ Greyscale input image"""
    def __init__(self, factor):
        assert isinstance(factor, int)
        self.factor = factor
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[1:3]
        down_sized = self.factor
        if h >=w:
            new_w = round((down_sized * w)/h)
            new_h = down_sized
        else:
            new_h = round((down_sized * h)/w)
            new_w = down_sized
            
        
        image = TF.resize(image.unsqueeze(0), size = [new_h, new_w])
        mask = TF.resize(mask.unsqueeze(0), size = [new_h, new_w])
          
        return {'image': image.squeeze(0), 'mask' : mask.squeeze(0)}
    
    
class combinedPad():
    """ Greyscale input image"""
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[1:3]
        if h >= w:
            base = h
            h_pad = 0
            w_pad = round((h - w)/2)
        else:
            base = w
            h_pad = round((w - h)/2)
            w_pad = 0
            
        image = TF.pad(image, padding = [w_pad, h_pad])
        mask = TF.pad(mask, padding = [w_pad, h_pad])
        
        image = TF.resize(image.unsqueeze(0), size = [base, base])
        mask = TF.resize(mask.unsqueeze(0), size = [base, base])
    
        return {'image': image.squeeze(0), 'mask' : mask.squeeze(0)}    
        






class Resize_CenterCrop(object):
    """ Greyscale input image"""
    def __init__(self, factor):
        assert isinstance(factor, int)
        self.factor = factor
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        down_sized = self.factor
        
        #preprocess = transforms.Compose([transforms.ToTensor()])
        #image = preprocess(image)
        #mask = preprocess(mask)
        
        image = torch.from_numpy(image)
        mask = torch.from_numpy(mask)
        #print(image.shape)
        
        image = TF.resize(image.unsqueeze(0), size = [down_sized])
        mask = TF.resize(mask.unsqueeze(0), size = [down_sized])
        
        preprocess = transforms.Compose([transforms.CenterCrop(down_sized)])
        image = preprocess(image)
        mask = preprocess(mask)
          
        return {'image': image.squeeze(0), 'mask' : mask.squeeze(0)}


class ImgRandRotate:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.dtype, mask.dtype)
        
        
        angle_counter = random.randint(0,3)
        if angle_counter == 0:
            angle = random.randint(0, 360)
            image = TF.rotate(torch.from_numpy(image).unsqueeze(0), angle)
            mask = TF.rotate(torch.from_numpy(mask).unsqueeze(0), angle)
            image, mask = image.numpy(), mask.numpy()
                
        flip_counter = random.randint(0,3)
        if flip_counter == 0:
            hor_vert = random.randint(0, 1)
            if hor_vert == 0:
                preprocess = transforms.Compose([transforms.RandomHorizontalFlip(p=1)])
                image = preprocess(torch.from_numpy(image))
                mask = preprocess(torch.from_numpy(mask))
                image,mask = image.numpy(), mask.numpy()
                
            if hor_vert == 1:
                preprocess = transforms.Compose([transforms.RandomVerticalFlip(p=1)])
                image = preprocess(torch.from_numpy(image))
                mask = preprocess(torch.from_numpy(mask))
                image, mask = image.numpy(), mask.numpy()
        
        return {'image': image, 'mask' : mask}


class Rescale(object):
    """ Greyscale input image"""
    def __init__(self, factor):
        assert isinstance(factor, int)
        self.factor = factor
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        # print(h, w)
        
        down_sized = self.factor
        image = image[::down_sized]
        image = image[:, ::down_sized]
        
        mask = mask[::down_sized]
        mask = mask[:, ::down_sized]
        
        return {'image': image, 'mask' : mask}
    
class CMS_Crop(object):
    """ Greyscale input image"""
    def __init__(self, cropsize):
        assert isinstance(cropsize, int)
        self.crop = cropsize
    
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        h, w = image.shape[:2]
        new_w = self.crop
        new_h = self.crop
        
        if h - new_h <= 0:
            padding = new_h - h + 1
            image = np.pad(image, padding, mode='symmetric')
            mask = np.pad(mask, padding, mode='symmetric')
            h, w = image.shape[:2]

        if w - new_w <= 0:
            padding = new_w - w + 1
            image = np.pad(image, padding, mode='symmetric')
            mask = np.pad(mask, padding, mode='symmetric')
            h, w = image.shape[:2]
            
        # Calculate the cms
        cms = ndimage.measurements.center_of_mass(mask)
        #print(cms)
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        #print(cms, type(cms[0]))


        #print(cms[0].dtype)
        try:
            x = 0
            #print('Cms: ', int(cms[0]), int(cms[1]))
            while int(cms[0]) not in range(top, top + new_h):
                top = np.random.randint(0, h - new_h)
                x = x + 1
                if x == 250:
                    break

            #print('Crop Height Range: ',  top, '-', top + new_h)
            while int(cms[1]) not in range(left, left + new_w):
                left = np.random.randint(0, w - new_w)
                x = x + 1
                if x == 250:
                    print('no range')
                    break
        except:
            pass
        
        #print('Crop Width Range: ',  left, '-', left + new_w)
        image = image[top: top + new_h,
                      left: left + new_w]
        
        mask = mask[top: top + new_h,
                      left: left + new_w]

        return {'image': image, 'mask' : mask}
    


class ImgAugTransform:
    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']
        #print(image.dtype, mask.dtype)
        
        # Randomly add noise
        gaus_counter = random.randint(0, 3) # decide on a k each time the loop runs
        if gaus_counter == 0:
            mean = image.mean()   # some constant
            std = image.std()    # some constant (standard deviation)
            image = image + np.random.normal(mean/8, std/8, image.shape)
            image = np.clip(image, 0, 65535)  # we might get out of bounds due to noise
            image = image.astype(np.float32, casting='same_kind')
        
        
        speck_counter = random.randint(0,3)
        if speck_counter == 0:
            speckle = np.random.randint((image.std()/4), size = image.shape)
            image = image + speckle
            image = np.clip(image, 0, 65535)
            image = image.astype(np.float32, casting='same_kind')
        
        
        angle_counter = random.randint(0,1)
        if angle_counter == 0:
            angle = random.randint(0, 360)
            image = Image.fromarray(np.uint16(image))
            mask = Image.fromarray(np.uint16(mask))
            image = TF.rotate(image, angle)
            mask = TF.rotate(mask, angle)
            image = np.array(image)
            image = image.astype(np.float32, casting='same_kind')
            mask = np.array(mask)
            mask = mask.astype(np.int64, casting='same_kind')
            
            
        hflip_counter = random.randint(0,1)
        if hflip_counter == 0:
            image = Image.fromarray(np.uint16(image))
            mask = Image.fromarray(np.uint16(mask))
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            image = np.array(image)
            image = image.astype(np.float32, casting='same_kind')
            mask = np.array(mask)
            mask = mask.astype(np.int64, casting='same_kind')


        return {'image': image, 'mask' : mask}#