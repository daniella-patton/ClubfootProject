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

def input_image(label):
    # Grabbing labeled bones from the gorund truth image
    b = (label == 1) # calcanues
    c = (label == 2) # calcaneus-talus overlap 
    d = (label == 3) # talus
    first_metatarsel = (label == 4) # 1st metatarsel
    tibia = (label == 5) # tibia
    cuboid= (label == 6) # cuboid
    cuboid_overlap = (label == 7).astype(int) # cuboid
    calcaneus = b.astype(int) + c.astype(int) # Calcanues
    talus = c.astype(int) + d.astype(int) # Talus
    cuboid = cuboid_overlap + cuboid
    return (first_metatarsel, tibia, cuboid, calcaneus, talus)

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
                  'mask': bmp_back_arr.astype(dtype = 'int64')} #'long' # 


        if self.transform:
            sample = self.transform(sample)
            
        
        return sample  


##############################################################################
# Clubfoot Bone Background Dataset
##############################################################################    

class ClubfootDataset(Dataset):
    """Clubfoot Dataset."""

    def __init__(self, df, mask = True, bone = '1st_Metatarsel', 
                 group = 'both',transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df
        self.transform = transform
        self.data = [(row.nifti_path,
                      row.bmp_background,
                      row.vtk_path) for row in self.data.itertuples()]
        self.mask = mask
        self.bone = bone
        self.group = group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        # Read in  nifti, background mask, and bone mask
        path = self.data[idx]
        mask = self.mask
        bone = self.bone
        group = self.group
        image, back_mask, bone_mask = path[0], path[1], path[2]
        #print(image)

        nifti = sitk.ReadImage(image)
        nifti_arr = sitk.GetArrayViewFromImage(nifti)
        
        if nifti_arr.shape[2] == 3:
            nifti_arr = rgb2gray(nifti_arr)
            # np.expand_dims(x, axis=0)
            nifti_arr = torch.from_numpy(nifti_arr)
            nifti_arr = nifti_arr.unsqueeze(0)
            #print(nifti_arr.size())
            nifti_arr = nifti_arr.numpy()
            #print(nifti_arr.expand_dims(x, axis=0).shape)
    
        back_mask_2 = sitk.ReadImage(back_mask)
        back_mask_arr = sitk.GetArrayViewFromImage(back_mask_2)
    
        back_mask_arr2 = np.where(back_mask_arr[:,:,0] == 255,
                                  1, back_mask_arr[:,:,0])
    
        if  nifti_arr[0,].shape != back_mask_arr2.shape:
            print(path[0])
            
        masked_image = nifti_arr[0,] * back_mask_arr2
        
        # Cropping so that only the foot is being considered and to add more similarity
        # Axis 1 = y axis
        axis1 = np.sum(masked_image, axis=1)
        axis1_minmax = np.where(axis1 > 0)[0]
        axis1_min, axis1_max = axis1_minmax[0], axis1_minmax[-1]
        #print(axis1_min, axis1_max)
        
        # Axis 0 = x axis
        axis0 = np.sum(masked_image, axis=0)
        axis0_minmax = np.where(axis0 > 0)[0]
        axis0_min, axis0_max = axis0_minmax[0], axis0_minmax[-1]
        #print(axis0_min, axis0_max)
        
        if len(masked_image.shape) == 3:
            masked_image = masked_image[0,:,:]
        
        if mask:
            masked_image = masked_image[axis1_min: axis1_max,
                                        axis0_min: axis0_max]
            back_mask_arr2 =  back_mask_arr2[axis1_min: axis1_max, 
                                             axis0_min: axis0_max]
        else:
            masked_image = nifti_arr[0,][axis1_min: axis1_max, 
                                         axis0_min: axis0_max]
            back_mask_arr2 =  back_mask_arr2[axis1_min: axis1_max, 
                                             axis0_min: axis0_max]
                
        # Read in MASK and convert to array
        bone_mask2 = sitk.ReadImage(bone_mask)
        bone_mask_arr = sitk.GetArrayViewFromImage(bone_mask2)
        
        bone_mask_arr = bone_mask_arr[axis1_min: axis1_max, axis0_min: axis0_max]
        bone_mask_arr = bone_mask_arr * back_mask_arr2
        
        # Testing setup with first metatarsel
        first_metatarsel, tibia, cuboid, calcaneus, talus = input_image(bone_mask_arr)
        if bone =='1st_Metatarsel': return_mask = first_metatarsel
        if bone =='Tibia': return_mask = tibia
        if bone =='Cuboid': return_mask = cuboid
        if bone =='Calcaneus': return_mask = calcaneus
        if bone =='Talus': return_mask = talus
        
        if bone == 'Multi':
            #print(path[0])
            if 'AP' in group: # Image name
                #print('AP in image')
                return_mask = np.stack((first_metatarsel,
                                        calcaneus, 
                                        talus,
                                        cuboid))
            if 'LAT' in group: #Image Name
                #print('lat in image')
                return_mask = np.stack((first_metatarsel,
                                        tibia,
                                        calcaneus, 
                                        talus))
            if 'both' in group: #Image Name
                #print('lat in image')
                return_mask = np.stack((first_metatarsel,
                                        tibia,
                                        calcaneus, 
                                        talus,
                                        cuboid))
                
        
        
        #print(masked_image.shape)
        sample = {'image': masked_image.astype(dtype = 'float32'), 
                  'mask': return_mask.astype(dtype = 'int64')}#'long'


        if self.transform:
            sample = self.transform(sample)   
            #augmented = self.transform(image=masked_image,
            #                           mask=return_mask)
            #masked_image = augmented['image']
            #return_mask = augmented['mask']
            
        
        return sample
    
class ClubfootDataset2(Dataset):
    """Clubfoot Dataset."""

    def __init__(self, df, mask = True, bone = '1st_Metatarsel', 
                 group = 'both',transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = df
        self.transform = transform
        self.data = [(row.nifti_path,
                      row.bmp_background,
                      row.vtk_path) for row in self.data.itertuples()]
        self.mask = mask
        self.bone = bone
        self.group = group

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
                
        # Read in  nifti, background mask, and bone mask
        path = self.data[idx]
        mask = self.mask
        bone = self.bone
        group = self.group
        image, back_mask, bone_mask = path[0], path[1], path[2]
        #print(image)

        nifti = sitk.ReadImage(image)
        nifti_arr = sitk.GetArrayViewFromImage(nifti)
        
        if nifti_arr.shape[2] == 3:
            nifti_arr = rgb2gray(nifti_arr)
            nifti_arr = torch.from_numpy(nifti_arr)
            nifti_arr = nifti_arr.unsqueeze(0)
            nifti_arr = nifti_arr.numpy()
            
        back_mask_2 = sitk.ReadImage(back_mask)
        back_mask_arr = sitk.GetArrayViewFromImage(back_mask_2)
    
        back_mask_arr2 = np.where(back_mask_arr[:,:,0] == 255,
                                  1, back_mask_arr[:,:,0])
    
        if  nifti_arr[0,].shape != back_mask_arr2.shape:
            print(path[0])
            
        #masked_image = nifti_arr[0,] * back_mask_arr2
        
        # Cropping so that only the foot is being considered and to add more similarity
        # Axis 1 = y axis
        axis1 = np.sum(back_mask_arr2, axis=1)
        axis1_minmax = np.where(axis1 > 0)[0]
        axis1_min, axis1_max = axis1_minmax[0], axis1_minmax[-1]
        
        # Axis 0 = x axis
        axis0 = np.sum(back_mask_arr2, axis=0)
        axis0_minmax = np.where(axis0 > 0)[0]
        axis0_min, axis0_max = axis0_minmax[0], axis0_minmax[-1]
        
        masked_image = nifti_arr[0,]
        
        if len(masked_image.shape) == 3:
            masked_image = masked_image[0,:,:]
        
        if mask:
            masked_image = masked_image[axis1_min: axis1_max,
                                        axis0_min: axis0_max]
            back_mask_arr2 =  back_mask_arr2[axis1_min: axis1_max, 
                                             axis0_min: axis0_max]
        else:
            masked_image = nifti_arr[0,][axis1_min: axis1_max, 
                                         axis0_min: axis0_max]
            back_mask_arr2 =  back_mask_arr2[axis1_min: axis1_max, 
                                             axis0_min: axis0_max]
                
        # Read in MASK and convert to array
        bone_mask2 = sitk.ReadImage(bone_mask)
        bone_mask_arr = sitk.GetArrayViewFromImage(bone_mask2)
        
        bone_mask_arr = bone_mask_arr[axis1_min: axis1_max, axis0_min: axis0_max]
        bone_mask_arr = bone_mask_arr * back_mask_arr2
        
        # Testing setup with first metatarsel
        first_metatarsel, tibia, cuboid, calcaneus, talus = input_image(bone_mask_arr)
        if bone =='1st_Metatarsel': return_mask = first_metatarsel
        if bone =='Tibia': return_mask = tibia
        if bone =='Cuboid': return_mask = cuboid
        if bone =='Calcaneus': return_mask = calcaneus
        if bone =='Talus': return_mask = talus
        
        if bone == 'Multi':
            #print(path[0])
            if 'AP' in group: # Image name
                #print('AP in image')
                return_mask = np.stack((first_metatarsel,
                                        calcaneus, 
                                        talus,
                                        cuboid))
            if 'LAT' in group: #Image Name
                #print('lat in image')
                return_mask = np.stack((first_metatarsel,
                                        tibia,
                                        calcaneus, 
                                        talus))
            if 'both' in group: #Image Name
                #print('lat in image')
                return_mask = np.stack((first_metatarsel,
                                        tibia,
                                        calcaneus, 
                                        talus,
                                        cuboid))
                
    
        sample = {'image': masked_image.astype(dtype = 'float32'), 
                  'mask': return_mask.astype(dtype = 'int64')}#'long'


        if self.transform:
            sample = self.transform(sample)   
        
        return sample

    
