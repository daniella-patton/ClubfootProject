# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 11:04:20 2021
@author: danie


Description: A custom dataloader developed to read a .nii input image, 
a .vtk bone mask, and a background mask (.bmp) image from our dataset.
    
"""

# Required Packages
##############################################################################
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset
from skimage import exposure


#%%
# Helper Functions
##############################################################################
def rgb2gray(rgb):
    """
    Converts a 3-channel input image array and returns a greyscale image
    """
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

def input_image(label):
    """
    Converts our input vtk image and return a seperate mask for each bone we
    may be identifying in the algorithm (first metatarsel, tibia, calcaneus, 
    and talus).
    """
    # Grabbing labeled bones from the gorund truth image
    b = (label == 1) # calcanues
    c = (label == 2) # calcaneus-talus overlap 
    d = (label == 3) # talus
    first_metatarsel = (label == 4) # 1st metatarsel
    tibia = (label == 5) # tibia
    #cuboid= (label == 6) # cuboid
    #cuboid_overlap = (label == 7).astype(int) # cuboid
    calcaneus = b.astype(int) + c.astype(int) # Calcanues
    talus = c.astype(int) + d.astype(int) # Talus
    #cuboid = cuboid_overlap + cuboid
    return (first_metatarsel, tibia, calcaneus, talus)


#%%
#Clubfoot Clahe
##############################################################################
class ClubfootClaheDataset(Dataset):
    """
    A custom datloader that reads in an input image (.nii), rescales the image
    from a range [0,1], applies the CLAHE filter, applies the foot mask, 
    and finally, create a stacked bone mask to return for algorithm training
    
    Inputs: 
        mask (True or False): Whether or not we want to apply a binary
                              mask to the input image. 
                              
        bone (1st_Metatarsel, Tibia, Calcaneus, Talus): 
            Whether or not we want to apply a binary
                              mask to the input image.
                              
        group (AP, LAT, or Both): 
            Whether or not we want to apply a binary
                              mask to the input image.
    """
    def __init__(self, df, mask = True, bone = '1st_Metatarsel', 
                 group = 'both', transform=None):
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
        
        # Preprocessing the Input Image
        ######################################################################
        nifti = sitk.ReadImage(image)
        nifti_arr = sitk.GetArrayViewFromImage(nifti)
        
        # Checking to see if the input image is rgb
        if nifti_arr.shape[2] == 3:
            nifti_arr = rgb2gray(nifti_arr)
            nifti_arr = torch.from_numpy(nifti_arr)
            nifti_arr = nifti_arr.unsqueeze(0)
            nifti_arr = nifti_arr.numpy()
        
        # Preprcossing - Change the min, max value of the input image from 0-1 
        nifti_arr_2 = nifti_arr.copy()
        slope = 1/(nifti_arr_2.max() - nifti_arr_2.min())
        nifti_arr_2_norm = np.subtract(nifti_arr_2, nifti_arr_2.min()) * slope
        
        # Preprocessing - Apply the CLAHE algorithm 
        nifti_arr = exposure.equalize_adapthist(nifti_arr_2_norm[0,], 
                                                kernel_size=(25,25),
                                                clip_limit=0.01,
                                                nbins=256)
        
    
        
        # Preprocessing - Read in the foot mask to crop the input image
        back_mask_2 = sitk.ReadImage(back_mask)
        back_mask_arr = sitk.GetArrayViewFromImage(back_mask_2)
        
        
        # Preprocessing - Change the range from 255 to 1 for foot pixels
        back_mask_arr2 = np.where(back_mask_arr[:,:,0] == 255,
                                  1, back_mask_arr[:,:,0])
        masked_image = nifti_arr * back_mask_arr2
        
        # Cropping so that only the foot is being considered and to add 
        #more similarity
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
                
            
        # Creating the bone mask array
        ######################################################################
        bone_mask2 = sitk.ReadImage(bone_mask)
        bone_mask_arr = sitk.GetArrayViewFromImage(bone_mask2)
        
        # Redefining the input image size as defined above
        bone_mask_arr = bone_mask_arr[axis1_min: axis1_max, 
                                      axis0_min: axis0_max]
        
        bone_mask_arr = bone_mask_arr * back_mask_arr2
        
         # creating the Masks
        first_metatarsel, tibia, calcaneus, talus = input_image(bone_mask_arr)
        if bone =='1st_Metatarsel': return_mask = first_metatarsel
        if bone =='Tibia': return_mask = tibia
        #if bone =='Cuboid': return_mask = cuboid
        if bone =='Calcaneus': return_mask = calcaneus
        if bone =='Talus': return_mask = talus
        if bone == 'Multi':
            #print(path[0])
            if 'AP' in group:
                return_mask = np.stack((first_metatarsel,
                                        calcaneus, 
                                        talus))
                
            if 'LAT' in group:
                return_mask = np.stack((first_metatarsel,
                                        tibia,
                                        calcaneus, 
                                        talus))
            
            if 'both' in group:
                return_mask = np.stack((first_metatarsel,
                                        tibia,
                                        calcaneus, 
                                        talus))
                
        sample = {'image': masked_image.astype(dtype = 'float32'), 
                  'mask': return_mask.astype(dtype = 'int64')}#'long'


        if self.transform:
            sample = self.transform(sample)   
            
        
        return sample
    