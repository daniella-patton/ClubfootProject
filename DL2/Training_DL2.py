# -*- coding: utf-8 -*-
"""
Created on Mon Feb 22 14:33:39 2021

@author: danie
"""

# Downloaded Packages
import os
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
#import torch.nn as nn
from torch.nn import BCELoss
import segmentation_models_pytorch as smp
import torch.optim as optim
import shutil
import torch
import time

# Custom written .py files
os.chdir('C:\\Users\\danie\\OneDrive\\Documents\\Clubfoot\\GroundTruthSegmentation\\Final_Seg_Maps_NIFTI2\\Code\\Dl2')
#print(os.getcwd())
from Helper_DL2 import ground_truth_files, training_results_figs, show_image_batch
from Transforms_DL2 import toTensor, LargeSizeCrop, combinedPad, Normalize
from Transforms_DL2 import RandomNoise, RandomRotate, RandomFlip, RandomZoom
from Dataset_DL2 import ClubfootClaheDataset
from Models_DL2 import FCNmodel_3pool


torch.cuda.empty_cache()
#%%
# Loading the Data from prediefined paths
##############################################################################
# Reading in Ground Truth Files
base = 'C:\\Users\\danie\\OneDrive\\Documents\\Clubfoot\\GroundTruthSegmentation\\\Final_Seg_Maps_NIFTI2'
data = ground_truth_files(base, clubfoot_type = None) # clubfoot_type = 'AP','LAT' 
len(data)

# Training and Validation Sets
split = pd.read_csv('Clubfoot_Update_012521.csv')

split['Folder ID'] = split['Folder ID'].astype(str)

data_2 = data.merge(split, left_on='folder', right_on='Folder ID',
                    how='inner')
data_2 = data_2.drop(columns = ['Folder ID'])

training_data = data_2[data_2['Split'] == 'Training']
validation_data = data_2[data_2['Split'] == 'Validation']

# Clubfoot_Update_012521
print(' The training data size is: ', len(training_data),'\n',
      'The validation data size is: ', len(validation_data), '\n')

del data, split, data_2
#%%
# Defining the Training and Validation Sets
##############################################################################
print('Loading the Dataset')
validation_dataset = ClubfootClaheDataset(df = validation_data,
                                     group = 'both',
                                     bone = 'Multi',
                                     transform=transforms.Compose([toTensor(),
                                                                   LargeSizeCrop(256),
                                                                   combinedPad(),
                                                                  ]))
                                                                   
validation_dataloader = DataLoader(validation_dataset, batch_size=4,
                        shuffle=False, num_workers=0)

training_dataset = ClubfootClaheDataset(df = training_data,
                                   bone = 'Multi',
                                    group = 'both',
                                    transform=transforms.Compose([toTensor(),
                                                                  LargeSizeCrop(256),
                                                                  combinedPad(),
                                                                  RandomNoise(0.5),
                                                                  RandomRotate(0.5),
                                                                  RandomFlip(0.5),
                                                                  RandomZoom(0.5)
                                                                 ]))
training_dataloader = DataLoader(training_dataset, batch_size=4,
                        shuffle=True, num_workers=0)
#%%    
# Loading the model and sending to GPU (if available)
############################################################################## 
print('Defining the model')

aux_params=dict(classes = 4)

model = smp.Unet(
    encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",     # use `imagenet` pretrained weights for encoder initialization
    in_channels=1,  # model input channels (1 for grayscale images, 3 for RGB, etc.)
    classes = 4,
    activation = 'sigmoid',
    aux_params=aux_params
)

#model = FCNmodel_3pool(n_class = 4)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#%%
# Defining the criterion and the loss function
##############################################################################
print('Defining the loss criteria')
def dice_coef(inputs, target):
    smooth = 1.

    iflat = inputs.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    #iflat = input.round()  # use only for evaluation
    #tflat = target.round()  # use only for evaluation
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    
def dice_loss(input, target):
    return 1 - dice_coef(input, target)

def dc_loss2(input, target):
    b = BCELoss(input, target, size_average=None,
                reduction='elementwise_mean')
    d = dice_coef(input, target)
    return 1 - torch.log(d) + b    
    
class DC_loss2(torch.nn.modules.loss._Loss):

    def __init__(self, weight=None, size_average=None, reduce=None, 
                 reduction='elementwise_mean', pos_weight=None):
        super(DC_loss2, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return dice_coef(input, target)    

criterion = DC_loss2()
criterion2 = BCELoss()
parameters = model.parameters()
optimizer = optim.Adam(parameters, lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5)
criterion.to(device)
criterion2.to(device)

#%%
# Defining the training loop
##############################################################################
# The path where training weights amd model results will be saved
logpath = 'C:\\Users\\danie\\OneDrive\\Documents\\Clubfoot\\GroundTruthSegmentation\\Final_Seg_Maps_NIFTI2\\Models\\'
# Filename: the name of the file you are saving 
filename = 'unet_clahe_augmentation'
#%%
# Define the number of epochs
epoch_range = 100

def save_checkpoint(model, optimizer, train_epoch_metrics, val_epoch_metrics,
                    epoch, is_best, logpath, filename):
    logpath = logpath + filename
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_epoch_metrics': train_epoch_metrics,
        'val_epoch_metrics': val_epoch_metrics}

    filename = "{}.pth.tar".format(logpath)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, '{}.best.pth.tar'.format(logpath))
        
    
def run_epoch(mode, loader, net, criterion, criterion2, optimizer, alpha=0.99):
    if mode == 'train':
        net.train()
    else:
        net.eval()
    total_loss = 0
    total_dice = 0
    total_bce = 0
    for i, data in enumerate(loader):
        inputs = data['image']
        targets = data['mask']

        inputs, targets = inputs.to(device), targets.to(device)

        with torch.set_grad_enabled(mode == 'train'):
            outputs, labels = net(inputs) # outputs, labels
            d = criterion(outputs, targets.float())
            b = criterion2(outputs, targets.float())
            loss = 1 - torch.log(d) + b

            total_loss += loss.item()
            total_dice += d.item()
            total_bce += b.item()

            if mode == 'train':
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

    print('finished epoch', epoch, '{} mode'.format(mode), 
          'loss', total_loss / (i + 1))

    return {'loss': total_loss / (i + 1), 
            'dice': total_dice / (i + 1), 
            'bce': total_bce / (i + 1)}

train_epoch_metrics = {
        'loss': [],
        'dice': [],
        'bce': []
    }
val_epoch_metrics = {
        'loss': [],
        'dice': [],
        'bce': []
    }

is_best = False
best_loss = 0

for epoch in range(epoch_range):
    print('Starting Epoch # ', epoch)
    start_time = time.time()
    torch.cuda.empty_cache()
    train_metrics = run_epoch('train', training_dataloader, 
                              model, criterion, criterion2, optimizer)
    
    torch.cuda.empty_cache()
    valid_metrics = run_epoch('validation', validation_dataloader,
                              model, criterion, criterion2, optimizer)
    
    torch.cuda.empty_cache()
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(valid_metrics['loss'])
    else:
        scheduler.step()
    
    for m in valid_metrics:
        train_epoch_metrics[m].append(train_metrics[m])
        val_epoch_metrics[m].append(valid_metrics[m])
    
    if epoch == 0 or valid_metrics['loss'] < best_loss:
        is_best = True
        best_loss = valid_metrics['loss']
        
    save_checkpoint(model, optimizer, train_epoch_metrics, 
                    val_epoch_metrics, epoch, is_best, 
                    logpath, filename)
    
    is_best = False
    elapsed_time = time.time() - start_time
    
    elapse_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    print('Training epoch # ', epoch, ' took ', elapse_time)

print('Finished Training')

#%%
# Comparing the Results 
##############################################################################
training_results_figs(train_epoch_metrics, logpath, 
                     filename, group = 'Train')

training_results_figs(val_epoch_metrics, 
                      logpath, filename, group = 'Validation')
  

#%%
# Loading the model with the best weights
##############################################################################
os.chdir(logpath)
print('Loading the Best Model Weights')
# evaluate model:
model.eval();

optimizer_ = optim.Adam(model.parameters())
base_dir = os.getcwd()
print(filename + '.best.pth.tar')
checkpoint = torch.load(filename + '.best.pth.tar')
model.load_state_dict(checkpoint['model_state_dict'])
optimizer_.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']

print('The best model training paramters was found on epoch # ', epoch)
#%%
# Loading the model with the best weights
##############################################################################
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device('cpu')
model = model.to(device)
        
for i_batch, sample_batched in enumerate(validation_dataloader):
    print(i_batch, sample_batched['image'].size(),sample_batched['mask'].size())
    if i_batch == 0:
        mask, labels = model(sample_batched['image'].to(device)) # , label
        #print(mask.shape, label.shape)
        plt.figure()
        show_image_batch(sample_batched, torch.round(mask), logpath, filename)
        break

#%%
# Loading the CSV
##############################################################################
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#criterion.to(device)

model = model.to('cpu')#.to('cpu')
colnames = ['1st metarsel', 'tibia', 'calcaneus', 'talus']
data = []
image_dice = []
for i_batch, sample_batched in enumerate(validation_dataloader):
    mask, labels = model(sample_batched['image'].to('cpu'))#.to('cuda')#.to('cpu')) # , label
    mask =  torch.round(mask)
    
    for j in range(sample_batched['mask'].shape[0]):
        gt = sample_batched['mask'][j,]
        model_mask = mask[j,]
        
        data = []
        for i in range(sample_batched['mask'].shape[1]):
            gt_bone = gt[i,]
            model_mask_bone = model_mask[i,].to('cpu')
            data.append(dice_coef(model_mask_bone, gt_bone).item())
            
        image_dice.append([data])
        
        
bone_dice = pd.DataFrame(image_dice, columns = ['Test']) 
bone_dice['Test']  = bone_dice['Test'].astype('str')
bone_dice['Test'] = bone_dice['Test'].str.strip('[]')
#bone_dice['Test'] = bone_dice['Test'].str.lstrip(']')
tmpDF = pd.DataFrame()
tmpDF = bone_dice['Test'].str.split(',',expand=True)
tmpDF.head()
tmpDF.columns=colnames
tmpDF.tail()

tmpDF['1st metarsel'] = tmpDF['1st metarsel'].astype(float)
tmpDF['tibia'] = tmpDF['tibia'].astype(float)
tmpDF['calcaneus'] = tmpDF['calcaneus'].astype(float)
tmpDF['talus'] = tmpDF['talus'].astype(float)

tmpDF.describe()
savefile = logpath + filename + '.csv'
tmpDF.to_csv(savefile)