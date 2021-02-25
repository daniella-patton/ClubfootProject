# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 14:28:29 2021

@author: danie
"""
from torch.nn import BCELoss
import torch


def dice_coef(inputs, target, smooth = 1e-6):
    iflat = inputs.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    #iflat = input.round()  # use only for evaluation
    #tflat = target.round()  # use only for evaluation
    return (2. * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)
    
def dice_loss(inputs, target):
    return 1 - dice_coef(inputs, target)

def dc_loss2(inputs, target):
    b = BCELoss(inputs, target, size_average=None, reduction='elementwise_mean')
    d = dice_coef(inputs, target)
    return 1 - torch.log(d) + b

def iou_val(inputs, target, smooth = 1e-6):
    iflat = inputs.view(-1)
    #print(iflat.shape)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    #print(intersection)
    union = iflat + tflat
    union[union!= 0] = 1 #.sum()
    union_sum = union.sum()
    #print(union_sum)
    iou = (intersection + smooth) / (union_sum + smooth)  # We smooth our devision to avoid 0/0
    #print(iou.item())
    return iou  # Or thresholded.mean() if you are interested in average across the batch

class DC_loss2(torch.nn.modules.loss._Loss):

    def __init__(self, weight=None, size_average=None, reduce=None, reduction='elementwise_mean', pos_weight=None):
        super(DC_loss2, self).__init__(size_average, reduce, reduction)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return dice_coef(input, target)