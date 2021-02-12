# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 17:02:38 2021

@author: danie
"""

import shutil
import torch
from loss_functions import iou_val, DC_loss2
from torch.nn import BCELoss
import matplotlib.pyplot as plt
import os

# Save best model at checkpoints
def save_checkpoint(model, optimizer, train_epoch_metrics, val_epoch_metrics,
                    epoch, is_best, base_dir, logpath):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_epoch_metrics': train_epoch_metrics,
        'val_epoch_metrics': val_epoch_metrics}

    filename = "{}.pth.tar".format(logpath)
    torch.save(state, os.path.join(base_dir, filename))
    if is_best:
        os.chdir(base_dir)
        shutil.copyfile(filename, '{}.best.pth.tar'.format(logpath))

        
def run_epoch(epoch, mode, data_loader, model, criterion, criterion2, 
              optimizer, device): #, alpha=0.99
    if mode == 'train':
        model.train()
    if mode == 'validate' or mode =='test':
        model.eval()
    #print(criterion2)
    if criterion2 == None:
        #print('true')
        criterion1 = DC_loss2()
        criterion2b = BCELoss()
        
    total_loss = 0
    total_dice = 0
    total_bce = 0
    total_iou = 0
    
    for i, data in enumerate(data_loader):
        inputs = data['image'].to(device)
        targets = data['mask'].to(device)

        with torch.set_grad_enabled(mode == 'train'):
            outputs = model(inputs)
            if criterion2 == None:
                if str(criterion) == 'DC_loss2()':
                    #print('True')
                    loss = 1 - criterion(outputs, targets.float())
                else:
                    loss = criterion(outputs, targets.float())
                d = criterion1(outputs, targets.float())
                b = criterion2b(outputs, targets.float())
            else:                
                d = criterion(outputs, targets.float())
                b = criterion2(outputs, targets.float())
                loss = 1 - torch.log(d) + b
            
            iou = iou_val(torch.round(outputs), targets.float())

            total_loss += loss.item()
            total_dice += d.item()
            total_bce += b.item()
            total_iou += iou.item()

            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    print('finished epoch', epoch, '{} mode'.format(mode), 'loss', total_loss / (i + 1))

    return {'loss': total_loss / (i + 1), 
            'dice': total_dice / (i + 1),
            'bce': total_bce / (i + 1),
            'iou': total_iou / (i + 1)}      

    
def model_training(model, params):

    criterion = params['criterion']
    criterion2 = params['criterion2']
    num_epochs = params['num_epochs']
    training_dl = params['train_dl']
    validation_dl = params['val_dl']
    logpath = params['log_path']
    optimizer = params['optimizer']
    scheduler = params['lr_scheduler']
    base_dir = params['base_dir']


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    criterion.to(device)
    
    if criterion2 != None:
        criterion2.to(device)

    train_epoch_metrics = {
            'loss': [],
            'dice': [],
            'bce':  [],
             'iou': []
        }
    val_epoch_metrics = {
            'loss': [],
            'dice': [],
            'bce':  [],
            'iou':  []
        }

    is_best = False
    best_loss = 0

    for epoch in range(num_epochs):
        train_metrics = run_epoch(epoch, 'train', training_dl, model, 
                                  criterion,
                                  criterion2, 
                                  optimizer,
                                  device)
        
        valid_metrics = run_epoch(epoch, 'test', 
                                  validation_dl,
                                  model, 
                                  criterion, 
                                  criterion2,
                                  optimizer, 
                                  device)

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(valid_metrics['loss'])
        else:
            scheduler.step()

        for m in valid_metrics:
            train_epoch_metrics[m].append(train_metrics[m])
            val_epoch_metrics[m].append(valid_metrics[m])

        if epoch == 0 or valid_metrics['loss'] < best_loss:
            is_best = True
            print('The best loss value at epoch', epoch, 'is: ', valid_metrics['loss'])
            best_loss = valid_metrics['loss']
        save_checkpoint(model, optimizer, train_epoch_metrics, val_epoch_metrics, epoch, 
                        is_best, base_dir, logpath)
        is_best = False

    print('Finished Training')
    return train_epoch_metrics, val_epoch_metrics


def save_images(train_epoch_metrics, val_epoch_metrics, base_dir, logpath):
    plt.rcParams['figure.figsize'] = 10, 10
    plt.plot([i for i in range(len(train_epoch_metrics['loss']))], train_epoch_metrics['loss'], label='loss')
    plt.plot([i for i in range(len(train_epoch_metrics['dice']))], train_epoch_metrics['dice'], label='dice')
    plt.plot([i for i in range(len(train_epoch_metrics['bce']))], train_epoch_metrics['bce'], label='bce')
    plt.plot([i for i in range(len(train_epoch_metrics['iou']))], train_epoch_metrics['iou'], label='iou')

    plt.title('Train metrics')
    plt.legend();
    plt.savefig(os.path.join(base_dir, logpath + '_train_'  + '.png'))
    plt.clf()

    plt.plot([i for i in range(len(val_epoch_metrics['loss']))], val_epoch_metrics['loss'], label='loss')
    plt.plot([i for i in range(len(val_epoch_metrics['dice']))], val_epoch_metrics['dice'], label='dice')
    plt.plot([i for i in range(len(val_epoch_metrics['bce']))], val_epoch_metrics['bce'], label='bce')
    plt.plot([i for i in range(len(val_epoch_metrics['iou']))], val_epoch_metrics['iou'], label='iou')

    plt.title('Validation metrics')
    plt.legend()
    plt.savefig(os.path.join(base_dir, logpath + '_valid_'  + '.png'))
    plt.clf()
    plt.close('all')
    

