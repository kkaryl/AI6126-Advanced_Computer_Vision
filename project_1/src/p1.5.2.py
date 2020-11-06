#!/usr/bin/env python
# coding: utf-8

# # AI6126 ACV Project 1
# 

# In[1]:


nb_ver = 1.5
title = f'ai6126-p1-train-v{nb_ver}'
print(title)
comments = "52"
print(comments)


# ## Versioning & References

# ### Changelogs
# + V0.1 - Setup codes to download and unzip celeba to gDrive
# + V0.2 - Added training loop 
# + V0.3 - Added seeding + save/ load checkpoint
# + V0.4 - Added time taken + save output
# + V0.5 - Added RandomErasing to transforms
# + V0.6 - Added get_criterion (FocalLoss) 
# + V0.7 - Added FaceAttrMobileNetV2 & FaceAttrResNeXt
# + V0.8 - Added Albumentations
# + V0.9 - Updated Optimizer (SGD, AdamW works well)
# + V0.91 - Added ModelTimer() + Added more augmentations
# + V1.0 - Added ReduceLROnPlateau Scheduler
# + V1.1 - Updated Augmentations to more closely follow Tricks paper + Added OneCycleLR Scheduler + No bias decay
# + V1.2 - Added Early Stopping
# + V1.3 - Code Clean
# + V1.4 - Added LabelSmoothing to CrossEntropyLoss and FocalLoss
# + V1.5 - Added MixedUp + CosineWarmUpLR
# 

# ### ToDo:
# + 

# ### References
# + [Face Attribute Prediction on CelebA benchmark with PyTorch Implementation](https://github.com/d-li14/face-attribute-prediction)
# + [PyTorch Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
# + [Albumentations](https://albumentations.ai/)
# + [Focal Loss](https://github.com/kornia/kornia/blob/master/kornia/losses/focal.py)
# + [Bag of Tricks](https://arxiv.org/abs/1812.01187)
# + [Torch ToolBox](https://github.com/PistonY/torch-toolbox)
# + [Fastai Course](https://www.youtube.com/watch?v=vnOpEwmtFJ8)

# ### Dependencies

# In[2]:


# conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
# conda install matplotlib
# conda install pandas
# conda install tqdm
# conda install -c conda-forge jupyterlab
# conda install -c conda-forge tensorboard
# conda install -c conda-forge protobuf # for tensorboard
# conda install nb_conda_kernels # auto add kernels

# conda install -c conda-forge imgaug
# conda install albumentations -c conda-forge
# conda install seaborn


# ## Setup/ Configuration

# ### Colab

# In[3]:


# you can choose to mount your Google Drive (optional)
import sys, os
if 'google.colab' in sys.modules:
    from google.colab import drive
    drive.mount('/content/drive')
    file_name = f'ai6126-project1-colab-v{nb_ver}.ipynb'
    print(file_name)
    import subprocess
    path_to_file = subprocess.check_output('find . -type f -name ' + str(file_name), shell=True).decode("utf-8")
    print(path_to_file)
    path_to_file = path_to_file.replace(file_name,"").replace('\n',"")
    os.chdir(path_to_file)
    get_ipython().system('pwd')


# ### Download Dataset (JUPYTER ONLY)

# In[4]:


import os, glob
local_download_path = '../data/celeba/img_align_celeba'
download_dataset = True
if os.path.exists(local_download_path):
    images = glob.glob(local_download_path + '/*.jpg')
    if len(images) == 202599:
        download_dataset = False
print(f"download celeba dataset: {download_dataset}")

if download_dataset:
    # create dataset root and enter it
    get_ipython().system('mkdir -p data/celeba')
    get_ipython().run_line_magic('cd', 'data/celeba')

    # we have prepared a backup of `img_align_celeba.zip` of Celeb-A dataset in the Dropbox
    # download it directly, or manually download the original file from Google Drive above
    get_ipython().system('wget https://www.dropbox.com/s/8kzo40fqx7nodat/img_align_celeba.zip')

    # unzip the downloaded file
    get_ipython().system('unzip -qq img_align_celeba.zip')
    get_ipython().system('rm -f img_align_celeba.zip')

    # change the directory back to the root
    get_ipython().run_line_magic('cd', '../..')
    get_ipython().system('ls')


# ## Implementation

# ### Imports

# In[5]:


import sys, os
import shutil
import time
import random
import numpy as np
import copy
from datetime import datetime
from distutils.dir_util import copy_tree #for recursive filecopying
import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter

import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt

import config
from celeba_dataset import CelebaDataset
import models
import losses
import schedulers
from utils import Logger, AverageMeter, Bar, ModelTimer, savefig, adjust_learning_rate, accuracy, print_attribute_acc, create_dir_ifne, add_weight_decay, mixup_data


# In[6]:


# check PyTorch version and cuda status
print(torch.__version__, torch.cuda.is_available())

# define device
device = torch.device("cuda:"+config.gpu_id if torch.cuda.is_available() else "cpu")
print(device)

ISJUPYTER = False
if 'ipykernel' in sys.modules:
    ISJUPYTER = True
    # set the backend of matplotlib to the 'inline' backend
    get_ipython().run_line_magic('matplotlib', 'inline')
    config.disable_tqdm = False
    
print(f"disable_tqdm: {config.disable_tqdm}")


# ### Seeding

# In[7]:


# set random seed for reproducibility
def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(1, 10000) # create random seed
        print(f'random seed used: {seed}')
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    if 'torch' in sys.modules:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
    
seed_everything(seed=config.manual_seed)


# ### Data Transform

# In[8]:


# Data augmentation and normalization for training
# Just normalization for validation and testing
def load_dataloaders(print_info=True, albu_transforms = True):
    if config.evaluate:
        phases = ['test']
    else:
        phases = ['train', 'val']

    attribute_names = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald', 
                       'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 
                       'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
                       'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open', 'Mustache', 
                       'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
                       'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings', 
                       'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']
    
    attributes_list = {
        'train': config.TRAIN_ATTRIBUTE_LIST,
        'val': config.VAL_ATTRIBUTE_LIST,
        'test': config.TEST_ATTRIBUTE_LIST
    }

    batch_sizes = {
        'train': config.train_batch,
        'val': config.test_batch,
        'test': config.test_batch
    }

    if not albu_transforms:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                         std=[0.229, 0.224, 0.225])
        data_transforms = {
            'train': transforms.Compose([
                transforms.CenterCrop((198, 158)), #new
                transforms.RandomHorizontalFlip(p=0.5),
                #transforms.RandomRotation(degrees=10), #new
                transforms.ToTensor(),
                normalize,
            ]),
            'val': transforms.Compose([
                transforms.CenterCrop((198, 158)), #new
                transforms.ToTensor(),
                normalize
            ]),
            'test': transforms.Compose([
                transforms.CenterCrop((198, 158)), #new
                transforms.ToTensor(),
                normalize
            ])
        }
    else:
        normalize_A = A.Normalize(mean=(0.485, 0.456, 0.406), 
                                  std=(0.229, 0.224, 0.225))
        data_transforms = {
            'train': A.Compose([
                #A.RandomResizedCrop(148, 148), # cuts out too much attributes, use centercrop instead
                A.CenterCrop(height=198, width=158),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, 
                                 rotate_limit=15, p=0.5), # AFFACT https://arxiv.org/pdf/1611.06158.pdf
                A.HorizontalFlip(p=0.5),
                #A.HueSaturationValue(hue_shift_limit=14, sat_shift_limit=14, val_shift_limit=14, p=0.5),
                #A.FancyPCA(alpha=0.1, p=0.5), #http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf
                #A.RandomBrightnessContrast(p=0.5),
                #A.GaussNoise(var_limit=10.0, p=0.5), 
                #A.GaussianBlur(p=0.1), # AFFACT https://arxiv.org/pdf/1611.06158.pdf
                #A.CoarseDropout(max_holes=1, max_height=74, max_width=74, 
                #               min_height=49, min_width=49, fill_value=0, p=0.2), #https://arxiv.org/pdf/1708.04896.pdf
                normalize_A,
                ToTensorV2(),
                
            ]),
            'val': A.Compose([
                #Rescale an image so that minimum side is equal to max_size 178 (shortest edge of Celeba)
                #A.SmallestMaxSize(max_size=178), 
                A.CenterCrop(height=198, width=158),
                normalize_A,
                ToTensorV2(),
            ]),
            'test': A.Compose([
                #A.SmallestMaxSize(max_size=178),
                A.CenterCrop(height=198, width=158),
                normalize_A,
                ToTensorV2(),
            ])
        }

    image_datasets = {x: CelebaDataset(config.IMG_DIR, attributes_list[x], 
                                       data_transforms[x], albu=albu_transforms) 
                      for x in phases}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                  batch_size=batch_sizes[x],
                                                  pin_memory=True, shuffle=(x == 'train'), 
                                                  num_workers=config.dl_workers) 
                   for x in phases}
    if print_info:
        dataset_sizes = {x: len(image_datasets[x]) for x in phases}
        print(f"Dataset sizes: {dataset_sizes}")
        
    if config.evaluate:
        class_names = image_datasets['test'].targets
    else:
        class_names = image_datasets['train'].targets
    print(f"Class Labels: {len(class_names[0])}")
    assert len(attribute_names) == len(class_names[0])
    return dataloaders, attribute_names


# ### Model Architecture Initialization

# In[9]:


model_names = sorted(name for name in models.__dict__
                     if callable(models.__dict__[name])) # and name.islower() and not name.startswith("__"))
print(f"Available Models: {model_names}")


# In[10]:


def create_model(arch, layers, device):
    print("=> creating model '{}'".format(config.arch))
    if arch.startswith('FaceAttrResNet'):
        model = models.__dict__[arch](resnet_layers = layers)
    elif arch.startswith('FaceAttrResNeXt'):
        model = models.__dict__[arch](resnet_layers = layers)
    elif arch.startswith('FaceAttrMobileNetV2'):
        model = models.__dict__[arch]()
    model = model.to(device)
    return model

model = create_model(config.arch, config.pt_layers, device)


# ### Criterion & Optimizer & Scheduler

# In[11]:


def get_criterion():
    criterion = nn.CrossEntropyLoss().to(device)
    if config.criterion == 'CE' and config.label_smoothing:
        criterion = losses.LabelSmoothingCrossEntropy(ls=config.label_smoothing).to(device) 
    elif config.criterion == 'FocalLoss':
        criterion = losses.FocalLossLS(alpha=0.25, gamma=3, reduction='mean', ls=config.label_smoothing).to(device) 
        
    if config.mixed_up > 0:
        criterion = losses.MixedUp(criterion).to(device) 
        
    return criterion

criterion = get_criterion()


# In[12]:


def get_optimizer(model, opt_name=config.optimizer, no_bias_bn_decay=config.no_bias_bn_decay):
    weight_decay = config.weight_decay
    if no_bias_bn_decay: #bag of tricks paper
        parameters = add_weight_decay(model, weight_decay)
        weight_decay = 0.
    else:
        parameters = model.parameters()
    
    optimizer = None
    if opt_name == 'SGD':
        optimizer = torch.optim.SGD(parameters, config.lr,
                                momentum=config.momentum,
                                weight_decay=weight_decay)
    elif opt_name == 'Adam':
        optimizer = torch.optim.Adam(parameters, config.lr,
                            weight_decay=weight_decay)
    elif opt_name == 'AdamW':
        optimizer = torch.optim.AdamW(parameters, config.lr,
                            weight_decay=weight_decay)
    return optimizer


# In[13]:


def get_scheduler(optimizer, steps_per_epoch, epochs):
    scheduler = None # Manual
    if config.scheduler == 'ReduceLROnPlateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',
                                                               factor=0.1,
                                                               patience=config.patience)
    elif config.scheduler == 'OneCycleLR': 
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, epochs=epochs,
                                                        steps_per_epoch=int(steps_per_epoch), 
                                                        anneal_strategy='cos') #https://arxiv.org/pdf/1708.07120.pdf
    elif config.scheduler == 'CosineWarmupLR':
        scheduler = schedulers.CosineWarmupLR(optimizer, batches=int(steps_per_epoch),
                                              epochs=epochs, base_lr=0.001, target_lr=0, warmup_epochs=5,
                                              warmup_lr = 0.01)
    
    return scheduler    


# ### Resume Checkpoint if any

# In[14]:


def format_checkpoint(modelname, opt_name, bias_decay=False, ckp_resume=None):
    best_prec1 = 0

    if ckp_resume and os.path.isfile(ckp_resume): 
        print(f"=> formatting model: {ckp_resume}")
        checkpoint = torch.load(ckp_resume)
        print(checkpoint['arch'])
        try:
            total_time = checkpoint['total_time']
        except:
            total_time = 0
        
        state = {
            'epoch': checkpoint['epoch'],
            'arch': modelname,
            'state_dict': checkpoint['state_dict'],
            'best_prec1': checkpoint['best_prec1'],
            'opt_name': opt_name,
            'optimizer' : checkpoint['optimizer'],
            'lr': checkpoint['lr'],
            'total_time': total_time,
            'bias_decay': bias_decay
        }
        torch.save(state, ckp_resume)
        
    else:
        raise
        
#format_checkpoint('FaceAttrResNeXt_50', 'SGD', True, ckp_resume=config.bestmodel_fname)


# In[15]:


def resume_checkpoint(device, ckp_logger_fname, ckp_resume=None):
    if not ckp_logger_fname:
        print("[W] Logger path not found.")
        raise

    start_epoch = 0
    best_prec1 = 0
    lr = config.lr
    
    if ckp_resume == '':
        ckp_resume = None
    
    if ckp_resume and os.path.isfile(ckp_resume): 
        print(f"=> resuming checkpoint: {ckp_resume}")
        checkpoint = torch.load(ckp_resume)
        
        try:
            total_time = checkpoint['total_time']
            model_timer = ModelTimer(total_time)
            print(f"=> model trained time: {model_timer}")
        except:
            print(f"=> old model")
            model_timer = ModelTimer()
        best_prec1 = checkpoint['best_prec1']
        print(f"=> model best val: {best_prec1}")
        
        start_epoch = checkpoint['epoch']
        print(f"=> model epoch: {start_epoch}")
        lr = checkpoint['lr']

        print(f"=> resuming model: {checkpoint['arch']}")
        model = create_model(checkpoint['arch'].split('_')[0], 
                             int(checkpoint['arch'].split('_')[1]), 
                             device)
        model.load_state_dict(checkpoint['state_dict'])
        
        print(f"=> resuming optimizer: {checkpoint['opt_name']}")
        bias_decay = True
        if checkpoint['bias_decay']:
            bias_decay = checkpoint['bias_decay']
            
        optimizer = get_optimizer(model, checkpoint['opt_name'], bias_decay)
        if optimizer:
            optimizer.load_state_dict(checkpoint['optimizer'])
        logger = Logger(ckp_logger_fname, title=model.name, resume=True)
        
    else:
        print(f"=> restarting training: {ckp_resume}")
        model_timer = ModelTimer()
        model = create_model(config.arch, config.pt_layers, device)
        optimizer = get_optimizer(model)
        logger = Logger(ckp_logger_fname, title=model.name)
        logger.set_names(['Learning Rate', 'Train Loss', 'Valid Loss', 'Train Acc.', 'Valid Acc.'])
              
    return best_prec1, model_timer, lr, start_epoch, logger, model, optimizer


# In[16]:


def load_inference_model(device, ckp_resume):
    if not (ckp_resume and os.path.isfile(ckp_resume)):
        print("[W] Checkpoint not found for inference.")
        raise 
    
    print(f"=> loading checkpoint: {ckp_resume}")
    checkpoint = torch.load(ckp_resume)
    try:
        total_time = checkpoint['total_time']
        model_timer = ModelTimer(total_time)
        print(f"=> model trained time: {model_timer}")
    except:
        print(f"=> old model")
    best_prec1 = checkpoint['best_prec1']
    print(f"=> model best val: {best_prec1}")
    start_epoch = checkpoint['epoch']
    print(f"=> model epoch: {start_epoch}")

    print(f"=> resuming model: {checkpoint['arch']}")
    model = create_model(checkpoint['arch'].split('_')[0], 
                         int(checkpoint['arch'].split('_')[1]), 
                         device)
    model.load_state_dict(checkpoint['state_dict'])
              
    return best_prec1, model


# ## Train & Validate Function

# In[17]:


def train(train_loader, model, criterion, optimizer):
    bar = Bar('Processing', max=len(train_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(40)]
    top1 = [AverageMeter() for _ in range(40)]

    # switch to train mode
    model.train()

    end = time.time()
    for i, (X, y) in enumerate(tqdm(train_loader, disable=config.disable_tqdm)):
        # measure data loading time
        data_time.update(time.time() - end)

        # Overlapping transfer if pinned memory
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if config.mixed_up > 0:
            X, y, lam = mixup_data(X, y, alpha=config.mixed_up)
            criterion.set_lambda(lam)
    
        # compute output
        output = model(X)
        # measure accuracy and record loss
        loss = []
        prec1 = []
        for j in range(len(output)): 
            if config.mixed_up > 0:
                labels = y[:, :, j]
                actual_labels = y[0, :, j] * lam + y[1, :, j] * (1-lam)
            else:
                labels = y[:, j]
                actual_labels = y[:, j]
            crit = criterion(output[j], labels)
            loss.append(crit)
            prec1.append(accuracy(output[j], actual_labels, topk=(1,), mixedup=config.mixed_up))
            losses[j].update(loss[j].detach().item(), X.size(0))
            top1[j].update(prec1[j][0].item(), X.size(0))
            
        losses_avg = [losses[k].avg for k in range(len(losses))]
        top1_avg = [top1[k].avg for k in range(len(top1))]
        loss_avg = sum(losses_avg) / len(losses_avg)
        prec1_avg = sum(top1_avg) / len(top1_avg)

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss_sum = sum(loss)
        loss_sum.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        print_line = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                        batch=i + 1,
                        size=len(train_loader),
                        data=data_time.avg,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=loss_avg,
                        top1=prec1_avg,
                        )
        if not config.disable_tqdm and (i+1)% 100 == 0:
            print(print_line)
        bar.suffix  = print_line
        bar.next()
    bar.finish()
    return (loss_avg, prec1_avg)


# In[18]:


def validate(val_loader, model, criterion):
    bar = Bar('Processing', max=len(val_loader))

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = [AverageMeter() for _ in range(40)]
    top1 = [AverageMeter() for _ in range(40)]

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (X, y) in enumerate(tqdm(val_loader, disable=config.disable_tqdm)):
            # measure data loading time
            data_time.update(time.time() - end)

            # Overlapping transfer if pinned memory
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            
            # compute output
            output = model(X)
            # measure accuracy and record loss
            loss = []
            prec1 = []
            for j in range(len(output)):
                if config.mixed_up > 0:
                    loss.append(criterion(output[j], y[:, j], mixed=False))
                else:
                    loss.append(criterion(output[j], y[:, j]))
                prec1.append(accuracy(output[j], y[:, j], topk=(1,)))
                
                losses[j].update(loss[j].detach().item(), X.size(0))
                top1[j].update(prec1[j][0].item(), X.size(0))
            losses_avg = [losses[k].avg for k in range(len(losses))]
            top1_avg = [top1[k].avg for k in range(len(top1))]
            loss_avg = sum(losses_avg) / len(losses_avg)
            prec1_avg = sum(top1_avg) / len(top1_avg)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # plot progress
            print_line = '({batch}/{size}) Data: {data:.3f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | top1: {top1: .4f}'.format(
                            batch=i + 1,
                            size=len(val_loader),
                            data=data_time.avg,
                            bt=batch_time.avg,
                            total=bar.elapsed_td,
                            eta=bar.eta_td,
                            loss=loss_avg,
                            top1=prec1_avg,
                            )

            bar.suffix  = print_line
            bar.next()  

    if not config.disable_tqdm:
        print(print_line)        
    bar.finish()
    return (loss_avg, prec1_avg, top1)


# ## Main Function

# In[19]:


def trainer(dataloaders, model, criterion, optimizer, logger, start_epoch, best_prec1, run_name, model_timer):
    # visualization
    writer = SummaryWriter(os.path.join(config.tensorboard_dir, run_name))
    
    scheduler = get_scheduler(optimizer, len(dataloaders['train']), config.epochs-start_epoch)
    
    stagnant_val_loss_ctr = 0
    min_val_loss = 1.
    
    for epoch in range(start_epoch, config.epochs):
        model_timer.start_epoch_timer()
        if not scheduler:
            lr = adjust_learning_rate(optimizer, config.lr_decay, epoch, gamma=config.gamma, step=config.step,
                                     total_epochs=config.epochs, turning_point=config.turning_point,
                                     schedule=config.schedule)
        else:
            lr = optimizer.param_groups[0]['lr']

        print('\nEpoch: [%d | %d] LR: %.16f' % (epoch + 1, config.epochs, lr))

        # train for one epoch
        train_loss, train_acc = train(dataloaders['train'], model, criterion, optimizer)

        # evaluate on validation set
        val_loss, prec1, _ = validate(dataloaders['val'], model, criterion)
        
        if scheduler:
            scheduler.step(None if config.scheduler != 'ReduceLROnPlateau' else val_loss)
            
        # append logger file
        logger.append([lr, train_loss, val_loss, train_acc, prec1])

        # tensorboardX
        writer.add_scalar('learning rate', lr, epoch + 1)
        writer.add_scalars('loss', {'train loss': train_loss, 'validation loss': val_loss}, epoch + 1)
        writer.add_scalars('accuracy', {'train accuracy': train_acc, 'validation accuracy': prec1}, epoch + 1)

        is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)
        model_timer.stop_epoch_timer()
        model.save_ckp({
            'epoch': epoch + 1,
            'arch': model.name,
            'state_dict': model.state_dict(),
            'best_prec1': best_prec1,
            'opt_name': config.optimizer,
            'optimizer' : optimizer.state_dict(),
            'lr': lr,
            'total_time': model_timer.total_time,
            'bias_decay': config.no_bias_bn_decay,
        }, is_best, config.checkpoint_fname,config.bestmodel_fname)
        
        if config.early_stopping:
            if is_best:
                stagnant_val_loss_ctr = 0
                min_val_loss = val_loss
            elif val_loss >= min_val_loss:
                stagnant_val_loss_ctr += 1
                if (epoch+1) > config.es_min and stagnant_val_loss_ctr >= config.es_patience: 
                    break
            else:
                stagnant_val_loss_ctr = 0
                min_val_loss = val_loss

    logger.close()
    logger.plot()
    save_path = None
    if config.train_saveplot:
        save_path = os.path.join(config.CHECKPOINT_DIR, "losses.jpg")
    logger.plot_special(save_path)
    savefig(config.train_plotfig)
    writer.close()

    print('Best accuracy:')
    print(best_prec1)
    return model_timer


# In[20]:


def get_run_name_time(model, criterion, optimizer, comments, start_epoch=0):
    try:
        if criterion.name:
            p_criterion = criterion.name
    except:
        p_criterion = 'CE'

    p_optimizer = f'{str(optimizer).split("(")[0].strip()}'
    p_scheduler = f'lr{config.lr}_wd{config.weight_decay}'
    if config.scheduler == 'Manual':
        p_scheduler += f'_{config.lr_decay}'
        if config.lr_decay == 'step':
            p_scheduler += f'_g{config.gamma}_sp{config.step}'
        elif config.lr_decay == 'linear2exp':
            p_scheduler += f'_g{config.gamma}_tp{config.turning_point}'
        elif config.lr_decay == 'schedule':
            p_scheduler += f'_g{config.gamma}_sch{config.schedule}'
    else: 
        p_scheduler += f'_{config.scheduler}'
    
    run_name = f'{model.name}_{config.manual_seed}_s{start_epoch}e{config.epochs}_'                 + f'tb{config.train_batch}_vb{config.test_batch}_'                 + f'{p_criterion}_{p_optimizer}_'                 + f'{comments}_'                 + f'{p_scheduler}'
    
    run_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(run_name, run_time)
    return run_name, run_time


# ## Training Loop

# In[ ]:


# config.epoch = 1
#model = create_model(device)
dataloaders, attribute_names = load_dataloaders()
criterion = get_criterion()
#optimizer = get_optimizer(model)

print(f"=> Training model: {not config.evaluate}")
if config.evaluate:
    best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
    test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
    print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
    attr_acc = print_attribute_acc(top1, attribute_names)
    if config.test_preds_fname:
        json.dump(attr_acc, open(config.test_preds_fname,'w'))
else:
    best_prec1, model_timer, lr, start_epoch, logger, model, optimizer = resume_checkpoint(device, config.ckp_logger_fname, config.ckp_resume)
    run_name, run_time = get_run_name_time(model, criterion, optimizer, comments, start_epoch)
    mtimer = trainer(dataloaders, model, criterion, optimizer, logger, start_epoch, best_prec1, run_name, model_timer)
    print(f"=> Model trained time: {mtimer}")


# ## Testing Loop

# In[ ]:


if not config.evaluate:
    config.evaluate = True
    #model = create_model(device)
    dataloaders, attribute_names = load_dataloaders()
    criterion = get_criterion()
    #optimizer = get_optimizer(model)
    
    best_prec1, model = load_inference_model(device, config.bestmodel_fname) # checkpoint_fname bestmodel_fname
    #best_prec1, mtimer, _, _, logger, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.checkpoint_fname)
    test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
    print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
    attr_acc = print_attribute_acc(top1, attribute_names)
    if config.test_preds_fname:
        json.dump(attr_acc, open(config.test_preds_fname,'w'))
#     best_prec1, mtimer, _, _, _, = resume_checkpoint(model, optimizer, config.ckp_logger_fname, config.bestmodel_fname)# config.bestmodel_fname  config.checkpoint_fname
#     #print(model)
#     test_loss, prec1, top1 = validate(dataloaders['test'], model, criterion)
#     print(f"=> Best test accuracy: {prec1}, Model val acc: {best_prec1}")
#     print_attribute_acc(top1, attribute_names)


# ## Save & Backup

# In[ ]:


if ISJUPYTER:
    # Wait for notebook to save
    get_ipython().run_line_magic('autosave', '1')
    time.sleep(150)
    get_ipython().run_line_magic('autosave', '120')


# In[ ]:


def backup_everything(run_time, run_name, title, backup_nb=ISJUPYTER):
    # backup checkpoints
    print(f"=> backing up checkpoints... ")
    run_dir = os.path.join(config.BACKUP_DIR, run_name, run_time)
    create_dir_ifne(run_dir)
    fromDirectory = config.CHECKPOINT_DIR
    toDirectory = run_dir
    copy_tree(fromDirectory, toDirectory)
    
    if backup_nb:
        print(f"=> backing up notebook... ")
        # backup notebook html
        nb_name = title + '.ipynb'
        html_name = title + '.html'
        save_name = os.path.join(run_dir, html_name)
        get_ipython().system('jupyter nbconvert --to html $nb_name')
        shutil.move(html_name, save_name)
    
backup_everything(run_time, run_name, title, backup_nb=ISJUPYTER)


# In[ ]:


if config.auto_hibernate and False:
    os.system('shutdown -h')


# In[ ]:





# In[ ]:




