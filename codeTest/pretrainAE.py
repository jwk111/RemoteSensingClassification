# -*- coding:utf-8 -*-
import sys
sys.path.append('/home/jwk/Project/remoteSensing/Remote-Sensing-Image-Classification')
import os, argparse, time

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms, models
from torch.nn.parallel.data_parallel import data_parallel
import torchvision
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from dataset.dataset import *
from networks.network import *
from networks.lr_schedule import *
from networks.PuzzleAE import *
from metrics.metric import *
from utils.plot import *
from config import config





def train():

    # model
    if config.model == 'VAE_ResNet18':
        model = VAE_ResNet18(z_dim=256)
    elif config.model == 'VAE_ResNet50':
        model = VAE_ResNet50(z_dim=256)
    else:
        print('ERROR: No model {}!!!'.format(config.model))
    print(model)
    # model = torch.nn.DataParallel(model)
    model.cuda()
    
    # freeze layers
    if config.freeze:
        for p in model.backbone.layer1.parameters(): p.requires_grad = False
        for p in model.backbone.layer2.parameters(): p.requires_grad = False
        for p in model.backbone.layer3.parameters(): p.requires_grad = False
        # for p in model.backbone.layer4.parameters(): p.requires_grad = False


    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    # 图片切成小片，作为模型的输入


    # train data
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.RandomCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ColorJitter(0.05, 0.05, 0.05),
                                    transforms.RandomRotation(10),
                                    transforms.Resize((config.width, config.height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_train = RSDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_train = DataLoader(dst_train, shuffle=True, batch_size=int(config.batch_size), num_workers=config.num_workers)

    # validation data
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./data/valid.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=int(config.batch_size/2), num_workers=config.num_workers)

    # log
    if not os.path.exists('./log'):
        os.makedirs('./log')
    log = open('./log/log.txt', 'a')

    log.write('-'*30+'\n')
    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
               config.model, config.num_classes, config.num_epochs, config.lr, 
               config.width, config.height, config.iter_smooth))

    # load checkpoint
    if config.resume:
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))

    # train
    for epoch in range(config.num_epochs):
        # ep_start = time.time()

        # adjust lr
        # lr = half_lr(config.lr, epoch)
        lr = step_lr(epoch)

        # optimizer
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                                     lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)

        l_sum = 0
        i=0
        for x,y in dataloader_train:
            # x = torch.sigmoid(x).cuda()
            x = x.cuda()
            # print(x.requires_grad)
            optimizer.zero_grad()
            recon_x, mu, logvar = model.forward(x)
            #一次性打印上述三个值的维度
            # print(recon_x.shape, mu.shape, logvar.shape)
            loss = vae_loss(recon_x, x, mu, logvar).cuda()
            l_sum += loss
            loss.backward()
            optimizer.step()
        # print("loss\n", l_sum)
        # print(epoch, "\n")
        print('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f'
                %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                lr, l_sum))
        log.write('Epoch [%d/%d], Iter [%d/%d], lr: %f, Loss: %.4f\n'
                    %(epoch+1, config.num_epochs, i+1, len(dst_train)//config.batch_size, 
                    lr,l_sum))
    log.close()

    i = 0
    with torch.no_grad():
        for t_img, y in dataloader_valid:
            t_img = Variable(t_img).cuda()
            result, mu, logvar = model.forward(t_img)
            utils.save_image(result.data, str(i) + '.png', normalize=True)
            i += 1
# validation


if __name__ == '__main__':
    train()


##########################