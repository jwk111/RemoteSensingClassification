# AE pretrained on UCMerced dataset
import sys
sys.path.append('/home/jwk/Project/remoteSensing/Remote-Sensing-Image-Classification')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
import os
from networks.unet import *
from dataset.dataset import *
from dataset.Aug import *
from config import *
import time
from utils.plot import *


# 建立模型文件夹，存放模型、log和图像，文件夹名字包含运行时间
modelDir = config.model + '_' + time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
if not os.path.exists('./pretrainModel/' + modelDir):
    os.makedirs('./pretrainModel/' + modelDir)

modelPath = './pretrainModel/' + modelDir

CNN_fc_hidden1, CNN_fc_hidden2 = 1024, 1024
CNN_embed_dim = 256     # latent dim extracted by 2D CNN
res_size = 224        # ResNet image size
dropout_p = 0.2       # dropout probability

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

#自监督预训练代码，任务为图像重建，保存重建图像在文件夹中
def train():
    # 定义数据预处理方式
    # transform = transforms.Compose([transforms.Resize([res_size, res_size]),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # MAE
    transform = transforms.Compose([
            transforms.RandomResizedCrop(res_size, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # 定义数据集

    # train_loader
    dst_train = RSDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    train_dataloader = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)

    # validation data
    transform = transforms.Compose([transforms.Resize((res_size, res_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                                         std=[0.229, 0.224, 0.225])])
    dst_valid = RSDataset('./data/valid.txt', width=config.width, 
                          height=config.height, transform=transform)
    dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=int(config.batch_size/2), num_workers=config.num_workers)

    # log
    log = open(modelPath + '/log.txt', 'a')

    log.write('-'*30+'\n')
    log.write('model:{}\nnum_classes:{}\nnum_epoch:{}\nlearning_rate:{}\nim_width:{}\nim_height:{}\niter_smooth:{}\n'.format(
               config.model, config.num_classes, config.num_epochs, config.lr, 
               config.width, config.height, config.iter_smooth))

    if not os.path.exists('./checkpoints'):
        os.makedirs('./checkpoints')

    # load checkpoint
    if config.resume:
        model = torch.load(os.path.join('./checkpoints', config.checkpoint))


    # 定义编码器、解码器、自编码器
    model = Net().to(device)

    print("Using", torch.cuda.device_count(), "GPU!")
    model_params = list(model.parameters())
    optimizer = torch.optim.Adam(model_params, lr=config.lr)

    # 定义优化器
    # optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # 新建文件夹用于保存重建图像
    if not os.path.exists('./recon_images'):
        os.makedirs('./recon_images')

    # 使用dataset.Aug中的函数，将图像切成小块并进行数据增强，再重新拼接成原图尺寸的图像

    sum = 0
    train_loss_sum = 0
    train_draw_loss = []
    val_draw_loss = []
    criterion = torch.nn.MSELoss()

    # 训练模型
    for epoch in range(config.num_epochs):
        ep_start = time.time()

        model.train()
        for i, (data, _) in enumerate(train_dataloader):
            data = data.cuda()
            out = model(data)
            train_loss = criterion(out, data)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            train_loss = train_loss.data.cpu().numpy()
            # train_loss_sum += train_loss.data.cpu().numpy()
            # sum += 1
            if (i+1) % config.iter_smooth == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, config.num_epochs , i+1, len(train_dataloader), train_loss))
                log.write('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}\n'
                        .format(epoch+1, config.num_epochs , i+1, len(train_dataloader), train_loss))
                sum = 0
                train_loss_sum = 0

        train_draw_loss.append(train_loss)    

        epoch_time = (time.time() - ep_start) / 60.
        if epoch % 1 == 0 and epoch < config.num_epochs:
            val_time_start = time.time()
            val_loss = eval(model, dataloader_valid)
            val_draw_loss.append(val_loss)
            val_time = (time.time() - val_time_start) / 60.
            print('Epoch [{}/{}], Val Loss: {:.4f}, Val Time: {:.2f} s'
                    .format(epoch+1, config.num_epochs, val_loss, val_time))
            print('epoch time: {:.2f} s'.format(epoch_time))
            if epoch % config.modelSaveEpoch == 0:
                torch.save(model,'{}/{}_epoch{}.pth'.format(modelPath, config.model, epoch))
            log.write('Epoch [{}/{}], Val Loss: {:.4f}, Val Time: {:.2f} s\n'
                    .format(epoch+1, config.num_epochs, val_loss, val_time))
    draw_loss(train_draw_loss, val_draw_loss, modelPath)
    log.write('-'*30+'\n')
    log.close()



def eval(model, dataloader_valid):
    sum = 0
    val_loss_sum = 0
    criterion = torch.nn.MSELoss()
    model.eval()
    for i, (data, _) in enumerate(dataloader_valid):
        data = data.cuda()
        X_reconst = model(data)
        loss = criterion(X_reconst, data)
        loss = loss.data.cpu().numpy()
    #     val_loss_sum += loss.data.cpu().numpy()
    #     sum += 1
    # avg_loss = val_loss_sum / sum
    return loss



# 输入一张图片，用预训练好的模型重建图像，并将输入图像和重建图像作比较
def draw_recon(model, dataloader_valid,modelPath):
    model.eval()
    for i, (data, _) in enumerate(dataloader_valid):
        data = data.cuda()
        X_reconst, z, mu, logvar = model(data)
        X_reconst = X_reconst.data.cpu().numpy()
        data = data.data.cpu().numpy()
        for j in range(len(data)):
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.imshow(data[j].transpose(0,1,2))
            plt.title('input image')
            plt.subplot(1, 2, 2)
            plt.imshow(X_reconst[j].transpose(0,1,2))
            plt.title('reconstructed image')
            if not os.path.exists(modelPath + '/recon_images'):
                os.makedirs(modelPath + '/recon_images')
            plt.savefig(modelPath + '/recon_images/{}_{}.png'.format(i, j))
            plt.close()
        break


    

if __name__ == '__main__':
    train()


    # modelPath = '/home/jwk/Project/remoteSensing/Remote-Sensing-Image-Classification/pretrainModel/ResNet_VAE_100epoch'
    # model = torch.load(modelPath + '/ResNet_VAE_epoch80.pth')
    # transform = transforms.Compose([transforms.Resize((res_size, res_size)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize(mean=[0.485, 0.456, 0.406], 
    #                                                      std=[0.229, 0.224, 0.225])])
    # dst_valid = RSDataset('./data/valid.txt', width=config.width, 
    #                       height=config.height, transform=transform)
    # dataloader_valid = DataLoader(dst_valid, shuffle=False, batch_size=int(config.batch_size), num_workers=config.num_workers)
    # draw_recon(model, dataloader_valid,modelPath)

    #输入一张图片，作为输入，用预训练好的模型重建图像，并将输入图像和重建图像作比较



