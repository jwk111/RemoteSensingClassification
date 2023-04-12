# gptTrain2.py
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
from networks.gptNet import *
from dataset.dataset import *
from dataset.Aug import *
from config import *


#自监督预训练代码，任务为图像重建，保存重建图像在文件夹中
def train():
    # 定义数据预处理方式
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 定义数据集
    dst_train = RSDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform)
    train_dataloader = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)

    # 定义编码器、解码器、自编码器
    encoder = Encoder(latent_dim)
    decoder = Decoder(latent_dim)
    vae = VAE(encoder, decoder, latent_dim)

    # 定义优化器
    optimizer = optim.Adam(vae.parameters(), lr=1e-3)

    # 新建文件夹用于保存重建图像
    if not os.path.exists('./recon_images'):
        os.makedirs('./recon_images')

    # 使用dataset.Aug中的函数，将图像切成小块并进行数据增强，再重新拼接成原图尺寸的图像



    # 训练模型
    for epoch in range(config.num_epochs):
        for batch_idx, (data, _) in enumerate(train_dataloader):
            data = data.cuda()
            recon, mu, logvar = vae(data)
            loss = vae.loss(recon, data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                      .format(epoch+1, config.num_epochs , batch_idx+1, len(train_dataloader), loss.item()))

                # 保存重建图像
                torchvision.utils.save_image(recon, './recon_images/recon_image_{}_{}.png'.format(epoch+1, batch_idx+1))
         
if __name__ == '__main__':
    train()


