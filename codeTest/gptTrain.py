
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
from networks.lr_schedule import *
from metrics.metric import *
from utils.plot import *
from config import config
from dataset.Aug import *



def train():

    # 超参数
    batch_size = config.batch_size
    image_size = config.width
    patch_size = 32
    num_patches = (image_size // patch_size) ** 2
    latent_dim = 128
    num_epochs = config.num_epochs
    learning_rate = config.lr


    # 数据增强
    transform_train = transforms.Compose([
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
    ])

    vae = VAE(latent_dim=latent_dim)
    # 新建文件夹用于保存重建图像
    if not os.path.exists('./recon_images'):
        os.makedirs('./recon_images')


    dst_train = RSDataset('./data/train.txt', width=config.width, 
                          height=config.height, transform=transform_train)
    train_dataloader = DataLoader(dst_train, shuffle=True, batch_size=batch_size, num_workers=config.num_workers)

    # 训练模型
    for epoch in range(num_epochs):
        lr = step_lr(epoch)
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, vae.parameters()), 
                                        lr=lr, betas=(0.9, 0.999), weight_decay=0.0002)
        # vae.train()
        for batch_idx, (data, _) in enumerate(train_dataloader):
            print(1)
            optimizer.zero_grad()
            
            # 数据增强：将原图分成小块并打乱顺序，然后重新拼接成大图
            
            patches = data.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            patches = patches.permute(1, 2, 3, 0, 4, 5).reshape(num_patches, -1, 3, patch_size, patch_size)
            idx = torch.randperm(num_patches)
            patches = patches[idx]
            data_aug = patches.reshape(-1, 3, patch_size, patch_size)

            
            # 将数据送入模型
            recon_data, mu, logvar = vae(data_aug)
            loss = vae_loss(recon_data, data_aug, mu, logvar)
            loss.backward()
            optimizer.step()
            
            # 保存每一轮的重建图像
            if batch_idx % 100 == 0:
                with torch.no_grad():
                    vae.eval()
                    recon_images, _, _ = vae(data_aug)
                    recon_images = recon_images.reshape(-1, 3, patch_size, patch_size)
                    for i in range(recon_images.shape[0]):
                        img = np.transpose(recon_images[i].cpu().numpy(), (1, 2, 0))
                        img = np.clip(img, 0, 1)
                        img_path = f'recon_images/epoch_{epoch+1}_batch_{batch_idx}_image_{i}.jpg'
                        cv2.imwrite(img_path, img * 255.0)
                
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item()}')

if __name__== '__main__':
    train()