import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import numpy as np
import torchvision.models as models
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm


# 定义超参数
latent_dim=128



# 定义编码器类，使用models.ResNet18
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * 7 * 7, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.encoder(x)

# 定义解码器，使用转置卷积层，封装成类
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (256, 7, 7)),
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)



# 定义自编码器类，继承nn.Module
class VAE(nn.Module):
    def __init__(self, encoder, decoder, latent_dim):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = latent_dim
        self.mu_layer = nn.Linear(256 * 7 * 7, latent_dim)
        self.logvar_layer = nn.Linear(256 * 7 * 7, latent_dim)

    def encode(self, x):
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        x = self.decoder(z)
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar


#损失函数
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = nn.MSELoss()(recon_x, x)
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + kl_div







# def train_vae():
#     # 定义数据预处理方式
#     transform = transforms.Compose([
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#     ])

#     # 定义数据集
#     dst_train = RSDataset('./data/train.txt', width=config.width, 
#                           height=config.height, transform=config.transform)
#     train_dataloader = DataLoader(dst_train, shuffle=True, batch_size=config.batch_size, num_workers=config.num_workers)

#     # 定义编码器、解码器、自编码器
#     encoder = Encoder(latent_dim)
#     decoder = Decoder(latent_dim)
#     vae = VAE(encoder, decoder, latent_dim)

#     # 定义优化器
#     optimizer = optim.Adam(vae.parameters(), lr=1e-3)

#     # 训练代码




    # for epoch in range(10):
    #     for i, (x, _) in enumerate(train_dataloader):
    #         optimizer.zero_grad()
    #         x_recon, mu, logvar = vae(x)
    #         loss = vae_loss(x_recon, x, mu, logvar)
    #         loss.backward()
    #         optimizer.step()
    #         if i % 10 == 0:
    #             print('Epoch: {}, Iter: {}, Loss: {:.3f}'.format(epoch, i, loss.item()))
    #     # 保存重建图像
    #     torchvision.utils.save_image(x_recon, 'recon_{}.png'.format(epoch))