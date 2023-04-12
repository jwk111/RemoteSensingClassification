import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import os
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision
from networks.PuzzleAE import *
epoch_num = 100
batch_size = 16
vae = VAE_ResNet18(z_dim=256).cuda()
optimizer = optim.Adam(vae.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

root = "./dataset/"



transform = transforms.Compose([transforms.Resize([224, 224]),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                # gray -> GRB 3 channel (lambda function)
                                transforms.Normalize(mean=[0.0, 0.0, 0.0],
                                                     std=[1.0, 1.0, 1.0])])  # for grayscale images

# MNIST dataset (images and labels)
MNIST_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
MNIST_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)

# Data loader (input pipeline)
train_iter = torch.utils.data.DataLoader(dataset=MNIST_train_dataset, batch_size=batch_size, shuffle=True)
test_iter = torch.utils.data.DataLoader(dataset=MNIST_test_dataset, batch_size=batch_size, shuffle=False)

for epoch in range(0, epoch_num):
    l_sum = 0
    scheduler.step()
    for x, y in train_iter:
        # x = torch.sigmoid(x).cuda()
        x = x.cuda()
        # print(x.requires_grad)
        optimizer.zero_grad()
        recon_x, mu, logvar = vae.forward(x)
        loss = loss_func(recon_x, x, mu, logvar)
        l_sum += loss
        loss.backward()
        optimizer.step()
    print("loss\n", l_sum)
    print(epoch, "\n")

i = 0
with torch.no_grad():
    for t_img, y in test_iter:
        t_img = Variable(t_img).cuda()
        result, mu, logvar = vae.forward(t_img)
        utils.save_image(result.data, str(i) + '.png', normalize=True)
        i += 1