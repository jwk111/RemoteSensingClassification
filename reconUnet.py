#用自编码器重建单张图像

# 载入预训练权重，输入图片，输出重建图像
import sys
sys.path.append('/home/jwk/Project/remoteSensing/Remote-Sensing-Image-Classification')
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
from dataset.dataset import *
from dataset.Aug import *
from networks.unet import *
from config import *

from PIL import Image
import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda" if use_cuda else "cpu")   # use CPU or GPU

# 加载预训练的自编码器模型
model = Net()
state_dict = torch.load('pretrainModel/Unet_AE_lr_1e5_30epoch/Unet_AE_epoch20.pth')
model_dict = model.state_dict()
model.load_state_dict(model_dict)

# 将模型设置为评估模式
# model.eval()

# 加载测试图片

img = Image.open("denseresidential01.tif")

# 转换为张量并归一化
transform = transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor()])
img_tensor = transform(img)

# img_tensor = img_tensor.to(device)

# 经过自编码器进行图像重建

# 将输入图像输入到自编码器模型中，得到重建图像
with torch.no_grad():
    model.eval()
    output = model(img_tensor.unsqueeze(0))
    output = output.squeeze().cpu().numpy().transpose((1, 2, 0))


# 将输入图像和重建图像可视化
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title('Input Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(output)
plt.title('Reconstructed Image')
plt.axis('off')
plt.savefig('recon.png')
plt.show()


