# 以重建任务为例，训练卷积自编码器，输入为遥感数据
# Path: trainAE.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from networks.AE import AE
from utils import trainAE
from utils import testAE

