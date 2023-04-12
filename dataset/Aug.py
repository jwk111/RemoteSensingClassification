## 图像切成小块并进行数据增强，再重新拼接成原图尺寸的图像
import torch
from torchvision import transforms
import random
import numpy as np
from PIL import Image

# 设置随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)


def split_image(image, patch_size):
    """将图像切成多个小块"""
    patches = []
    height, width = image.size
    for y in range(0, height - patch_size + 1, patch_size):
        for x in range(0, width - patch_size + 1, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)
    return patches

def augment_and_shuffle(patches):
    """对小块进行数据增强和打乱顺序"""
    # 其他数据增强操作，例如随机旋转、随机裁剪等
    random.shuffle(patches)
    return patches

def reassemble_image(patches, image_size):
    """将处理后的小块重新拼接成原图尺寸的图像"""
    image = Image.new('RGB', image_size)
    patch_size = patches[0].size[0]
    width = image_size[0] // patch_size
    for i, patch in enumerate(patches):
        x = (i % width) * patch_size
        y = (i // width) * patch_size
        image.paste(patch, (x, y))
    return image



