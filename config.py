# -*- coding:utf-8 -*- 
class DefaultConfigs(object):
    data_root = '/home/jwk/Project/remoteSensing/datasets/UCMerced_LandUse/Images' # 数据集的根目录
    # model = 'VAE_ResNet18' # ResNet18, ResNet34, ResNet50, ResNet101, ResNet152 使用的模型
    # model = 'ResNet_VAE'
    model = 'Unet_AE'
    freeze = False # 是否冻结卷基层

    seed = 1000 # 固定随机种子
    num_workers = 10 # DataLoader 中的多线程数量
    num_classes = 21 # 分类类别数
    num_epochs = 100 # 训练的epoch数
    batch_size = 30 
    lr = 1e-5 # 初始lr
    width = 256 # 输入图像的宽
    height = 256 # 输入图像的高
    iter_smooth = 20 # 打印&记录log的频率
    modelSaveEpoch = 20 # 模型保存的间隔

    resume = False #
    # checkpoint = 'ResNet152.pth' # 训练完成的模型名
    checkpoint = 'Unet_AE.pth' # 训练完成的模型名

config = DefaultConfigs()
