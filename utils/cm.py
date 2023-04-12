import numpy as np
import matplotlib.pyplot as plt
import argparse
from matplotlib import rcParams


# 训练集：仿真数据 01训03验  测试：真实数据试验2 4类 Resnet50
cm = np.array([[31,68,1,0],[19,78,3,0],[50,14,35,0],[45,34.5,20,0]])
# 训练集：仿真数据 01训03验  测试：真实数据试验1全部 4类 Resnet50
cm = np.array([[12.5,79.23,8.27,0],[5.48,83.68,10.84,0],[55.28,43.06,1.67,0],[71.71,16.20,12.07,0.03]])
# 训练集：仿真数据 01训03验  测试：真实数据试验1切片 4类 Resnet50
cm = np.array([[4,92,0,0],[2.5,89,0,0],[16.1,79,0,0],[46.63,47,0,0]])
# 训练集：仿真数据 真实数据试验1  测试：真实数据试验2 4+3类 Densenet
cm = np.array([[59,0,18,22],[43,17,19,20],[56,17,19,20],[56,4,7,32]])
# 训练集：仿真数据 真实数据试验1  测试：真实数据试验2 4类 Densenet
cm = np.array([[33.9,0,7,59],[46.4,0,19.25,34],[6,40,9.75,43.8],[56.6,4,7.4,32]])
# # 训练集：仿真数据 01训03验  测试：真实数据试验2 4+3类 Resnet50
# cm = np.array([[26,72,2,0],[28,51,21,0],[25,47,28,0],[45,34,20,0]])

labels = ["DT","ZTQ","YR1","YR2"]
def show_confusion_matrix(classes, confusion_matrix, work_dir):
    plt.figure()
    proportion = []
    length = len(confusion_matrix)
    for i in confusion_matrix:
        for j in i:
            temp = j / (np.sum(i))
            proportion.append(temp)

    pshow = []  # 百分比(行遍历)
    for i in proportion:
        pt = "%.2f%%" % (i * 100)
        pshow.append(pt)
    proportion = np.array(proportion).reshape(length, length)   # reshape(列的长度，行的长度)
    pshow = np.array(pshow).reshape(length, length)
    tick_marks = np.arange(len(classes))
    # Yclass是classes的逆序
    Yclass = []
    for i in range(len(classes)):
        Yclass.append(classes[len(classes) - 1 - i])
        
    plt.xticks(tick_marks, classes, fontsize=12, rotation=20)
    plt.yticks(tick_marks, Yclass, fontsize=12)
    config = {"font.family": 'Times New Roman'} 
    rcParams.update(config)
    plt.imshow(proportion, interpolation='nearest', cmap=plt.cm.Blues) 
    plt.colorbar()

    thresh = confusion_matrix.max() / 2.

    iters = np.reshape([[[i, j] for j in range(length)] for i in range(length)], (confusion_matrix.size, 2))
    for i, j in iters:
        if i == j:
            # plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10, color='red',
            #          weight=5)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10, color='red')
        else:
            # plt.text(j, i + 0.12, format(confusion_matrix[i, j]), va='center', ha='center', fontsize=10)  
            plt.text(j, i - 0.12, pshow[i, j], va='center', ha='center', fontsize=10)

    plt.ylabel('True label', fontsize=16)
    plt.xlabel('Predict label', fontsize=16)
    plt.tight_layout()
    plt.savefig(work_dir+'/verification_confusion_matrix.jpg', dpi=1000)
    plt.show()

show_confusion_matrix(labels, cm, '20230410CM2')