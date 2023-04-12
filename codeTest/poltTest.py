import sys
sys.path.append("/home/jwk/Project/remoteSensing/Remote-Sensing-Image-Classification") 
from utils.plot import draw_loss
import random
#生成draw_loss函数需要的数据

arrayX1 = []
arrayY1 = []
for i in range(100):
    arrayX1.append(i)

for i in range(100):
    # 随机
    arrayY1.append(random.randint(0, 100))

draw_loss(arrayX1, arrayY1)
