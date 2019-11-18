# coding: utf-8
import os
import sys
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
sys.path.append(os.curdir)
import numpy as np
from common.util import im2col

N = 1
C = 3
H = 7
W = 7
stride = 1
pad = 0


x1 = np.random.rand(N, C, H, W)
print(x1)

col1 = im2col(x1, 5,5,stride=1, pad=0)
print(col1.shape) # (9, 75)

x2 = np.random.rand(10,3,7,7) # 10个数据

col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape) # (90,75)
