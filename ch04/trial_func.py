import numpy as np

x = np.array([-3.0, 1.0, 2.0, 4.0])
print(x)

# enumerate 将列表元素进行索引（编号）输出
x_num = enumerate(x)
print(x_num)

for i, num in x_num:
    print(i, num)

print('*' * 100)

# demo of enumerate 函数
for i, num in enumerate(x, start=1):
    print(i, num)

# 研究np.random
W = np.random.randn(2, 3) # 返回2x3维的矩阵，符合正态分布
print(W)
