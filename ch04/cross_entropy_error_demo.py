import sys, os
sys.path.append(os.pardir)
sys.path.append(os.curdir)
import numpy as np
from common.functions import cross_entropy_error

t = [0, 0, 1, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1]

error = cross_entropy_error(np.array(y), np.array(t))
print(error)

t = [0, 0, 0, 1]
y = [0.8, 1]

error = cross_entropy_error(np.array(y), np.array(t))
print(error)

# 展示y和t的关系
# y是预测结果，y.shape[0]代表多少组结果，应该和t的shape一致（与y的行数y.shape[0]一致)
# t是标签，t.size应该是1，如果不是1，则t的标签是one-hot形式的，可以使用t.argmax(axis=1)来获取结果
def sumlog(y, t):
    y = np.array(y)
    t = np.array(t)
    print(y)
    print(t)
    print(y.size)
    print(y.shape)
    print(t.shape[0])
    return np.sum(np.log(y[np.arange(y.shape[0]), t]))

y = [[1, 8, 4, 5],[3, 16, 5, 3], [1, 5, 7, 8]]
t = [1, 0, 2]

print(sumlog(y, t))

print('\n')

print(cross_entropy_error(np.array(y), np.array(t)))
