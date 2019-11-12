import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))   

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x) 

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T 

    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t)**2)


# y是预测结果，y.shape[0]代表多少组结果，应该和t的shape一致（与y的行数y.shape[0]一致)
# t是标签，t.size应该是1，如果不是1，则t的标签是one-hot形式的，可以使用t.argmax(axis=1)来获取结果
def cross_entropy_error(y, t):

    '''
    print("input y is: ", y)
    print("input t is: ", t)
    print("y.ndim is: ", y.ndim)
    '''

    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
     # 监督数据是one-hot-vector的情况下，转换为正确解标签的索引
    if t.size == y.size:
        t = t.argmax(axis=1)

    delta = 1e-7
    batch_size = y.shape[0]

    '''
    print("t.size is: ", t.size)
    print("batch size (y.shape[0]) is: ", batch_size)
    print("np.arange(batch_size) is: ", np.arange(batch_size))
    print("t now is: ", t)
    temp1 = y[np.arange(batch_size), t]
    print("y[np.arange(batch_size), t] is: ", temp1)
    temp2 = np.log(temp1 + delta)
    print("np.log(temp1 + delta) is ", temp2)
    '''

    return -np.sum(np.log(y[np.arange(batch_size), t] + delta)) / batch_size

