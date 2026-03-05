import numpy as np

def step_function(x):
    return np.array(x > 0, dtype=int)

x = np.array([1.0, 2.0, -1.0, 0.0])
y = step_function(x)
# print(y)

def sigmoid(x):
    return 1 / (1 +  np.exp(-x))

y1 = sigmoid(x)
# print(y1)

def relu(x):
    return np.maximum(0, x)

y2 = relu(x)
# print(y2)

def softmax0(x):
    return np.exp(x) / np.sum(np.exp(x))#softmax 会让任意实数向量转换为一个概率分布，确保输出值总合为一
#且输入较大的值对应的输出概率较大，较小的值会被压缩

def softmax1(x):
    x = x - np.max(x) # 溢出对策，将值压缩到 0~1之间避免值过大
    return np.exp(x) / np.sum(np.exp(x))

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)#按列压缩 在 NumPy 中，纯粹的一维数组 (2,) 是没有方向的，它既不是行也不是列，只是一串数字
        y = np.exp(x) / np.sum(np.exp(x), axis=0) # 广播机制有一条铁律：如果两个数组的维度数不同，就在维度较少的数组的前面（左边）补 1。
        return y.T
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def softmax_better(x):
    if x.dim == 2:
        x = x - np.max(x, axis=1, keepdims=True) # 直接按行压缩
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)# 按行求最大值，保持形状，再按行sofamax
    
    x = x - np.max(x) # 溢出对策
    return np.exp(x) / np.sum(np.exp(x))

def identity_function(x):
    return x

if __name__ == '__main__':
    x_test = np.array([1.0, 2.0, -1.0, 0.0])
    
    y_step = step_function(x_test)
    print("Step:", y_step)

    y_sig = sigmoid(x_test)
    print("Sigmoid:", y_sig)

    y_relu = relu(x_test)
    print("ReLU:", y_relu)





