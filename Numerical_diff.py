import numpy as np

def softmax(x):
    if x.ndim == 2:
        x = x - np.max(x, axis=0, keepdims=True)
        y = np.exp(x) / np.sum(np.exp(x), axis=0, keepdims=True)
        return y
    x = x - np.max(x)
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t ):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    if y.ndim == t.ndim:
        t = t.argmax(axis=1) # 找正确标签索引

    batch_size = y.shape[0] # 行
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size 


def numerial_gradient0(f, x):
    h = 1e-4 # 步长
    # return (f(x+h) - f(x-h)) / 2*h
    grad = np.zeros_like(x) # 全0数组和x形状相同

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h # float防止整数陷阱
        fxh1 = f(x)

        x[idx] = float(tmp_val) - h
        fxh2 = f(x)

        grad[idx] = fxh1 - fxh2 / 2*h

        x[idx] = tmp_val # 还原
    return grad


def numerical_gradient(f, x):
    h = 1e-4 # 步长
    grad = np.zeros_like(x) # 全零
    # np.nditer是矩阵走路机器人，flags=['multi_index]是精确坐标元组， op_flags=['readwriter']赋予机器人修改权限
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:
        idx = it.multi_index # 拿到当前的坐标，ex(0,1)
        temp_val = x[idx]

        x[idx] = float(temp_val) + h
        fx1 = f(x)

        x[idx] = temp_val - h
        fx2 = f(x)

        grad[idx] = (fx1 - fx2) / (2*h)
        x[idx] = temp_val # 还原
        it.iternext() # 走向下一个元素

    return grad


class simpleNet:
    def __init__(self): 
        self.W = np.random.randn(2,3)

    def forward(self,x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.forward(x)
        y = softmax(z) # 转概率
        loss = cross_entropy_error(y, t)

        return loss
    
x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simpleNet()

f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)

print(dW)

    





