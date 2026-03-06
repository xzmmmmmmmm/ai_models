from test import *
from Numerical_diff import numerical_gradient,cross_entropy_error

class TwoLayerNet:

    def __init__(self, input_szie, hidden_size, output_size, weight_init_std=0.01):
        # 初始化权重
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_szie, hidden_size) # 标准正态分布
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def forward(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y
    
    # x:输入数据 ， t : 监督标签
    def loss(self, x, t):
        y = self.forward(x)

        return cross_entropy_error(y, t)
    
    def acc(self, x, t):
        y = self.forward(X)
        y = np.argmax(y, axis=1)
        t = t.reshape(-1)


        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    
