import numpy as np

def mean_squred_error(y, t):
    return 0.5 * np.sum((y-t)**2)

def cross_entorpy_error(y, t):
    # 1. 维度处理， 如果一维ndim=1,变成1*N
    if y.ndim == 1:
        t = t.reshape(1, t.szie)
        y = y.reshape(1, y.szie)

    # 2.标签转换，将[0,0,1]转为 索引2
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0] # 获取这批数据有多少张图

    # np.arange(batch_size) 产生行号[0,1,2]
    # y[np.arange(batch_size), t] 从每一行预测中，只挑出正确标签 t 对应的概率
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size