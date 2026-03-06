import numpy as np
# from Numerical_diff import numerical_gradient
# def numerical_grad(f, x):

def numerical_gradient(f, x):
    h = 1e-4 # 步长
    grad = np.zeros_like(x) # 全零
    # np.nditer是矩阵走路机器人，flags=['multi_index]是精确坐标元组， op_flags=['readwriter']赋予机器人修改权限
    it = np.nditer(x, flags=['multi_index'],  op_flags=['readwrite'])
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


def gradient_descent(f, init_X, lr=0.01, step_num=100):
    x = init_X
    x_history = []
    for i in range(step_num):
        x_history.append(x.copy()) # 记录下山脚印

        grad = numerical_gradient(f,x) # 下山方向
        x -= lr * grad # 下山

    return x, np.array(x_history)

