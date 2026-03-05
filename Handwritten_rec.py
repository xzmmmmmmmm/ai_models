import numpy as np # 科学计算的核心，专门做矩阵乘法和高级数学运算
import pandas as pd # 处理表格数据csv
import joblib # 专门用来保存和加载模型参数 nn_sample
from sklearn.model_selection import train_test_split # 用于数据集切分test train
from sklearn.preprocessing import MinMaxScaler # 用于归一化，把数据按比例压缩到0~1
from test import sigmoid, softmax

def get_data():
    # 1.读取数据，跳过格式损坏的脏数据行
    data = pd.read_csv("/home/xzm/ai_models/train.csv", on_bad_lines='skip')

    # 2.剥离特征和标签
    X = data.drop("label", axis=1) # X是特征，把为label的列剥离
    y = data["label"].to_numpy() # y是真实答案label,把pandas series(单列数据)变成纯粹的Numpy数组，方便后续切片

    # 3.划分数据集：30%test
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    # 4. 数据归一化
    prepocessor = MinMaxScaler()
    x_train  = prepocessor.fit_transform(x_train) # 观察训练数据集顶下压缩标准，并压缩训练集
    x_test = prepocessor.transform(x_test) #严格按照上面的标准，压缩测试集

    # 5. 返回考试卷
    # 这次只是做评估，所以只返回 x_test, y_test
    return x_test, y_test


def init_network():
    # 使用 joblib 直接从硬盘读取一个字典， 装满了训练好的参数
    network = joblib.load("./nn_sample") # ./代表当前目录 /代表根目录 ../代表上级目录
    return network


def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3) # 用sofamax 把最终的得分转化为每个类别的概率

    return y


# 1. 拿考卷和标准答案
x, t = get_data()
network = init_network()

# 2. 设置批次大小
batch_size = 100 # 批处理batch就是把多个数据拼成一个大型举证交给cpu/gpu进行并行计算，既可以极大突破单样本挨个计算的速度瓶颈，还可以让模型训练更稳定，准确
acc = 0 #记录总共对了多少到题

# 3. 开始循环批处理（Batch Processing)
# 假设有3000 道题 ，range(0, 3000, 100) 会产生 0 100 200...
for i in range(0, len(x), batch_size):

    # 切片拿题： 一次性抽出100道题（变成一个100行的矩阵）
    x_batch = x[i:i+batch_size]

    # 模型预测 ： 极其强大的numpy 会用底层的c语言并行计算，瞬间把100道题做完
    y_batch = forward(network, x_batch)

    # 找最高分， y_batch (100, n),n为类别数
    # np.argmax(y_batch, axis=1) 在每一行找到概率最大的那个索引，就是把位置为1（列）的位置消灭掉
    # 原来是（100，10），np.argmax(y_batch, axis=1)后就变为（100，）也就是p
    p = np.argmax(y_batch, axis=1)

    # 找答案对分
    acc += np.sum(p == t[i:i+batch_size])

# 4. 打印最终正确率

print("Accuracy:" + str(float(acc) / len(x)))

