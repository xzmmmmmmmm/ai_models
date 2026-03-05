import numpy as np 
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from test import sigmoid, softmax

def get_data():
    #加载数据集
    data = pd.read_csv("/home/xzm/ai_models/train.csv", on_bad_lines='skip')
    #划分测试集和训练集
    X = data.drop("label", axis=1)#data是pandas的表格型数据结构，删除列为label，剩余数据赋值给X
    y = data["label"].values # 从data中提取名为label的列给y
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)#train_test_split是sklearn.model_selection中的函数，用于随机拆分数据
    # X，y待拆分的特征和标签数据
    # test_size=0.3代表测试集占总数据集30%
    
    # 归一化
    prepocessor = MinMaxScaler() #sklearn归一化工具,MinMaxSaler 把数据按比例压缩到0~1之间
    x_train= prepocessor.fit_transform(x_train)# fit_transform边制定标准边转换，transform严格按照已学到的旧标准直接转换
    x_test = prepocessor.transform(x_test)
    return x_test, y_test

def init_network():

        # 加载模型
        network = joblib.load("./nn_sample")
        return network
    

def forward(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100 # 批次
accuracy_cut = 0

for i in range(0, len(x), batch_size):
     x_batch = x[i:i+batch_size]
     y_batch = forward(network, x_batch)
     p = np.argmax(y_batch, axis=1)
     accuracy_cut += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cut) / len(x)))














