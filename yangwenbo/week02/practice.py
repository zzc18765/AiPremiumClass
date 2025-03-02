import numpy as np
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')  # 或者 'Qt5Agg', 'Agg' 等
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# 梯度下降
plot_x = np.linspace(-1, 6, 100)
plot_y = (plot_x - 2.5) ** 2 - 1

def loss(x):
    try:
        return (x-2.5)**2 -1
    except:
        return float("inf")
def derivate(x):
    return 2*(x -2.5)

eta = 0.1
theta = 0

theta_history = []
while True:
    last_theta = theta
    theta -= eta * derivate(theta)
    loss_v = loss(theta)
    theta_history.append(theta)
    if abs(loss_v - loss(last_theta)) < 1e-9:
        break



plt.plot(plot_x, plot_y)
plt.plot(theta_history,loss(np.array(theta_history)),color='r',marker="+")
plt.show(block=False)

# 数据生成
# make_classification
X,y = make_classification(n_samples=100,n_features=10)
"""X 是特征矩阵（也称为输入数据），形状为 (n_samples, n_features)，其中：

n_samples 是样本数量。

n_features 是每个样本的特征数量。

y 是目标标签（也称为输出数据），形状为 (n_samples,)，表示每个样本的类别标签（通常是 0 或 1，表示二分类问题）。"""
print(X)
print(y)

# 数据拆分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
print(X_train.shape)
print(y_train.shape)

# 权重参数
theta = np.random.randn(1,10)
bias = 0
# 超参数
lr = 0.01
epochs = 3000

# 模型运算
# 计算运算函数
def forward(X, theta, bias):
    # 线性运算
    z = np.dot(theta, X.T) + bias
    # sigmoid
    y_hat = 1 / (1 + np.exp(-z))
    return y_hat
# 计算损失函数
def loss(y_hat, y):
    e = 1e-8
    return -y * np.log(y_hat+e) - (1 - y) * np.log(1 - y_hat+e)
# 计算梯度
def gradient(x,y,y_hat):
    m = x.shape[-1]
    delta_theta =np.dot((y_hat - y), x) / m
    delta_bias = np.mean(y_hat - y)
    return delta_theta, delta_bias
# 模型训练
for epoch in range(epochs):
    # 前向传播
    y_hat = forward(X_train, theta, bias)
    # 损失计算
    current_loss = loss(y_hat, y_train)
    # 梯度计算
    delta_theta, delta_bias = gradient(X_train, y_train, y_hat)
    # 参数更新
    theta = theta - lr * delta_theta
    bias = bias - lr * delta_bias
    accuracy = np.mean(np.round(y_hat) == y_train)
    if epoch % 100 == 0:
        print(accuracy)

# 模型推理
idx = np.random.randint(len(X_test))
x = X_test[idx]
y = y_test[idx]
prediction = np.round(forward(x, theta, bias))
print(prediction,y)

"""作业
使用sklearn数据集训练逻辑回归模型
调整学习率，观察训练结果
把模型训练的好的参数进行保存在另一个文件中打开来实现预测功能"""
# sklearn中iri数据集
from sklearn.datasets import load_iris
X,y = load_iris(return_X_y=True)
print(X.shape,y.shape)
print(x[:100])