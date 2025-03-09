import torch
from torch import nn
from torch.onnx.symbolic_opset9 import tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt


from sklearn.datasets import load_iris

#数据准备
x,y = load_iris(return_X_y=True)
x=x[:100]
y=y[:100]
# 创建张量
tensor_x  = torch.tensor(x,dtype=torch.float32)
tensor_y = torch.tensor(y,dtype=torch.float32)

#参数设置
learning_rate = 0.001
epochs = 500
#模型参数
w = torch.randn(1,4 , requires_grad=True) #gradient=true 表示该张量需要 梯度计算， w.grad 默认值none 保存梯度值
b = torch.randn(1 , requires_grad=True) # bias
#前向计算
torch.li




##################预习内容##########################
train_data = datasets.FashionMNIST(
    root='../data',
    train=True, # 训练集
    download=True,
    transform=transforms.ToTensor())

test_data = datasets.FashionMNIST(
    root='../data',
    train=False, # 测试集
    download=True,
    transform=transforms.ToTensor()
)

#每个批次大小
batch_size = 64

#数据加载器
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size)

for x,y in test_loader :
    print("x.shape [N,C,H,W] = ",x.shape)
    print("y.shape =" , y.shape,y.dtype)


