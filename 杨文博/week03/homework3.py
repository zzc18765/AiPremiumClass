import torch
from torch.utils.data import DataLoader
from torchvision.transforms.v2 import Compose, ToImage, ToDtype
from torchvision.datasets import KMNIST
import matplotlib.pyplot as plt
import ssl
import torch.nn as nn
ssl._create_default_https_context = ssl._create_unverified_context

transform = Compose([ToImage(), ToDtype(torch.float32, scale=True)])
# 加载数据
train_dataset = KMNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = KMNIST(root='./data', train=False, transform=transform, download=True)
data, lable = train_dataset[0]
print(data.shape)
print(type(data))
img, clz = train_dataset[2]
img = img.reshape(28,28)
plt.imshow(img,cmap="gray")
plt.show()
labels = set([clz for img,clz in train_dataset])
print(labels)

LR = 1e-1
epoches = 20
batch_size =256

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = nn.Sequential(
    nn.Linear(784, 128),  # 输入层到第一个隐藏层
    nn.Sigmoid(),         # 激活函数
    nn.Linear(128, 64),   # 第一个隐藏层到第二个隐藏层
    nn.Sigmoid(),         # 激活函数
    nn.Linear(64, 10)    # 第二个隐藏层到输出层
)

# 损失函数
loss_fn = nn.CrossEntropyLoss()
# 优化器
optimizer = torch.optim.SGD(model.parameters(),lr=LR)

for epoch in range(epoches):
    for data, target in train_loader:
        # 前向计算
        output = model(data.reshape(-1, 784))
        loss = loss_fn(output, target)  # 计算损失
        # 反向传播
        optimizer.zero_grad()  # 梯度归零
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        print(loss.item())

test_dl = DataLoader(test_dataset, batch_size=batch_size)
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_dl:
     output = model(data.reshape(-1,784))
     _,predicted = torch.max(output.data,1)
     total += target.size(0)
     correct += (predicted == target).sum().item()
print(f"Accuracy of the network on the 10000 test images: {100*correct/total}")