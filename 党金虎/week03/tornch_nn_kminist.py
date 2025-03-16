import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 定义改进的神经网络模型
class ImprovedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ImprovedMLP, self).__init__()
        # 输入层 -> 隐藏层 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        # 隐藏层 1 -> 隐藏层 2
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # 隐藏层 2 -> 输出层
        self.fc3 = nn.Linear(hidden_size, num_classes)
        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # 输入层 -> 隐藏层 1
        x = self.fc1(x)
        x = self.relu(x)
        # 隐藏层 1 -> 隐藏层 2
        x = self.fc2(x)
        x = self.relu(x)
        # 隐藏层 2 -> 输出层
        x = self.fc3(x)
        return x

# 定义超参数
input_size = 28 * 28  # 输入特征维度（KMNIST 图像大小为 28x28）
hidden_size = 256      # 隐藏层维度（增加神经元数量）
num_classes = 10       # 输出类别数（KMNIST 有 10 个类别）
learning_rate = 0.001  # 学习率
num_epochs = 5         # 训练轮数
batch_size = 128       # 批量大小（调整批量大小）

# 加载 KMNIST 数据集
transform = transforms.Compose([
    transforms.ToTensor(),  # 将图像转换为张量
    transforms.Normalize((0.5,), (0.5,))  # 标准化
])

train_dataset = datasets.KMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.KMNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 实例化模型
model = ImprovedMLP(input_size, hidden_size, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # 交叉熵损失
optimizer = optim.Adam(model.parameters(), lr=learning_rate)  # Adam 优化器

# 训练模型
total_steps = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 将图像展平为向量
        images = images.reshape(-1, 28 * 28)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()  # 清空梯度
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新参数

        # 打印训练进度
        if (i + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

# 测试模型
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        # 将图像展平为向量
        images = images.reshape(-1, 28 * 28)
        # 前向传播
        outputs = model(images)
        # 获取预测结果
        _, predicted = torch.max(outputs.data, 1)
        # 统计正确预测的数量
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f"测试集准确率: {100 * correct / total:.2f}%")