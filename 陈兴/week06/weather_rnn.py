import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

# 2. 使用RNN实现一个天气预测模型，能预测1天和连续5天的最高气温。要求使用tensorboard，提交代码及run目录和可视化截图。
#   数据集：URL_ADDRESS   数据集：https://www.kaggle.com/datasets/smid80/weatherww2

# 加载数据
df = pd.read_csv("./陈兴/week06/data/weather.csv")

# 选择需要的列，例如日期和最高气温
df = df[["Date", "MaxTemp"]]

# 处理缺失值
df = df.dropna()

# 将日期转换为 datetime 格式
df["Date"] = pd.to_datetime(df["Date"])

# 按日期排序
df = df.sort_values(by="Date")

# 归一化最高气温
scaler = MinMaxScaler(feature_range=(0, 1))
df["MaxTemp"] = scaler.fit_transform(df[["MaxTemp"]])


# 定义序列数据生成函数
def create_sequences(data, seq_length, pred_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - pred_length + 1):
        seq = data[i : i + seq_length]
        label = data[i + seq_length : i + seq_length + pred_length]
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)


# 定义输入序列长度和预测长度
seq_length = 30  # 输入30天的数据
pred_length = 5  # 预测未来5天

# 生成序列数据
sequences, labels = create_sequences(
    df["MaxTemp"].values, seq_length, pred_length
)


class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=5):
        super(WeatherRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    sequences, labels, test_size=0.2, random_state=42
)

# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
y_test = torch.tensor(y_test, dtype=torch.float32)

# 创建数据加载器
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# 初始化模型、损失函数和优化器
model = WeatherRNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 初始化 TensorBoard
writer = SummaryWriter("./陈兴/week06/weather_runs/weather_prediction")

# 训练模型
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    for batch_X, batch_y in train_loader:
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 记录训练损失
    writer.add_scalar("Loss/train", loss.item(), epoch)

    # 在测试集上评估
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        test_loss = criterion(test_outputs, y_test)
        writer.add_scalar("Loss/test", test_loss.item(), epoch)

    print(
        f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}"
    )

# 保存模型
torch.save(model.state_dict(), "./陈兴/week06/models/weather_rnn.pth")

# 关闭 TensorBoard
writer.close()
