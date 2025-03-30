import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 1. 生成模拟天气数据（温度）
def generate_weather_data(days=100):
    np.random.seed(42)
    time = np.arange(days)
    # 正弦波模拟季节变化 + 随机噪声
    temperature = 10 * np.sin(2 * np.pi * time / 30) + 5 + np.random.normal(0, 2, days)
    return temperature.astype(np.float32)

# 2. 数据预处理
def create_dataset(data, window_size=7):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    
    X, y = [], []
    for i in range(len(data_normalized) - window_size):
        X.append(data_normalized[i:i+window_size])
        y.append(data_normalized[i+window_size])
    return torch.tensor(X).unsqueeze(-1), torch.tensor(y), scaler

# 3. 定义RNN模型
class WeatherRNN(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, output_size=1):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.rnn(x)  # out.shape: [batch, seq_len, hidden_size]
        return self.fc(out[:, -1, :])  # 只取最后一个时间步

# 4. 主程序
if __name__ == "__main__":
    # 生成数据
    temperature_data = generate_weather_data(days=365)
    window_size = 7
    X, y, scaler = create_dataset(temperature_data, window_size)
    
    # 划分训练集和测试集
    split = int(0.8 * len(X))
    X_train, y_train = X[:split], y[:split]
    X_test, y_test = X[split:], y[split:]
    
    # 初始化模型
    model = WeatherRNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 训练
    epochs = 100
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()
        
        if (epoch+1) % 20 == 0:
            print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    # 测试
    model.eval()
    with torch.no_grad():
        test_pred = model(X_test).squeeze()
    
    # 反归一化
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
    test_pred_actual = scaler.inverse_transform(test_pred.reshape(-1, 1)).flatten()
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_actual, label="True Temperature (°C)", color='blue', alpha=0.6)
    plt.plot(test_pred_actual, label="Predicted Temperature (°C)", color='orange', linestyle='--')
    plt.title("RNN Weather Prediction (7-Day Window)")
    plt.xlabel("Test Day Index")
    plt.ylabel("Temperature")
    plt.legend()
    plt.grid(True)
    plt.show()