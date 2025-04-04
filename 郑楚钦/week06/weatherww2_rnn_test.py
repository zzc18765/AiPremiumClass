from torch import nn
from torch.utils.tensorboard import SummaryWriter
from rnnClassifier import RNN_Classifier
import torch
import csv
import numpy as np
from tqdm import tqdm
import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


def trainAndTest(model_type: torch.nn.RNNBase, bidirectional: bool = False, epoches=100, lr=0.001, hidden_size: int = 128,  num_layers: int = 2):

    writer = SummaryWriter("logs")
    train_dl, test_dl = get_dataloaders()
    model_name = ('bi_' if bidirectional else '') + model_type.__name__
    model = RNN_Classifier(model_type, bidirectional, 1, hidden_size,
                           1, num_layers).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoches):
        model.train()
        for batch_idx, (features, target) in enumerate(train_dl):
            optimizer.zero_grad()
            ouput = model(features)
            loss = loss_fn(ouput, target.to(dtype=torch.long))
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            writer.add_scalars('loss', {model_name: loss.item()},
                               epoch * len(train_dl) + batch_idx)

        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            for features, target in test_dl:
                output = model.predict(features.squeeze())
                total += target.size(0)
                correct += (output == target).sum().item()
            accuracy = 100 * correct / total
            writer.add_scalars('accuracy', {model_name: accuracy}, epoch)
    torch.save(model, f'model_{model_name}.pth')
    torch.save(model.state_dict(), f'model_{model_name}_params.pth')
    writer.close()


class RNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn = torch.nn.RNN(
            input_size=1, hidden_size=20, num_layers=1, batch_first=True)
        self.fc = torch.nn.Linear(20, 1)

    def forward(self, x):
        x, h = self.rnn(x)
        y = self.fc(h)
        return y


def fit(model, dataloader, epochs=10):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    bar = tqdm(range(1, epochs+1))
    for epoch in bar:
        model.train()
        train_loss = []
        for batch in dataloader['train']:
            X, y, _ = batch
            X, y = X.to(device, dtype=torch.float)[:, np.newaxis], torch.tensor(
                y).to(device, dtype=torch.float)
            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
        model.eval()
        eval_loss = []
        with torch.no_grad():
            for batch in dataloader['eval']:
                X, y, _ = batch
                X, y = X.to(device, dtype=torch.float)[:, np.newaxis], torch.tensor(
                    y).to(device, dtype=torch.float)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                eval_loss.append(loss.item())
        bar.set_description(f"loss {np.mean(train_loss):.5f} val_loss {
                            np.mean(eval_loss):.5f}")


def predict(key, model, data):
    writer = SummaryWriter("run")
    model.eval()
    with torch.no_grad():
        for X, y, date in data:
            pred = model(torch.tensor([[X]]).to(device, dtype=torch.float))
            writer.add_scalars(
                key, {'y_hat': pred.item(), 'y': y}, X)
    writer.close()


with open('./week06/homework/Summary of Weather.csv', 'r') as f:
    reader = csv.reader(f)
    print('表头', reader.__next__())
    staDict = {}
    beginDate = datetime.datetime.strptime('1942-01-01', '%Y-%m-%d')
    for row in reader:
        date_arr = [int(n) for n in row[1].split('-')]
        curDate = datetime.datetime.strptime(row[1], '%Y-%m-%d') - beginDate
        date = curDate.days
        # print(f'STA {row[0]}, Date {date}, MaxTemp {row[4]}')
        arr = staDict.setdefault(row[0], [])
        arr.append([date, float(row[4]), date_arr[0] *
                   10000 + date_arr[1] * 100 + date_arr[2]])
    batch_size = 5
    for key in staDict:
        data = staDict[key]
        length = len(data)
        print(f'key {key}, data length {length}')
        train_len = int(length * 0.8)
        data.sort(key=lambda x: x[0])
        train_dl = torch.utils.data.DataLoader(
            data[:train_len], batch_size=batch_size)
        val_dl = torch.utils.data.DataLoader(
            data[train_len:], batch_size=batch_size)
        rnn = RNN()
        fit(rnn, {'train': train_dl, 'eval': val_dl}, epochs=100)
        y_pred = predict(key, rnn, data[train_len:])
