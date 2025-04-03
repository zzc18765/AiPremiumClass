import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import copy
import uuid
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import TensorDataset, DataLoader
from 邪王真眼.models.rnn import RNNModel

uuid = uuid.uuid4().hex[:8]


def main():
    # hyperparameter
    lr = 0.0001
    epochs = 100
    bs = 4
    weight_decay = 1e-4
    model_type = 'birnn' # 'rnn' 'lstm' 'gru' 'birnn'

    dataset_path = "./邪王真眼/dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    writer = SummaryWriter(f'./邪王真眼/week06/runs/face')

    # dataset
    faces = fetch_olivetti_faces(data_home=dataset_path, shuffle=False)
    X = faces.images  # (400, 64, 64)
    y = faces.target  # (400,)

    mean = X.mean()
    std = X.std()
    X = (X - mean) / std

    X = torch.tensor(X, dtype=torch.float32)  # (400, 64, 64)
    X = X.permute(0, 2, 1)  # (400, 64, 64)
    y = torch.tensor(y, dtype=torch.long)

    # 7:3 for each person
    train_idx = np.array([i * 10 + j for i in range(40) for j in range(7)])
    test_idx = np.array([i * 10 + j for i in range(40) for j in range(7, 10)])

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=bs, shuffle=False)

    # model
    model = RNNModel(model_type=model_type, input_size=64, hidden_size=256, num_classes=40, num_layers=3, dropout=0.3).to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda iter: (1 - iter / epochs) ** 0.9
        )
    
    # train
    max_correct = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_correct, train_total = 0, 0
        batch = 1

        print(f"\nEpoch {epoch+1}/{epochs}")
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

            print(f'\rBatch [{batch + 1}/{len(train_loader)+1}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100.*train_correct/train_total:.2f}%', end='', flush=True)
            batch += 1
        
        avg_loss = running_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        writer.add_scalar(f'{model_type}_Loss_{uuid}/train', avg_loss, epoch)
        writer.add_scalar(f'{model_type}_Accuracy_{uuid}/train', train_acc, epoch)

        scheduler.step()

        # val
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100. * val_correct / val_total
        writer.add_scalar(f'{model_type}_Accuracy_{uuid}/val', val_acc, epoch)

        print(f'\n  VAL:   Accuracy: {100.*val_correct/val_total:.2f}%')
        if val_correct > max_correct:
            max_correct = val_correct
            best_model = copy.deepcopy(model)
            print(f'  VAL:   best model!')

        model.train()

    writer.close()
    # save model
    save_path = f"./邪王真眼/week06/results/face_{model_type}_acc_{100.*max_correct/val_total:.2f}.pth"
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(best_model.state_dict(), save_path)

if __name__ == "__main__":
    main()