import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_olivetti_faces
from torch.utils.data import TensorDataset, DataLoader
from 邪王真眼.models.cnn import CNN2


def main():
    # hyperparameter
    lr = 0.0001
    epochs = 20
    bs = 4
    weight_decay = 1e-4

    dataset_path = "./邪王真眼/dataset"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset
    faces = fetch_olivetti_faces(data_home=dataset_path, shuffle=False)
    X = faces.images  # (400, 64, 64)
    y = faces.target  # (400,)

    X = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # (400, 1, 64, 64)
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
    model = CNN2().to(device)
    criterion = nn.CrossEntropyLoss()
    
    optimizer_sgd = optim.SGD(model.parameters(), lr=10 * lr, momentum=0.9, nesterov=True)
    optimizer_adam = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer_rmsprop = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.99, eps=1e-08)
    optimizer_adamw = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    optimizer = optimizer_adam

    # train
    max_correct = 0
    best_model = None
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0
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
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            print(f'\rBatch [{batch + 1}/{len(train_loader)+1}], Loss: {running_loss/len(train_loader):.4f}, Accuracy: {100.*correct/total:.2f}%', end='', flush=True)
            batch += 1

        # val
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        print(f'\n  VAL:   Accuracy: {100.*correct/total:.2f}%')
        if correct > max_correct:
            max_correct = correct
            best_model = copy.deepcopy(model)
            print(f'  VAL:   best model!')

        model.train()

    # save model
    save_path = f"./邪王真眼/week04/olivetti_CNN_acc_{100.*max_correct/total:.2f}.pth"
    dir_path = os.path.dirname(save_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    torch.save(best_model.state_dict(), save_path)

if __name__ == "__main__":
    main()