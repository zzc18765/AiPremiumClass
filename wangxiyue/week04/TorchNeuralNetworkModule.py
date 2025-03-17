###########################
# 本周作业
# 1 搭建神经网络、使用olivettiface （scikit-learn） 数据进行训练
# 2 结合归一化和正则化优化网络模型 ，观察对比loss结果
# 3 尝试不同optimizer 训练模型，对比loss结果
# 4 注册kaggle 并尝试激活Accelerator 使用gpu加速
###########################
import os
import shutil
import time

import numpy as np
import torch
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms


def check_device():
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
    elif (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'use {device} ')
    return device


##############
###olivetti###
# data (400,4096)
# target(400,)
# images(400,64,64)
#
def load_data_olivettiface():
    x,y = fetch_olivetti_faces(data_home='../data/face_data',return_X_y=True)
    return train_test_split(x,y,test_size=0.2,shuffle=True)

def data_loader(batch_size):
    transform_train = transforms.Compose([
        transforms.RandomRotation(16),
        transforms.RandomAffine(7, translate=(0.11, 0.13), shear=0.16),
        transforms.ToTensor()
    ])
    ######
    transform_train = None
    ######

    xTrain, xTest, yTrain, yTest = load_data_olivettiface()
    print(xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
    train_datasets = OlivettiDataset(xTrain,yTrain , transform=transform_train)
    test_datasets = OlivettiDataset(xTest, yTest)

    train_data = DataLoader(train_datasets, batch_size=batch_size, shuffle=True
                            , num_workers=8, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4)  # ,generator=torch.Generator(device=device))
    test_data = DataLoader(test_datasets, batch_size=batch_size, shuffle=True
                            , num_workers=8, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4)  # ,generator=torch.Generator(device=device))
    return train_data, test_data

class OlivettiDataset(Dataset):
    def __init__(self,x,y,transform=None):
        # super(OlivettiDataset,self).__init__()
        self.x = x
        self.y = y
        if transform is not None:
            self.transform = transform
        else :
            self.transform = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self,idx):
        data = self.x[idx]
        target = self.y[idx]
        if self.transform :
            data = self.transform(data[idx])
            target=self.transform(target[idx])
        else:
            data =torch.tensor(data)
            target = torch.tensor(target)
        return data, target

class TorchNeuralNetworkModule(nn.Module):
    def __init__(self):
        super(TorchNeuralNetworkModule,self).__init__()
        self.linear1 = nn.Linear(4096, 512)
        self.batchnorm1d_1 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 512)
        self.batchnorm1d_2 = nn.BatchNorm1d(512)
        self.linear3 = nn.Linear(512, 10)
        self.drop = nn.Dropout1d(p=0.1)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.linear1(x)
        x = self.batchnorm1d_1(x)

        x = self.linear2(x)
        x = self.batchnorm1d_2(x)

        x = self.act(x)
        x = self.drop(x)

        x = self.linear3(x)
        return x

class color:
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    DARKCYAN = '\033[36m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

loss_fc = nn.CrossEntropyLoss()

if __name__ == '__main__':

    # check device
    device = check_device()
    torch.set_default_device(device)

    print('model test!!!')
    model = TorchNeuralNetworkModule()  # make model obj
    # loss_fc = nn.CrossEntropyLoss()
    ############################# 超参 #################################
    Learning_Rate = 0.005
    Epochs = 20
    BATCH_SIZE = 10
    Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, momentum=0.85)
    ############################ 配置信息 ################################
    LOG_DIR = '../data/logs_olivetti'
    if os.path.exists(LOG_DIR):
        shutil.rmtree(LOG_DIR)

    total_train_step = 0
    total_test_step = 0

    writer = SummaryWriter(LOG_DIR)
    start_time = time.time()
    ############################# loaddata ########################
    train_data, test_data = data_loader(BATCH_SIZE)
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    model.to(device)
    for epoch in range(Epochs):
        train_loss = []
        model.train()
        for train_x, train_y in train_data:
            print(train_x.shape)
            print(train_y)

            train_x = train_x.to(device)
            train_y = train_y.to(device)
            pred = model(train_x)
            loss = loss_fc(pred, train_y)
            model.zero_grad()
            loss.backward()
            Optimizer.step()
            train_loss.append(loss.item())
            total_train_step += 1
            if total_train_step % 100 == 0:
                end_time = time.time()
                print(
                    f'Epoch:{epoch + 1}/{Epochs} , 训练次数:{total_train_step} , Loss:{np.average(train_loss):.6f}, 耗时: {(end_time - start_time):.1f} 秒')

        model.eval()

        acc = 0
        test_data_size = 0
        test_loss = []
        with (torch.no_grad()):
            for data, labels in test_data:
                data = data.to(device)
                labels = labels.to(device)
                pred_test = model(data)
                loss_test = loss_fc(pred_test, labels)
                test_loss.append(loss_test.item())
                acc += (pred_test.argmax(1) == labels).sum().item()
                test_data_size += labels.size(0)

                # writer.add_image('Un Pred Img',data[~torch.isin(labels, pred_test.argmax(1))])
                # total_test_acc += acc

        print(
            f'acc={acc},test_data_size={test_data_size} | 测试集整体 ->{color.RED} Loss avg:{np.average(test_loss):.6f} , Acc :{(acc / test_data_size * 100):.3f}% {color.END}')
        print('')
        writer.add_scalar('test_loss_avg', np.average(test_loss), total_test_step)
        writer.add_scalar('test_acc', acc / test_data_size, total_test_step)
        total_test_step += 1
        torch.save(model, f'../data/model/week02_conv2d_model_gpu{epoch + 1}.pth')
        writer.flush()
    writer.close()


