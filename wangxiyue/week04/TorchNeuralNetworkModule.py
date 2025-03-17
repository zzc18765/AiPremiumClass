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
    return train_test_split(x,y,test_size=0.2,random_state=22,shuffle=True)

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
    print('load data ==> ',xTrain.shape, xTest.shape, yTrain.shape, yTest.shape)
    train_datasets = OlivettiDataset(xTrain,yTrain , transform=transform_train)
    test_datasets = OlivettiDataset(xTest, yTest)

    trainData = DataLoader(train_datasets, batch_size=batch_size, shuffle=True
                            ,generator=torch.Generator(device=device))
    testData = DataLoader(test_datasets, batch_size=batch_size, shuffle=True
                            ,generator=torch.Generator(device=device))
    return trainData, testData

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
        self.calc_model = nn.Sequential(

            nn.Linear(4096, 4096),

            nn.BatchNorm1d(4096),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Linear(2048, 1024),

            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Linear(512, 256),

            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Linear(128, 64),

            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 40),


        )


    def forward(self, x):
        y =self.calc_model(x)
        return y

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

# loss_fc = nn.CrossEntropyLoss()

if __name__ == '__main__':

    # check device
    device = check_device()
    torch.set_default_device(device)

    print('model initialized')
    model = TorchNeuralNetworkModule()  # make model obj
    model.to(device)
    loss_fc = nn.CrossEntropyLoss()
    ############################# 超参 #################################
    Learning_Rate = 0.005
    Epochs = 50
    BATCH_SIZE = 40
    Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate, momentum=0.85)
    # Optimizer = torch.optim.AdamW(model.parameters(),lr=Learning_Rate , betas=(0.8,0.9),weight_decay=0.01)

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

    for epoch in range(Epochs):
        train_loss = []
        model.train()
        for train_x, train_y in train_data:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            pred = model(train_x)
            loss = loss_fc(pred, train_y)
            model.zero_grad()
            loss.backward()
            Optimizer.step()
            train_loss.append(loss.item())
            total_train_step += 1
            if total_train_step % 10 == 0:
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

        print(f'acc={acc},test_data_size={test_data_size} | 测试集整体 ->{color.RED} Loss avg:{np.average(test_loss):.6f} , Acc :{(acc / test_data_size * 100):.3f}% {color.END}')
        print('')
        writer.add_scalar('test_loss_avg', np.average(test_loss), total_test_step)
        writer.add_scalar('test_acc', acc / test_data_size, total_test_step)
        total_test_step += 1
        torch.save(model, f'../data/model/torch_nn_module_{epoch + 1}.pth')
        writer.flush()
    writer.close()


