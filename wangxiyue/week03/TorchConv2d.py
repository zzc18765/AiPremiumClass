import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from  torchvision import  datasets
from  torch.utils.data import DataLoader

def check_device():
    if (torch.backends.mps.is_available()):
        device = torch.device('mps')
    elif (torch.cuda.is_available()):
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f'use {device} ')
    return device
# check device
device = check_device()

torch.set_default_device(device)

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

def loadData(batch_size):
    train_datasets = datasets.KMNIST( root='../data',train=True,download=True,transform=ToTensor())
    test_datasets = datasets.KMNIST( root='../data',train=False,download=True,transform=ToTensor())
    train_data = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,generator=torch.Generator(device=device))
    test_data = DataLoader(test_datasets, batch_size=batch_size, generator=torch.Generator(device=device))
    return train_data, test_data

# Conv2d model
class Conv2dNet(nn.Module):

    def __init__(self):
        super(Conv2dNet, self).__init__()

        self.flatten = nn.Flatten() # 展开所有张量为 1 维
        self.conv2d_module = nn.Sequential(
            nn.Conv2d(1, 36, kernel_size=1 , padding=0),
            nn.Conv2d(36, 36, kernel_size=(2,2), padding=0),
            nn.Conv2d(36, 1, kernel_size=(2,2),padding=0),
            nn.BatchNorm2d(1),
            nn.Flatten(),
            nn.Linear(676, 256),
            # nn.PReLU(),
            # nn.Linear(512, 256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 10),

        )

    def forward( self,x):
            # x=self.flatten(x)
            y = self.conv2d_module(x)
            return y

model = Conv2dNet()
model.to(device)
loss_fc = nn.CrossEntropyLoss()

############################# 超参 #################################

Learning_Rate = 0.005
Epochs = 10
BATCH_SIZE = 36
Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate,momentum=0.9)

###################################################################

total_train_step = 0
total_test_step = 0
writer = SummaryWriter('../data/logs')
start_time = time.time()


############################# loaddata ########################
train_data , test_data =  loadData(BATCH_SIZE)
train_data_size = len(train_data)
test_data_size = len(test_data)

#train
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
        if total_train_step % 500 == 0:
            end_time = time.time()
            print(f'Epoch:{epoch+1}/{Epochs} , 训练次数:{total_train_step} , Loss:{np.average(train_loss):.6f}, 耗时: {(end_time - start_time):.1f} 秒')

    model.eval()

    test_loss = []
    with (torch.no_grad()):
        for data, labels in test_data:
            data = data.to(device)
            labels = labels.to(device)
            pred_test = model(data)
            loss_test = loss_fc(pred_test, labels)
            test_loss.append(loss_test.item())
            acc = (pred_test.argmax(1) == labels).sum().item()
            test_data_size = labels.size(0)
            # total_test_acc += acc

    print(f'测试集整体 ->{color.RED} Loss avg:{np.average(test_loss):.6f} , Acc:{acc/test_data_size*100:.3f}% {color.END}')
    writer.add_scalar('test_loss_avg',np.average(test_loss),total_test_step)
    writer.add_scalar('test_acc', acc/test_data_size, total_test_step)
    total_test_step += 1
    torch.save(model,f'week02_conv2d_model_gpu{epoch+1}.pth')
writer.close()



