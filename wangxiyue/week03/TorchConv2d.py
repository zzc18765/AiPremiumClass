import time

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor
from  torchvision import  datasets
from  torch.utils.data import DataLoader

torch.set_default_device('cuda')

def loadData(batch_size):
    train_datasets = datasets.KMNIST( root='../data',train=True,download=True,transform=ToTensor())
    test_datasets = datasets.KMNIST( root='../data',train=False,download=True,transform=ToTensor())
    train_data = DataLoader(train_datasets, batch_size=batch_size, shuffle=True,generator=torch.Generator(device='cuda'))
    test_data = DataLoader(test_datasets, batch_size=batch_size, generator=torch.Generator(device='cuda'))
    return train_data, test_data

# Conv2d model
class Conv2dNet(nn.Module):

    def __init__(self):
        super(Conv2dNet, self).__init__()
        self.conv2d_module = nn.Sequential(

            nn.Conv2d(1, 128, kernel_size=1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(64*3, 10),
        )

    def forward( self,x):
            y = self.conv2d_module(x)
            return y

model = Conv2dNet()
model.cuda()
loss_fc = nn.CrossEntropyLoss()

############################# 超参 #################################

Learning_Rate = 0.01
Epochs = 10
BATCH_SIZE = 128
Optimizer = torch.optim.SGD(model.parameters(), lr=Learning_Rate)

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

    model.train()
    for train_x, train_y in train_data:
        train_x,train_y = train_x.cuda() , train_y.cuda()
        pred = model(train_x)
        loss = loss_fc(pred, train_y)
        model.zero_grad()
        loss.backward()
        Optimizer.step()

        total_train_step += 1
        if total_train_step % 100 == 0:
            end_time = time.time()
            print(f'Epoch:{epoch}/{Epochs} , 训练次数:{total_train_step} , Loss:{loss.item()}, 耗时:{end_time - start_time}')

    model.eval()
    total_test_loss= 0
    total_test_acc = 0
    with torch.no_grad():
        for data, labels in test_data:
            data, labels = data.cuda() , labels.cuda()
            pred_test = model(data)
            loss = loss_fc(pred_test, labels)
            total_test_loss += loss.item()
            acc = (pred_test.argmax(1) == labels).sum()
            total_test_acc += acc.item()
    print(f'测试集整体 -> Loss:{total_test_loss} , Acc:{total_test_acc/test_data_size}')
    writer.add_scalar('test_loss',total_test_loss,total_test_step)
    writer.add_scalar('test_acc', total_test_acc/test_data_size, total_test_step)
    total_test_step += 1
    torch.save(model,f'week02_conv2d_model_gpu{epoch}.pth')
writer.close()
