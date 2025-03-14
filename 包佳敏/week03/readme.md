acc:0.7776 
model:
linear = nn.Linear(784, 128, bias=True)
act = nn.Sigmoid()
linear2 = nn.Linear(128, 64, bias=True)
linear3 = nn.Linear(64, 10, bias=True)
model = nn.Sequential(linear, act, linear2, act,linear3)
超参数：
epochs = 40 
batch_size = 32 
lr = 1e-2

acc:0.8494
model:
linear = nn.Linear(784, 128, bias=True)
act = nn.Sigmoid()
linear2 = nn.Linear(128, 64, bias=True)
linear3 = nn.Linear(64, 10, bias=True)
model = nn.Sequential(linear, act, linear2, linear3)
超参数：
epochs = 40 
batch_size = 32 
lr = 1e-2

实验下来，当学习率小的时候，收敛很慢，调整至1e-2会有明显收敛，