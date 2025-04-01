import torch.nn as nn
import torch.nn.init as init

class LogisticRegression(nn.Module):
    def __init__(self, input_size=30):
        super(LogisticRegression, self).__init__()
        
        self.bn = nn.BatchNorm1d(input_size)
        self.fc = nn.Linear(input_size, 1)
        self.sigmod = nn.Sigmoid()

        self._initialize_weights()

    def _initialize_weights(self):
        init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('sigmoid'))
        init.zeros_(self.fc.bias)

    def forward(self, x):
        x = self.bn(x)
        x = self.fc(x)
        x = self.sigmod(x)
        return x