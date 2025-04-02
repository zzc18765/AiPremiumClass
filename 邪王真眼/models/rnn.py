import torch.nn as nn


class RNNModel(nn.Module):
    def __init__(self, model_type='rnn', input_size=64, hidden_size=128, num_classes=40, 
                 num_layers=1, dropout=0.0):
        super(RNNModel, self).__init__()
        self.model_type = model_type.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        if self.model_type == 'rnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=False,
                             dropout=dropout if num_layers > 1 else 0.0)
        elif self.model_type == 'lstm':
            self.rnn = nn.LSTM(input_size, hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=False,
                              dropout=dropout if num_layers > 1 else 0.0)
        elif self.model_type == 'gru':
            self.rnn = nn.GRU(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=False,
                             dropout=dropout if num_layers > 1 else 0.0)
        elif self.model_type == 'birnn':
            self.rnn = nn.RNN(input_size, hidden_size, num_layers=num_layers,
                             batch_first=True, bidirectional=True,
                             dropout=dropout if num_layers > 1 else 0.0)
        else:
            raise ValueError(f"Unknown model type: {model_type}. "
                           f"Available options are: 'rnn', 'lstm', 'gru', 'birnn'")
        
        fc_input_size = hidden_size * (2 if model_type == 'birnn' else 1)

        self.fc = nn.Linear(fc_input_size, num_classes)
        
    def forward(self, x):
        out, _ = self.rnn(x)
        
        out = out[:, -1, :]
        
        out = self.fc(out)
        return out
