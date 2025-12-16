import torch.nn as nn

from zprp_project_25z_group_11.generators.components import Components

class Multiplication(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=2, hidden_size=93, batch_first=True)
        self.fc = nn.Linear(93, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        
        return self.fc(last)
    
    # todo: training loop, loss function, optimizer