import torch.nn as nn
import torch
import torch.nn.functional as F


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        # nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)

EPS = 0.0003


class Actor(nn.Module):
    def __init__(self, input_size, out_size, max_u=1.0):
        super().__init__()
        hidden_size = 256
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)
        self.max_u = max_u
        # self.apply(weights_init)
        nn.init.uniform_(self.out.weight.data, a=-EPS, b=EPS)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.max_u * torch.tanh(self.out(x))
        

class Critic(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        hidden_size = 256
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        # self.apply(weights_init)
        nn.init.uniform_(self.out.weight.data, a=-EPS, b=EPS)

    def forward(self, s, a):
        x = torch.cat([s,a], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

        
