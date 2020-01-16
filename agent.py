import torch
import torch.nn as nn
import torch.nn.functional as F

class Agent(nn.Module):
    def __init__(self, n, hidden):
        super().__init__()

        self.fc1 = nn.Linear(n**2, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.log_softmax(self.fc3(x), dim=-1)

