# src/ai/neural_network.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MancalaNet(nn.Module):
    def __init__(self, input_size=14, hidden_size=128, output_size=14):
        super(MancalaNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)  # No activation on output layer
        return F.softmax(x, dim=1)  # Probability distribution over moves
