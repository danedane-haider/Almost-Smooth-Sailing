import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA

class NeuralNet(nn.Module):
    def __init__(self, input_dim=28*28, size=1600):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_dim, size, bias=True)
        self.linear2 = nn.Linear(size, 10)

    def forward(self, x):
        x = F.sigmoid(self.linear1(x))
        x = F.log_softmax(self.linear2(x), dim=1)
        return x
    
class AutoEncoder(nn.Module):
    def __init__(self, input_dim=28*28, end_layer_size=256, mid_layer_size=32):
        super(AutoEncoder, self).__init__()
        self.linear1 = nn.Linear(input_dim, end_layer_size, bias=True)
        self.middle1 = nn.Linear(end_layer_size, mid_layer_size, bias=True)
        self.middle2 = nn.Linear(mid_layer_size, end_layer_size, bias=True)
        self.linear2 = nn.Linear(end_layer_size, input_dim, bias=True)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.middle1(x))
        x = F.relu(self.middle2(x))
        x = F.sigmoid(self.linear2(x))
        return x