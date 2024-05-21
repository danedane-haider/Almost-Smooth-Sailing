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


class SmoothSailing(nn.Module):
    def __init__(self, beta=0):
        super(SmoothSailing, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.beta = beta

    def forward(self, inputs, targets, W=None):
        base_loss = self.loss(inputs, targets)

        if W is None or self.beta == 0:
            return base_loss
        else:
            W_norm = LA.norm(W, 2)**2
            W_fro = LA.norm(W, 'fro')**2 / W.shape[0]
            reg_loss = base_loss + self.beta * 0.5 * (W_norm - W_fro)
            return base_loss, reg_loss
        
def kappa(W):
    W_max = LA.norm(W, 2)
    W_min = LA.norm(LA.pinv(W), 2)
    return W_max * W_min 