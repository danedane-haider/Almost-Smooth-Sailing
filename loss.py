import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.linalg as LA

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
            N = min(W.shape[1], W.shape[0])
            W_norm = LA.norm(W, 2)**2
            W_fro = LA.norm(W, 'fro')**2 / N
            reg_loss = base_loss + self.beta * 0.5 * (W_norm - W_fro)
            W_norm = LA.norm(W, 2)**2
            W_fro = LA.norm(W, 'fro')**2 / W.shape[1]
            reg_loss = base_loss + self.beta * 0.5 * (W_norm - W_fro)
            return base_loss, reg_loss
        
class SmoothSailingAE(nn.Module):
    def __init__(self, beta_end=0, beta_mid=0):
        super(SmoothSailingAE, self).__init__()
        self.loss = nn.MSELoss()
        self.beta_end = beta_end
        self.beta_mid = beta_mid

    def forward(self, inputs, targets, W1=None, M1=None, M2=None, W2=None):
        base_loss = self.loss(inputs, targets)

        if W1 is None or self.beta_end == 0:
            return base_loss
        else:
            N = min(W1.shape[1], W1.shape[0])
            M = min(M1.shape[1], M1.shape[0])
            W1_norm = LA.norm(W1, 2)**2
            W1_fro = LA.norm(W1, 'fro')**2 / N
            M1_norm = LA.norm(M1, 2)**2
            M1_fro = LA.norm(M1, 'fro')**2 / M
            M2_norm = LA.norm(M2, 2)**2
            M2_fro = LA.norm(M2, 'fro')**2 / M
            W2_norm = LA.norm(W2, 2)**2
            W2_fro = LA.norm(W2, 'fro')**2 / N
            reg_loss = base_loss + self.beta_end * (W1_norm + W2_norm - W1_fro - W2_fro) + self.beta_mid * (M1_norm + M2_norm - M1_fro - M2_fro)
            return base_loss, reg_loss
        
def kappa(W):
    W_max = LA.norm(W, 2).item()
    W_min = LA.norm(LA.pinv(W), 2).item()
    return W_max * W_min 