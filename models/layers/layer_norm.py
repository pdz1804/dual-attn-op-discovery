from __future__ import division
import torch
import torch.nn as nn

class LayerNorm(nn.Module):
    def __init__(self, n_state, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(n_state))  # Learnable scale
        self.beta = nn.Parameter(torch.zeros(n_state))  # Learnable shift
        self.eps = eps                                  # eps avoids division by zero.

    def forward(self, x):
        # Normalizes the last dimension of the input tensor (hidden_dim).
        mu = x.mean(-1, keepdim=True)
        var = (x - mu).pow(2).mean(-1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return self.gamma * x + self.beta
    
    