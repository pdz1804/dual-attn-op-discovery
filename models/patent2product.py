import torch.nn as nn

class Patent2Product(nn.Module):
    def __init__(self, dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 512),
            nn.ReLU(),
            nn.Linear(512, dim)
        )

    def forward(self, x):
        return self.net(x)
