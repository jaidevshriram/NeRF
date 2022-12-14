import torch.nn as nn

class ShiftedSoftplus(nn.Module):
    def __init__(self, beta=1, threshold=20):
        super().__init__()

        self.softplus = nn.Softplus(beta=beta, threshold=threshold)

    def forward(self, x):
        return self.softplus(x - 1)
