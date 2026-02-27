import numpy.random as random
import numpy as np


import torch
import torch.nn as nn
import torch.nn.functional as F


class GRN(nn.Module):
    """
    Global Response Normalization Layer
    Gx is the per-channel L2 magnitude of the feature map: it computes ||x||₂ over the spatial dimensions (here axes 2,3,4 for depth/height/width in 3D).
    Nx normalizes that response by the mean magnitude across channels, giving a channel-wise gain factor.
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        Nx = Gx / (Gx.mean(dim=1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x
