import torch
import torchvision
import einops
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

# hyperparameters
batch_size: int = 16
in_channels: int = 3
image_size: int = 32
patch_size: int = 4
num_classes: int = 10
dim: int = 16
depth: int = 8
token_dim: int = 16
channel_dim: int = 16
epochs: int = 10

# blocks
class MlpBlock(nn.Module):
    def __init__(self, mlp_dim):
        super().__init__()
        self.mlp_dim: int = mlp_dim

    def forward(self, x):
        y = nn.Linear()
