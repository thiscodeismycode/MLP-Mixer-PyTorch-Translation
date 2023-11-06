import torch
import torchvision
import einops
import numpy as np
from torch import nn
from einops.layers.torch import Rearrange
import torchvision.transforms as transforms

# hyperparameters
"""batch_size: int = 16
in_channels: int = 3
image_size: int = 32
patch_size: int = 4
num_classes: int = 10
dim: int = 16
depth: int = 8
token_dim: int = 16
channel_dim: int = 16
epochs: int = 10"""

# blocks
class MlpBlock(nn.Module):
    def __init__(self, mlp_dim):
        super().__init__()
        self.mlp_dim: int = mlp_dim

    def forward(self, x):
        dim = x.shape()[-1]
        # y = nn.Dense(output_dim)(input)
        # FLAX nn.Dense() is a linear transformation applied over the last dimension of the input.
        y = nn.Linear(dim, self.mlp_dim)(x) # nn.Linear(input_dim, output_dim)(input)
        y = nn.GELU()(y)
        # Back to the original shape
        y = nn.Linear(y.shape()[-1], x.shape()[-1])
        return y

class MixerBlock(nn.Module):
    def __init(self, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim

    def forward(self, x):


class MlpMixer(nn.Module):
    def __init__(self, num_classes, num_blocks, patch_size, hidden_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.num_classes: int = num_classes
        self.num_blocks: int = num_blocks
        self.patch_size: int = patch_size
        self.hidden_dim: int = hidden_dim
        self.tokens_mlp_dim: int = tokens_mlp_dim
        self.channels_mlp_dim: int = channels_mlp_dim
